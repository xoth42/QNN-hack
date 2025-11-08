"""
Quantum Hybrid CNN for CIFAR-10
Integrates quantum layer into classical CNN architecture

Usage:
    python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import argparse
from track_performance import ExperimentTracker

# =================== Quantum Layer Definition ===================
def create_quantum_circuit(n_qubits: int, n_layers: int = 2, use_local: bool = False):
    """
    Create a parameterized quantum circuit
    Args:
        n_qubits: Number of qubits (should match input dimension)
        n_layers: Number of variational layers
        use_local: If True, use local simulator instead of AWS Braket
    """
    if use_local:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("braket.aws.qubit", 
                         device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                         wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        # Encode classical data into quantum state
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        for layer in range(n_layers):
            # Rotation gates
            for i in range(n_qubits):
                qml.RY(weights[layer * n_qubits * 2 + i], wires=i)
                qml.RZ(weights[layer * n_qubits * 2 + n_qubits + i], wires=i)
            
            # Entanglement (linear chain)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Measure expectations
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_circuit

# =================== Hybrid Model ===================
class QuantumHybridCNN(nn.Module):
    def __init__(self, n_qubits: int = 4, n_quantum_layers: int = 2, use_local: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Classical feature extraction (match teammate's architecture)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Reduce to quantum input dimension
        self.fc_to_quantum = nn.Linear(32 * 4 * 4, n_qubits)
        
        # Quantum layer
        n_weights = n_qubits * 2 * n_quantum_layers
        quantum_circuit = create_quantum_circuit(n_qubits, n_quantum_layers, use_local)
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (n_weights,)})
        
        # Final classifier
        self.fc_output = nn.Linear(n_qubits, 10)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Classical feature extraction
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        
        # Reduce to quantum dimension
        x = torch.tanh(self.fc_to_quantum(x))  # Normalize to [-1, 1] for quantum encoding
        
        # Quantum processing (process each sample individually due to PennyLane batching)
        quantum_outputs = []
        for i in range(batch_size):
            sample = x[i]  # Get single sample (1D tensor)
            q_out = self.quantum_layer(sample)
            quantum_outputs.append(q_out)
        x = torch.stack(quantum_outputs, dim=0)
        
        # Final classification
        x = self.fc_output(x)
        return x

# =================== Training & Evaluation ===================
def get_data_loaders(batch_size: int = 32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_size = 5000
    train_subset, valid_subset = torch.utils.data.random_split(
        full_train, [len(full_train) - valid_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, validloader, testloader

def train_model(model, trainloader, validloader, epochs, lr, device, tracker):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    tracker.start_training()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = running_loss / len(trainloader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        avg_val_loss = val_loss / len(validloader.dataset)
        tracker.log_epoch(avg_train_loss, avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    tracker.end_training()

def test_model(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--quantum-qubits', type=int, default=4)
    parser.add_argument('--quantum-layers', type=int, default=2)
    parser.add_argument('--local', action='store_true', help='Use local simulator instead of AWS Braket')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator_type = "local" if args.local else "AWS Braket"
    print(f"Using device: {device}")
    print(f"Quantum simulator: {simulator_type}")
    
    # Setup tracking
    tracker = ExperimentTracker('quantum', f'{args.quantum_qubits}qubits_{args.quantum_layers}layers_{simulator_type.replace(" ", "_")}')
    tracker.set_hyperparameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        quantum_qubits=args.quantum_qubits,
        quantum_layers=args.quantum_layers,
        optimizer='Adam',
        quantum_simulator=simulator_type,
        architecture='16->32->32 filters (matching classical)'
    )
    
    # Load data
    trainloader, validloader, testloader = get_data_loaders(args.batch_size)
    
    # Create model
    model = QuantumHybridCNN(n_qubits=args.quantum_qubits, n_quantum_layers=args.quantum_layers, use_local=args.local).to(device)
    print(f"Model created with {args.quantum_qubits} qubits and {args.quantum_layers} quantum layers")
    
    # Train
    train_model(model, trainloader, validloader, args.epochs, args.lr, device, tracker)
    
    # Test
    accuracy = test_model(model, testloader, device)
    tracker.set_test_accuracy(accuracy)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save results
    tracker.add_note(f"Quantum hybrid CNN with {args.quantum_qubits} qubits using {simulator_type}")
    saved_path = tracker.save()
    print(f"\n✓ Experiment results saved to: {saved_path}")

if __name__ == '__main__':
    main()
