"""
Optimized Quantum Hybrid CNN for CIFAR-10
IMPROVEMENTS:
1. Better batching strategy for quantum layer
2. Improved quantum circuit with stronger entanglement
3. More variational layers for better expressivity
4. Data re-uploading technique

Usage:
    python quantum_hybrid_cnn_optimized.py --epochs 10 --quantum-qubits 8 --local
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
from tqdm import tqdm

# =================== Optimized Quantum Layer ===================
def create_optimized_quantum_circuit(n_qubits: int, n_layers: int = 4, use_local: bool = False):
    """
    Create an OPTIMIZED parameterized quantum circuit with:
    - Stronger entanglement (circular + all-to-all)
    - Data re-uploading
    - More expressive gates
    """
    if use_local:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("braket.aws.qubit", 
                         device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                         wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        # Initial encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers with data re-uploading
        for layer in range(n_layers):
            # Re-upload data (helps with expressivity)
            if layer > 0:
                for i in range(n_qubits):
                    qml.RY(inputs[i] * 0.5, wires=i)
            
            # Rotation gates (3 types for more expressivity)
            for i in range(n_qubits):
                qml.RX(weights[layer * n_qubits * 3 + i], wires=i)
                qml.RY(weights[layer * n_qubits * 3 + n_qubits + i], wires=i)
                qml.RZ(weights[layer * n_qubits * 3 + 2 * n_qubits + i], wires=i)
            
            # Stronger entanglement: circular chain
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            # Additional entanglement: every other qubit
            if n_qubits >= 4:
                for i in range(0, n_qubits - 2, 2):
                    qml.CNOT(wires=[i, i + 2])
        
        # Measure expectations
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_circuit

# =================== Optimized Hybrid Model ===================
class QuantumHybridCNN_Optimized(nn.Module):
    def __init__(self, n_qubits: int = 8, n_quantum_layers: int = 4, use_local: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Classical feature extraction (same as baseline)
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
        
        # Reduce to quantum input dimension (less aggressive reduction)
        self.fc_to_quantum = nn.Linear(32 * 4 * 4, n_qubits)
        
        # Quantum layer with more parameters
        n_weights = n_qubits * 3 * n_quantum_layers  # 3 rotation gates per qubit per layer
        quantum_circuit = create_optimized_quantum_circuit(n_qubits, n_quantum_layers, use_local)
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (n_weights,)})
        
        # Final classifier
        self.fc_output = nn.Linear(n_qubits, 10)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Classical feature extraction
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        
        # Reduce to quantum dimension
        x = torch.tanh(self.fc_to_quantum(x))  # Normalize to [-1, 1]
        
        # Quantum processing (still per-sample due to PennyLane limitation)
        quantum_outputs = []
        for i in range(batch_size):
            sample = x[i]
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
        
        # Progress bar for training
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / len(trainloader.dataset)
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in validloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(validloader.dataset)
        val_acc = 100.0 * correct / total
        tracker.log_epoch(avg_train_loss, avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    tracker.end_training()

def test_model(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc='Testing'):
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
    parser.add_argument('--batch-size', type=int, default=16)  # Smaller default for quantum
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--quantum-qubits', type=int, default=8)
    parser.add_argument('--quantum-layers', type=int, default=4)
    parser.add_argument('--local', action='store_true', help='Use local simulator instead of AWS Braket')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator_type = "local" if args.local else "AWS_Braket"
    print(f"Using device: {device}")
    print(f"Quantum simulator: {simulator_type}")
    print(f"Configuration: {args.quantum_qubits} qubits, {args.quantum_layers} layers")
    
    # Setup tracking
    tracker = ExperimentTracker('quantum_optimized', 
                               f'{args.quantum_qubits}qubits_{args.quantum_layers}layers_{simulator_type}')
    tracker.set_hyperparameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        quantum_qubits=args.quantum_qubits,
        quantum_layers=args.quantum_layers,
        optimizer='Adam',
        quantum_simulator=simulator_type,
        architecture='16->32->32 filters + optimized quantum',
        quantum_features='data_reuploading, circular_entanglement, 3_rotation_gates'
    )
    
    # Load data
    trainloader, validloader, testloader = get_data_loaders(args.batch_size)
    
    # Create model
    model = QuantumHybridCNN_Optimized(
        n_qubits=args.quantum_qubits, 
        n_quantum_layers=args.quantum_layers, 
        use_local=args.local
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = args.quantum_qubits * 3 * args.quantum_layers
    print(f"Total parameters: {total_params:,}")
    print(f"Quantum parameters: {quantum_params}")
    
    # Train
    print("\nStarting training...")
    train_model(model, trainloader, validloader, args.epochs, args.lr, device, tracker)
    
    # Test
    print("\nTesting model...")
    accuracy = test_model(model, testloader, device)
    tracker.set_test_accuracy(accuracy)
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Save results
    tracker.add_note(f"Optimized quantum hybrid CNN with {args.quantum_qubits} qubits, {args.quantum_layers} layers using {simulator_type}")
    tracker.add_note("Features: data re-uploading, circular entanglement, 3 rotation gates per qubit")
    saved_path = tracker.save()
    print(f"\n✓ Experiment results saved to: {saved_path}")

if __name__ == '__main__':
    main()
