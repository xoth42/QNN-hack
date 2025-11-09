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
from time import time
from tqdm import tqdm
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
    epoch_times = []
    best_val_acc = 0.0
    
    tracker.start_training()
    
    print(f"\n{'='*70}")
    print(f"Starting Quantum Hybrid Training: {epochs} epochs on {device}")
    print(f"[NOTE] Quantum layer processing may be slower than classical CNN")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        epoch_start_time = time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                         leave=False, ncols=100, colour='green')
        
        for inputs, targets in train_pbar:
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
                
                # Update progress bar with current metrics
                current_loss = running_loss / total_train
                current_acc = 100. * correct_train / total_train
                train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ CUDA out of memory! Try reducing batch size (current: {inputs.size(0)})")
                    print(f"   Suggestion: Use --batch-size 16 or --batch-size 8 for quantum models")
                    raise
                elif "PennyLane" in str(e) or "quantum" in str(e).lower():
                    print(f"\n❌ Quantum circuit error: {e}")
                    print(f"   Suggestions:")
                    print(f"   - Check AWS credentials if using Braket (add --local to use simulator)")
                    print(f"   - Reduce number of qubits with --quantum-qubits")
                    print(f"   - Verify PennyLane installation: pip install pennylane")
                    raise
                else:
                    raise
        
        avg_train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100. * correct_train / total_train
        
        # Validation loop with progress bar
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(validloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", 
                       leave=False, ncols=100, colour='blue')
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                try:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()
                    
                    # Update progress bar
                    current_val_loss = val_loss / total_val
                    current_val_acc = 100. * correct_val / total_val
                    val_pbar.set_postfix({'loss': f'{current_val_loss:.4f}', 'acc': f'{current_val_acc:.2f}%'})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n❌ CUDA out of memory during validation!")
                        print(f"   Suggestion: Reduce batch size or use CPU")
                        raise
                    else:
                        raise
        
        avg_val_loss = val_loss / len(validloader.dataset)
        val_acc = 100. * correct_val / total_val
        
        # Calculate epoch time and estimate remaining time
        epoch_time = time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        estimated_time_remaining = avg_epoch_time * remaining_epochs
        
        tracker.log_epoch(avg_train_loss, avg_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s | ETA: {estimated_time_remaining/60:.1f}min")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), 'best_quantum_model.pth')
                print(f"  [OK] New best model saved (Val Acc: {val_acc:.2f}%)")
            except Exception as e:
                print(f"  [WARN] Warning: Could not save model checkpoint: {e}")
    
    tracker.end_training()
    
    total_time = sum(epoch_times)
    print(f"\n{'='*70}")
    print(f"Training Complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")

def test_model(model, testloader, device):
    try:
        model.load_state_dict(torch.load('best_quantum_model.pth'))
        print("[OK] Loaded best model checkpoint for testing")
    except FileNotFoundError:
        print("[WARN] Warning: No saved model found, using current model state")
    except Exception as e:
        print(f"[WARN] Warning: Could not load model checkpoint: {e}")
    
    model.eval()
    correct, total = 0, 0
    
    print("\nRunning final test evaluation...")
    test_pbar = tqdm(testloader, desc="Testing", ncols=100, colour='cyan')
    
    with torch.no_grad():
        for inputs, targets in test_pbar:
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update progress bar with running accuracy
                current_acc = 100. * correct / total
                test_pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ CUDA out of memory during testing!")
                    print(f"   Suggestion: Reduce batch size or use CPU")
                    raise
                elif "PennyLane" in str(e) or "quantum" in str(e).lower():
                    print(f"\n❌ Quantum circuit error during testing: {e}")
                    print(f"   Suggestion: Verify quantum layer configuration")
                    raise
                else:
                    raise
    
    accuracy = 100.0 * correct / total
    print(f"\n{'='*70}")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")
    return accuracy

def main():
    parser = argparse.ArgumentParser(
        description='Train Quantum Hybrid CNN on CIFAR-10',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with local simulator (recommended for testing)
  python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4 --local
  
  # Train with AWS Braket (requires credentials)
  python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4
  
  # Smaller batch size for memory constraints
  python quantum_hybrid_cnn.py --epochs 10 --batch-size 16 --local
        """
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32, try 16 or 8 if OOM)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--quantum-qubits', type=int, default=4, help='Number of qubits (default: 4, range: 2-8)')
    parser.add_argument('--quantum-layers', type=int, default=2, help='Number of quantum layers (default: 2)')
    parser.add_argument('--local', action='store_true', help='Use local simulator instead of AWS Braket')
    args = parser.parse_args()
    
    # Validate arguments
    if args.quantum_qubits < 2 or args.quantum_qubits > 16:
        print(f"[WARN] Warning: Unusual qubit count ({args.quantum_qubits}). Recommended range: 2-8")
        print(f"   Higher qubit counts may be very slow or cause memory issues")
    
    if args.batch_size > 64:
        print(f"[WARN] Warning: Large batch size ({args.batch_size}) may cause memory issues with quantum layers")
        print(f"   Suggestion: Try --batch-size 32 or smaller")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator_type = "local" if args.local else "AWS Braket"
    
    print(f"\n{'='*70}")
    print(f"Quantum Hybrid CNN Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Quantum simulator: {simulator_type}")
    print(f"Qubits: {args.quantum_qubits}")
    print(f"Quantum layers: {args.quantum_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    try:
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
        print("Loading CIFAR-10 dataset...")
        trainloader, validloader, testloader = get_data_loaders(args.batch_size)
        print(f"[OK] Data loaded: {len(trainloader.dataset)} train, {len(validloader.dataset)} val, {len(testloader.dataset)} test\n")
        
        # Create model
        print("Creating quantum hybrid model...")
        model = QuantumHybridCNN(n_qubits=args.quantum_qubits, n_quantum_layers=args.quantum_layers, use_local=args.local).to(device)
        print(f"[OK] Model created with {args.quantum_qubits} qubits and {args.quantum_layers} quantum layers\n")
        
        # Train
        train_model(model, trainloader, validloader, args.epochs, args.lr, device, tracker)
        
        # Test
        accuracy = test_model(model, testloader, device)
        tracker.set_test_accuracy(accuracy)
        
        # Save results
        tracker.add_note(f"Quantum hybrid CNN with {args.quantum_qubits} qubits using {simulator_type}")
        saved_path = tracker.save()
        print(f"[OK] Experiment results saved to: {saved_path}")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print(f"\nSuggestions:")
        print(f"  1. Install dependencies: pip install -r requirements.txt")
        print(f"  2. For AWS Braket: pip install amazon-braket-pennylane-plugin")
        print(f"  3. Verify PennyLane: pip install pennylane>=0.33.0")
    except FileNotFoundError as e:
        print(f"\n❌ File Not Found: {e}")
        print(f"\nSuggestions:")
        print(f"  1. Ensure you're in the correct directory")
        print(f"  2. Check that track_performance.py exists")
        print(f"  3. CIFAR-10 will auto-download on first run")
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e):
            print(f"\n❌ GPU Error: {e}")
            print(f"\nSuggestions:")
            print(f"  1. Reduce batch size: --batch-size 16 or --batch-size 8")
            print(f"  2. Reduce qubits: --quantum-qubits 2 or --quantum-qubits 3")
            print(f"  3. Use CPU: Set CUDA_VISIBLE_DEVICES='' before running")
        else:
            print(f"\n❌ Runtime Error: {e}")
            raise
    except KeyboardInterrupt:
        print(f"\n\n[WARN] Training interrupted by user")
        print(f"Partial results may be available in experiments/ directory")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print(f"\nIf this persists, please check:")
        print(f"  1. All dependencies are installed")
        print(f"  2. AWS credentials (if not using --local)")
        print(f"  3. System has sufficient memory")
        raise

if __name__ == '__main__':
    main()
