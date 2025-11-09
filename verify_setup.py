"""
Verification Script - Test all components before running expensive experiments

Usage:
    python verify_setup.py

This script performs quick sanity checks on:
- Library imports
- Data loading
- Classical CNN forward pass
- Quantum CNN forward pass (local simulator)
- Experiment tracker functionality
"""

import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(name: str, status: str, message: str = ""):
    """Print test result with color coding"""
    if status == "pass":
        symbol = f"{Colors.GREEN}✓{Colors.END}"
        status_text = f"{Colors.GREEN}PASS{Colors.END}"
    elif status == "fail":
        symbol = f"{Colors.RED}✗{Colors.END}"
        status_text = f"{Colors.RED}FAIL{Colors.END}"
    else:  # running
        symbol = f"{Colors.BLUE}→{Colors.END}"
        status_text = f"{Colors.BLUE}RUNNING{Colors.END}"
    
    print(f"{symbol} {Colors.BOLD}{name}{Colors.END}: {status_text}", end="")
    if message:
        print(f" - {message}")
    else:
        print()

def verify_imports():
    """Test 1: Verify all required libraries can be imported"""
    print_test("Import Test", "running")
    
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        print_test("Import Test", "pass", 
                  f"PyTorch {torch.__version__}, NumPy {np.__version__}")
        return True
    except ImportError as e:
        print_test("Import Test", "fail", str(e))
        print(f"  {Colors.YELLOW}Fix: Run 'conda activate cnn' and 'bash setup_cnn_env.sh'{Colors.END}")
        return False

def verify_quantum_imports():
    """Test 1b: Verify quantum libraries (optional)"""
    print_test("Quantum Import Test", "running")
    
    try:
        import pennylane as qml
        print_test("Quantum Import Test", "pass", f"PennyLane {qml.__version__}")
        return True
    except ImportError as e:
        print_test("Quantum Import Test", "fail", str(e))
        print(f"  {Colors.YELLOW}Fix: Run 'pip install pennylane'{Colors.END}")
        return False

def verify_data_loading():
    """Test 2: Verify CIFAR-10 can be loaded"""
    print_test("Data Loading Test", "running")
    
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Load small subset
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
        
        # Get one batch
        images, labels = next(iter(testloader))
        
        assert images.shape == torch.Size([4, 3, 32, 32]), f"Unexpected shape: {images.shape}"
        assert labels.shape == torch.Size([4]), f"Unexpected label shape: {labels.shape}"
        
        print_test("Data Loading Test", "pass", 
                  f"Loaded batch shape: {list(images.shape)}")
        return True
    except Exception as e:
        print_test("Data Loading Test", "fail", str(e))
        print(f"  {Colors.YELLOW}Fix: Check internet connection for CIFAR-10 download{Colors.END}")
        return False

def verify_classical_model():
    """Test 3: Verify classical CNN forward pass"""
    print_test("Classical CNN Test", "running")
    
    try:
        import torch
        import torch.nn as nn
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import SimpleCNN from cifar10_tinycnn.py
        from cifar10_tinycnn import SimpleCNN
        
        # Create model
        model = SimpleCNN(num_classes=10)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(4, 3, 32, 32)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == torch.Size([4, 10]), f"Unexpected output shape: {output.shape}"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print_test("Classical CNN Test", "pass", 
                  f"Output shape: {list(output.shape)}, Parameters: {total_params:,}")
        return True
    except Exception as e:
        print_test("Classical CNN Test", "fail", str(e))
        print(f"  {Colors.YELLOW}Fix: Ensure cifar10_tinycnn.py exists and is valid{Colors.END}")
        return False

def verify_quantum_model():
    """Test 4: Verify quantum hybrid CNN forward pass (local simulator)"""
    print_test("Quantum CNN Test", "running")
    
    try:
        import torch
        import pennylane as qml
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Check if quantum_hybrid_cnn.py exists
        if not Path("quantum_hybrid_cnn.py").exists():
            print_test("Quantum CNN Test", "fail", "quantum_hybrid_cnn.py not found")
            return False
        
        # Import QuantumHybridCNN
        from quantum_hybrid_cnn import QuantumHybridCNN
        
        # Create model with local simulator (use_local=True)
        model = QuantumHybridCNN(n_qubits=4, n_quantum_layers=2, use_local=True)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 32, 32)  # Small batch for speed
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == torch.Size([2, 10]), f"Unexpected output shape: {output.shape}"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print_test("Quantum CNN Test", "pass", 
                  f"Forward pass successful, Output: {list(output.shape)}, Parameters: {total_params:,}")
        return True
    except Exception as e:
        print_test("Quantum CNN Test", "fail", str(e))
        print(f"  {Colors.YELLOW}Fix: Ensure quantum_hybrid_cnn.py is valid and PennyLane is installed{Colors.END}")
        return False

def verify_experiment_tracker():
    """Test 5: Verify experiment tracker save/load"""
    print_test("Experiment Tracker Test", "running")
    
    try:
        import sys
        import json
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from track_performance import ExperimentTracker
        
        # Create tracker
        tracker = ExperimentTracker('test', 'verification_test')
        tracker.set_hyperparameters(batch_size=32, epochs=1, learning_rate=0.001)
        tracker.start_training()
        tracker.log_epoch(0.5, 0.6)
        tracker.end_training()
        tracker.set_test_accuracy(75.0)
        tracker.add_note("Verification test")
        
        # Save
        filepath = tracker.save()
        
        # Verify file exists
        assert filepath.exists(), f"File not created: {filepath}"
        
        # Load and verify
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert data['model_type'] == 'test'
        assert data['results']['test_accuracy'] == 75.0
        assert len(data['results']['train_loss']) == 1
        
        # Cleanup
        filepath.unlink()
        
        print_test("Experiment Tracker Test", "pass", 
                  f"Saved and loaded successfully")
        return True
    except Exception as e:
        print_test("Experiment Tracker Test", "fail", str(e))
        print(f"  {Colors.YELLOW}Fix: Ensure track_performance.py exists{Colors.END}")
        return False

def main():
    """Run all verification tests"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  Quantum CNN Comparison - Setup Verification{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    results = {}
    
    # Run tests
    results['imports'] = verify_imports()
    results['quantum_imports'] = verify_quantum_imports()
    results['data'] = verify_data_loading()
    results['classical_model'] = verify_classical_model()
    results['quantum_model'] = verify_quantum_model()
    results['tracker'] = verify_experiment_tracker()
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All verifications passed! ({passed}/{total}){Colors.END}")
        print(f"\n{Colors.GREEN}You're ready to run experiments!{Colors.END}")
        print(f"\nNext steps:")
        print(f"  1. Run classical baseline: python cifar10_tinycnn.py")
        print(f"  2. Run quantum hybrid: python quantum_hybrid_cnn.py --local --epochs 2")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some verifications failed ({passed}/{total} passed){Colors.END}")
        print(f"\n{Colors.YELLOW}Please fix the issues above before running experiments.{Colors.END}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
