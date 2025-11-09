"""
Train CNN on Quantum-Preprocessed Images
This trains a classical CNN on images that have been preprocessed with quantum filters

Usage:
    python train_quanvolutional_cnn.py --epochs 15
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
from track_performance import ExperimentTracker
from cifar10_tinycnn import SimpleCNN, train, test

def load_quanvolutional_data():
    """Load quantum-preprocessed CIFAR-10 data"""
    train_path = Path('data/quanvolutional_train.pt')
    test_path = Path('data/quanvolutional_test.pt')
    
    if not train_path.exists() or not test_path.exists():
        print("❌ Quantum-preprocessed data not found!")
        print("Run: python quanvolutional_preprocessing.py --local --num-samples 10000")
        return None, None, None
    
    print("Loading quantum-preprocessed data...")
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    print(f"✓ Train: {train_data['images'].shape}")
    print(f"✓ Test: {test_data['images'].shape}")
    
    return train_data, test_data

def create_dataloaders(train_data, test_data, batch_size=64, val_split=0.1):
    """Create dataloaders from preprocessed data"""
    
    # Split train into train/val
    n_train = len(train_data['images'])
    n_val = int(n_train * val_split)
    n_train = n_train - n_val
    
    indices = torch.randperm(len(train_data['images']))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        train_data['images'][train_indices],
        train_data['labels'][train_indices]
    )
    val_dataset = TensorDataset(
        train_data['images'][val_indices],
        train_data['labels'][val_indices]
    )
    test_dataset = TensorDataset(
        test_data['images'],
        test_data['labels']
    )
    
    # Create dataloaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load quantum-preprocessed data
    train_data, test_data = load_quanvolutional_data()
    if train_data is None:
        return
    
    # Create dataloaders
    trainloader, valloader, testloader = create_dataloaders(
        train_data, test_data, args.batch_size
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker('quanvolutional', 'quantum_preprocessed_cnn')
    tracker.set_hyperparameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        optimizer='Adam',
        architecture='16->32->32 filters on quantum-preprocessed images',
        preprocessing='quanvolutional (4-qubit, 2x2 patches)',
        train_samples=len(train_data['images']),
        test_samples=len(test_data['images'])
    )
    
    # Create model (same architecture as classical)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\n" + "="*80)
    print("TRAINING CNN ON QUANTUM-PREPROCESSED IMAGES")
    print("="*80)
    
    # Train
    train_losses, val_losses = train(
        model, trainloader, valloader, criterion, optimizer,
        args.epochs, device, tracker
    )
    
    # Test
    acc = test(model, testloader, device)
    
    # Log results
    tracker.set_test_accuracy(acc)
    tracker.add_note("CNN trained on quantum-preprocessed CIFAR-10 images")
    tracker.add_note("Quantum preprocessing: 4-qubit circuit on 2x2 patches")
    
    # Save
    saved_path = tracker.save()
    print(f"\n✓ Experiment results saved to: {saved_path}")
    
    print("\n" + "="*80)
    print(f"FINAL TEST ACCURACY: {acc:.2f}%")
    print("="*80)
    print("\nNext: Run compare_and_visualize.py to see how this compares to classical!")

if __name__ == '__main__':
    main()
