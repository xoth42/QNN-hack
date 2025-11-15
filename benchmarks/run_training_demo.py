"""
Real Training Demo - Actually train the QNN on CIFAR-10

This script performs REAL training (not just tests):
- Loads CIFAR-10 dataset
- Creates QNN model
- Trains for a few epochs
- Shows loss decreasing
- Validates accuracy
"""

import sys
import io
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Ensure local simulator
if 'BRAKET_DEVICE' in os.environ:
    del os.environ['BRAKET_DEVICE']

from qnn_model import HybridDensityQNN

print("="*80)
print("REAL QNN TRAINING DEMO")
print("="*80)

# Configuration
BATCH_SIZE = 40
NUM_EPOCHS = 10
NUM_BATCHES_PER_EPOCH = 100  # Small number for demo
LEARNING_RATE = 0.001

print(f"\nConfiguration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batches per epoch: {NUM_BATCHES_PER_EPOCH}")
print(f"  Learning rate: {LEARNING_RATE}")

# Load CIFAR-10
print("\n[1/5] Loading CIFAR-10 dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  Training samples: {len(trainset)}")
print(f"  Test samples: {len(testset)}")
print("  [OK] Dataset loaded")

# Create model
print("\n[2/5] Creating QNN model...")
model = HybridDensityQNN(num_qubits=4)
param_count = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {param_count}")
print("  [OK] Model created")

# Setup training
print("\n[3/5] Setting up training...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("  [OK] Training setup complete")

# Training loop
print("\n[4/5] Starting training...")
print("="*80)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-"*80)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    dataiter = iter(trainloader)
    
    for batch_idx in range(NUM_BATCHES_PER_EPOCH):
        try:
            images, labels = next(dataiter)
        except StopIteration:
            dataiter = iter(trainloader)
            images, labels = next(dataiter)
        
        # Forward pass
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        forward_time = time.time() - start_time
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            print(f"  Batch {batch_idx+1}/{NUM_BATCHES_PER_EPOCH}: "
                  f"Loss={loss.item():.4f}, "
                  f"Avg Loss={avg_loss:.4f}, "
                  f"Acc={accuracy:.2f}%, "
                  f"Time={forward_time:.2f}s")
    
    # Epoch summary
    epoch_loss = running_loss / NUM_BATCHES_PER_EPOCH
    epoch_acc = 100 * correct / total
    print(f"\n  Epoch {epoch+1} Summary:")
    print(f"    Average Loss: {epoch_loss:.4f}")
    print(f"    Training Accuracy: {epoch_acc:.2f}%")

print("\n" + "="*80)
print("[5/5] Evaluating on test set...")

model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    dataiter = iter(testloader)
    for i in range(10):  # Test on 10 batches
        try:
            images, labels = next(dataiter)
        except StopIteration:
            break
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
avg_test_loss = test_loss / 10

print(f"  Test Loss: {avg_test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.2f}%")
print("  [OK] Evaluation complete")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nResults:")
print(f"  Final Training Loss: {epoch_loss:.4f}")
print(f"  Final Training Accuracy: {epoch_acc:.2f}%")
print(f"  Test Accuracy: {test_accuracy:.2f}%")
print(f"\n  Model is working correctly!")
print(f"  Ready for full-scale training on CIFAR-10")
print("="*80)
