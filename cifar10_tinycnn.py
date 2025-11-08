"""
How to run:
    1. Run `python setup_pip.py` (no conda needed)
    2. Run: `python cifar10_tinycnn.py`

Edit the SimpleCNN class (see Model Definition section) to adjust architecture.
"""

# ======================= Imports =======================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime
from track_performance import ExperimentTracker

# ================== Config & Hyperparameters ==================
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
NUM_WORKERS = 2

# For reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

x = strftime("%H:%M:%S", gmtime())
print("Start time: ", x)

def get_data_loaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_size = 5000
    train_subset, valid_subset = torch.utils.data.random_split(full_train, [len(full_train)-valid_size, valid_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, validloader, testloader, full_train.classes

def show_samples(loader, classes):
    images, labels = next(iter(loader))
    plt.figure(figsize=(10, 2))
    for i in range(8):
        img = images[i] * torch.tensor((0.2470, 0.2435, 0.2616)).view(3,1,1) + torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
        img = torch.clamp(img, 0, 1)
        npimg = img.numpy()
        plt.subplot(1, 8, i+1)
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.suptitle("Sample CIFAR-10 Training Images", fontsize=14)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

# ================= Model Definition: SimpleCNN =================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def train(
    model: nn.Module,
    trainloader,
    validloader,
    criterion,
    optimizer,
    epochs: int,
    device: torch.device,
    tracker: ExperimentTracker = None
):
    train_losses, valid_losses = [], []
    best_val_acc = 0.0
    
    if tracker:
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
        train_losses.append(avg_train_loss)
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
        val_acc = 100. * correct / total
        valid_losses.append(avg_val_loss)
        
        # Log to tracker
        if tracker:
            tracker.log_epoch(avg_train_loss, avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%, Current time: {strftime('%H:%M:%S', gmtime())}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    if tracker:
        tracker.end_training()
    
    print('Finished Training')
    return train_losses, valid_losses

def test(model, testloader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def main():
    # Initialize experiment tracker
    tracker = ExperimentTracker('classical', 'baseline_16_32_32_filters')
    tracker.set_hyperparameters(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer='Adam',
        architecture='16->32->32 filters',
        device=str(DEVICE)
    )
    
    trainloader, validloader, testloader, classes = get_data_loaders()
    show_samples(trainloader, classes)
    model = SimpleCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, val_losses = train(model, trainloader, validloader, criterion, optimizer, EPOCHS, DEVICE, tracker)
    acc = test(model, testloader, DEVICE)
    
    # Log final accuracy
    tracker.set_test_accuracy(acc)
    tracker.add_note(f"Architecture: 16->32->32 filters, smaller than original 32->64->128")
    
    # Save experiment
    saved_path = tracker.save()
    print(f"\n✓ Experiment results saved to: {saved_path}")
    
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss Curves')
    plt.legend()
    plt.show()
    print(f"Final Test Accuracy: {acc:.2f}%")
    print("End time: ", strftime("%H:%M:%S", gmtime()))

if __name__ == '__main__':
    main()
