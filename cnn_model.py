# cnn_model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import time
import numpy as np

class PureCNN(nn.Module):
    """Pure CNN baseline model"""
    
    def __init__(self):
        super(PureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 4)
        
        self.fc_quantum_equiv = nn.Linear(4, 4)
        
        self.fc2 = nn.Linear(4, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.relu(self.fc_quantum_equiv(x))
        x = self.fc2(x)
        return x


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, model_name, device='cpu'):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.iteration_times = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_time = time.time() - start_time
        self.iteration_times.append(epoch_time)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        self.test_accuracies.append(accuracy)
        return accuracy
