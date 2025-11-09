"""
Efficient CNN vs QNN Comparison Framework
Includes:
- Optimized CNN with BatchNorm and proper regularization
- AWS-ready Quantum Circuit integration
- Comprehensive testing infrastructure
- Performance tracking and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import time
from typing import List, Dict, Tuple, Any
import json

# Import our optimized models
from cnn_model import PureCNN
from qnn_model import HybridDensityQNN, QuantumCircuit
load_dotenv()

class ModelTrainer:
    """Efficient model trainer with performance tracking"""
    
    def __init__(self, model: nn.Module, name: str, device: torch.device = None):
        self.model = model
        self.name = name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Performance tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.iteration_times = []
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Train for one epoch and return (loss, accuracy)"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        total_batches = len(dataloader)
        # Choose a sensible progress interval (10% of epoch or at least once every 10 batches)
        progress_interval = max(1, min(10, total_batches // 10))

        for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Print lightweight progress and ETA periodically
            if batch_idx % progress_interval == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                batches_done = batch_idx
                avg_time_per_batch = elapsed / batches_done if batches_done > 0 else 0
                remaining_batches = total_batches - batches_done
                eta = remaining_batches * avg_time_per_batch
                print(f"{self.name} epoch progress: batch {batch_idx}/{total_batches} - "
                      f"loss={loss.item():.4f} - ETA: {eta:.1f}s")
        
        epoch_loss = total_loss / len(dataloader.dataset)
        accuracy = 100. * correct / total
        
        # Track metrics
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(accuracy)
        self.iteration_times.append(time.time() - start_time)
        
        return epoch_loss, accuracy
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model and return accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        self.test_accuracies.append(accuracy)
        return accuracy

class PerformanceVisualizer:
    """Enhanced performance visualization with confidence bands"""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name: str, results: Dict[str, List[float]]):
        """Add or update model results"""
        self.results[model_name] = results
    
    def plot_metrics(self, iterations: List[int]):
        """Create comprehensive performance plots with trends and confidence bands"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = {
            'Training Loss': ('train_losses', ax1),
            'Training Accuracy': ('train_accuracies', ax2),
            'Test Accuracy': ('test_accuracies', ax3),
            'Iteration Time': ('iteration_times', ax4)
        }
        
        colors = {'Pure CNN': 'blue', 'Hybrid QNN': 'red'}
        
        for title, (metric_key, ax) in metrics.items():
            for model_name, results in self.results.items():
                data = results.get(metric_key, [])
                if data:
                    # Plot actual values
                    ax.plot(iterations[:len(data)], data, 
                           label=model_name, color=colors[model_name],
                           marker='o', markersize=4)
                    
                    # Add trend line with confidence band
                    if len(data) > 2:
                        z = np.polyfit(iterations[:len(data)], data, 2)
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(iterations), max(iterations), 100)
                        y_trend = p(x_trend)
                        
                        # Calculate confidence band
                        y_mean = np.mean(data)
                        y_std = np.std(data)
                        ax.fill_between(x_trend, 
                                      y_trend - y_std, 
                                      y_trend + y_std,
                                      color=colors[model_name], alpha=0.2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class ExperimentRunner:
    """Streamlined experiment runner with enhanced monitoring"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'batch_size': 32,
            'num_workers': 2,
            'num_iterations': 5,
            'test_size': 1000,
            'train_size': 5000
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self._setup_data()
        self._setup_models()
        self.visualizer = PerformanceVisualizer()
    
    def _setup_data(self):
        """Setup efficient data loading"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
        # Use subset for faster experimentation
        train_indices = torch.randperm(len(trainset))[:self.config['train_size']]
        test_indices = torch.randperm(len(testset))[:self.config['test_size']]
        
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(testset, test_indices)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
    
    def _setup_models(self):
        """Initialize models with trainers"""
        self.cnn_model = PureCNN()
        self.hybrid_model = HybridDensityQNN(num_sub_unitaries=2, num_qubits=10)
        
        self.cnn_trainer = ModelTrainer(self.cnn_model, "Pure CNN", self.device)
        self.hybrid_trainer = ModelTrainer(self.hybrid_model, "Hybrid QNN", self.device)
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results"""
        print("\nStarting CNN vs QNN Comparison")
        print("=" * 50)
        
        iterations = list(range(1, self.config['num_iterations'] + 1))
        
        for iteration in iterations:
            print(f"\nIteration {iteration}/{self.config['num_iterations']}")
            print("-" * 30)
            
            # Train and evaluate CNN
            cnn_loss, cnn_acc = self.cnn_trainer.train_epoch(self.train_loader)
            cnn_test_acc = self.cnn_trainer.evaluate(self.test_loader)
            print(f"CNN     - Loss: {cnn_loss:.4f}, Train Acc: {cnn_acc:.2f}%, Test Acc: {cnn_test_acc:.2f}%")
            
            # Train and evaluate Hybrid QNN
            hybrid_loss, hybrid_acc = self.hybrid_trainer.train_epoch(self.train_loader)
            hybrid_test_acc = self.hybrid_trainer.evaluate(self.test_loader)
            print(f"Hybrid  - Loss: {hybrid_loss:.4f}, Train Acc: {hybrid_acc:.2f}%, Test Acc: {hybrid_test_acc:.2f}%")
        
        # Collect and visualize results
        for trainer in [self.cnn_trainer, self.hybrid_trainer]:
            self.visualizer.add_model_results(trainer.name, {
                'train_losses': trainer.train_losses,
                'train_accuracies': trainer.train_accuracies,
                'test_accuracies': trainer.test_accuracies,
                'iteration_times': trainer.iteration_times
            })
        
        # Generate plots
        fig = self.visualizer.plot_metrics(iterations)
        
        # Save results
        results = {
            'config': self.config,
            'cnn_results': {
                'train_losses': self.cnn_trainer.train_losses,
                'train_accuracies': self.cnn_trainer.train_accuracies,
                'test_accuracies': self.cnn_trainer.test_accuracies,
                'iteration_times': self.cnn_trainer.iteration_times
            },
            'hybrid_results': {
                'train_losses': self.hybrid_trainer.train_losses,
                'train_accuracies': self.hybrid_trainer.train_accuracies,
                'test_accuracies': self.hybrid_trainer.test_accuracies,
                'iteration_times': self.hybrid_trainer.iteration_times
            }
        }
        
        print("\nExperiment completed!")
        return results

def save_results(results: Dict[str, Any], filename: str = 'experiment_results.json'):
    """Save experiment results to file"""
    # Convert numpy arrays to lists for JSON serialization
    processed_results = json.loads(
        json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    )
    
    with open(filename, 'w') as f:
        json.dump(processed_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 32,
        'num_workers': 2,
        'num_iterations': 5,
        'test_size': 1000,  # Use smaller test set for faster iteration
        'train_size': 5000  # Use smaller training set for faster iteration
    }
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()
    
    # Save results
    save_results(results)
    
    plt.show()  # Show the performance plots
