import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import time
from typing import List, Dict, Tuple, Any
import json

# Load environment variables
load_dotenv()
from cnn_model import PureCNN, ModelTrainer
from qnn_model import HybridDensityQNN, QuantumCircuit

class DryRunTester:
    """Comprehensive dry-run testing for all components"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_all = True
    
    def run_all_tests(self):
        """Run all dry-run tests"""
        print("🚀 STARTING COMPREHENSIVE DRY-RUN TEST SUITE")
        print("=" * 70)
        
        tests = [
            self.test_environment,
            self.test_quantum_circuit,
            self.test_cnn_model,
            self.test_hybrid_model_forward,
            self.test_hybrid_model_batch_processing,
            self.test_data_loading,
            self.test_training_step,
            self.test_model_initialization,
            self.test_performance_tracking,
            self.test_end_to_end_simulation
        ]
        
        for i, test in enumerate(tests, 1):
            print(f"\n📋 Test {i}/{len(tests)}: {test.__name__}...")
            try:
                result = test()
                self.test_results[test.__name__] = result
                if result["status"] == "PASS":
                    print(f"   ✅ {result['message']}")
                else:
                    print(f"   ❌ {result['message']}")
                    self.passed_all = False
            except Exception as e:
                self.test_results[test.__name__] = {
                    "status": "FAIL", 
                    "message": f"Exception: {str(e)}"
                }
                print(f"   💥 TEST CRASHED: {str(e)}")
                self.passed_all = False
        
        self._print_summary()
        return self.passed_all
    
    def test_environment(self):
        """Test environment setup and dependencies"""
        try:
            # Test PyTorch
            assert torch.__version__ is not None, "PyTorch not available"
            x = torch.randn(2, 2)
            y = x * 2
            assert y.shape == (2, 2), "Basic tensor operations failed"
            
            # Test PennyLane
            assert qml.__version__ is not None, "PennyLane not available"
            dev = qml.device("default.qubit", wires=2)
            
            # Test matplotlib
            import matplotlib
            assert matplotlib.__version__ is not None, "Matplotlib not available"
            
            return {"status": "PASS", "message": "All dependencies loaded correctly"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Environment setup failed: {str(e)}"}
    
    def test_quantum_circuit(self):
        """Test quantum circuit creation and execution"""
        try:
            # Test circuit initialization
            qc = QuantumCircuit(num_qubits=4, shots=100)
            assert hasattr(qc, 'circuit'), "Quantum circuit not created"
            assert callable(qc.circuit), "Circuit is not callable"
            
            # Test circuit execution with dummy data
            circuit_func = qc.circuit
            test_inputs = torch.tensor([0.1, 0.2, 0.3, 0.4])
            test_weights = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
            
            # This should not crash
            result = circuit_func(test_inputs, test_weights)
            assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"
            assert all(isinstance(x, (float, torch.Tensor)) for x in result), "Invalid output types"
            
            return {"status": "PASS", "message": "Quantum circuit created and executed successfully"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Quantum circuit test failed: {str(e)}"}
    
    def test_cnn_model(self):
        """Test pure CNN model"""
        try:
            model = PureCNN()
            
            # Test model initialization
            assert hasattr(model, 'forward'), "Model missing forward method"
            
            # Test forward pass with different batch sizes
            test_batches = [1, 4, 8]
            for batch_size in test_batches:
                test_input = torch.randn(batch_size, 3, 32, 32)
                output = model(test_input)
                expected_shape = (batch_size, 10)
                assert output.shape == expected_shape, \
                    f"Expected shape {expected_shape}, got {output.shape} for batch size {batch_size}"
            
            # Test parameter counting
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert param_count > 0, "Model has no trainable parameters"
            
            return {"status": "PASS", "message": f"CNN model works (params: {param_count})"}
        except Exception as e:
            return {"status": "FAIL", "message": f"CNN model test failed: {str(e)}"}
    
    def test_hybrid_model_forward(self):
        """Test hybrid model forward pass"""
        try:
            model = HybridDensityQNN(num_sub_unitaries=2, num_qubits=4)
            
            # Test single sample
            test_input = torch.randn(1, 3, 32, 32)
            output = model(test_input)
            assert output.shape == (1, 10), f"Single sample output shape incorrect: {output.shape}"
            
            return {"status": "PASS", "message": "Hybrid model single sample forward pass successful"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Hybrid model forward pass failed: {str(e)}"}
    
    def test_hybrid_model_batch_processing(self):
        """Test hybrid model with batch processing"""
        try:
            model = HybridDensityQNN(num_sub_unitaries=2, num_qubits=4)
            
            # Test various batch sizes
            batch_sizes = [2, 4]
            for batch_size in batch_sizes:
                test_input = torch.randn(batch_size, 3, 32, 32)
                output = model(test_input)
                expected_shape = (batch_size, 10)
                assert output.shape == expected_shape, \
                    f"Batch size {batch_size}: expected {expected_shape}, got {output.shape}"
            
            # Test mixing coefficients
            alpha_norm = torch.softmax(model.alpha, dim=0)
            assert alpha_norm.shape == (2,), f"Alpha shape incorrect: {alpha_norm.shape}"
            assert torch.allclose(alpha_norm.sum(), torch.tensor(1.0)), "Alpha doesn't sum to 1"
            
            return {"status": "PASS", "message": "Hybrid model batch processing successful"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Hybrid model batch processing failed: {str(e)}"}
    
    def test_data_loading(self):
        """Test data loading and preprocessing"""
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Test with small fake dataset first
            from torch.utils.data import TensorDataset, DataLoader
            
            # Create fake data
            fake_images = torch.randn(20, 3, 32, 32)
            fake_labels = torch.randint(0, 10, (20,))
            fake_dataset = TensorDataset(fake_images, fake_labels)
            
            # Test data loader
            loader = DataLoader(fake_dataset, batch_size=4, shuffle=True)
            
            # Test one batch
            for batch_idx, (data, target) in enumerate(loader):
                assert data.shape == (4, 3, 32, 32), f"Data shape incorrect: {data.shape}"
                assert target.shape == (4,), f"Target shape incorrect: {target.shape}"
                if batch_idx == 0:  # Just test first batch
                    break
            
            return {"status": "PASS", "message": "Data loading and batching successful"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Data loading test failed: {str(e)}"}
    
    def test_training_step(self):
        """Test a single training step"""
        try:
            # Create small fake dataset
            fake_images = torch.randn(8, 3, 32, 32)
            fake_labels = torch.randint(0, 10, (8,))
            fake_dataset = torch.utils.data.TensorDataset(fake_images, fake_labels)
            loader = torch.utils.data.DataLoader(fake_dataset, batch_size=4, shuffle=True)
            
            # Test training step
            model = PureCNN()
            trainer = ModelTrainer(model, "Test Model", device='cpu')
            
            # Single training step
            initial_loss, initial_acc = trainer.train_epoch(loader)
            
            assert isinstance(initial_loss, float), f"Loss should be float, got {type(initial_loss)}"
            assert isinstance(initial_acc, float), f"Accuracy should be float, got {type(initial_acc)}"
            assert 0 <= initial_acc <= 100, f"Accuracy out of range: {initial_acc}"
            
            return {"status": "PASS", "message": f"Training step successful (loss: {initial_loss:.4f})"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Training step test failed: {str(e)}"}
    
    def test_model_initialization(self):
        """Test that both models initialize correctly with same settings"""
        try:
            cnn_model = PureCNN()
            hybrid_model = HybridDensityQNN(num_sub_unitaries=2)
            
            # Test parameter counts are comparable
            cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
            hybrid_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
            
            param_ratio = hybrid_params / cnn_params
            assert 0.5 < param_ratio < 2.0, f"Parameter count ratio unreasonable: {param_ratio:.2f}"
            
            return {"status": "PASS", "message": f"Models initialized (CNN: {cnn_params}, Hybrid: {hybrid_params})"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Model initialization test failed: {str(e)}"}
    
    def test_performance_tracking(self):
        """Test performance tracking system"""
        try:
            tracker = PerformanceTracker()
            model = PureCNN()
            trainer = ModelTrainer(model, "Test Model")
            
            # Simulate some training results
            trainer.train_losses = [1.0, 0.8, 0.6]
            trainer.train_accuracies = [20.0, 40.0, 60.0]
            trainer.test_accuracies = [18.0, 38.0, 58.0]
            
            tracker.add_model_results("Test Model", trainer)
            
            assert "Test Model" in tracker.results, "Results not stored correctly"
            assert len(tracker.results["Test Model"]['train_losses']) == 3, "Loss history incorrect"
            
            # Test plotting doesn't crash
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            tracker.plot_performance([1, 2, 3])
            
            return {"status": "PASS", "message": "Performance tracking system working"}
        except Exception as e:
            return {"status": "FAIL", "message": f"Performance tracking test failed: {str(e)}"}
    
    def test_end_to_end_simulation(self):
        """Test a mini end-to-end simulation"""
        try:
            # Test with minimal settings
            runner = ExperimentRunner(num_iterations=1, batch_size=2)
            
            # Test data preparation
            assert hasattr(runner, 'train_loader'), "Train loader not created"
            assert hasattr(runner, 'test_loader'), "Test loader not created"
            
            # Test models are on correct device
            assert str(runner.cnn_model.conv1.weight.device) == str(runner.device), "CNN model on wrong device"
            assert str(runner.hybrid_model.conv1.weight.device) == str(runner.device), "Hybrid model on wrong device"
            
            # Test one iteration (this is the most comprehensive test)
            iterations = [1]
            iteration = 1
            
            print("      Running one training iteration...")
            cnn_loss, cnn_acc = runner.cnn_trainer.train_epoch(runner.train_loader)
            cnn_test_acc = runner.cnn_trainer.evaluate(runner.test_loader)
            
            hybrid_loss, hybrid_acc = runner.hybrid_trainer.train_epoch(runner.train_loader)
            hybrid_test_acc = runner.hybrid_trainer.evaluate(runner.test_loader)
            
            # Verify results are reasonable
            assert 0 <= cnn_acc <= 100, f"CNN accuracy out of range: {cnn_acc}"
            assert 0 <= hybrid_acc <= 100, f"Hybrid accuracy out of range: {hybrid_acc}"
            assert cnn_loss >= 0, f"CNN loss negative: {cnn_loss}"
            assert hybrid_loss >= 0, f"Hybrid loss negative: {hybrid_loss}"
            
            return {"status": "PASS", "message": "End-to-end simulation successful"}
        except Exception as e:
            return {"status": "FAIL", "message": f"End-to-end simulation failed: {str(e)}"}
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 DRY-RUN TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.test_results.values() if r["status"] == "PASS")
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        
        if self.passed_all:
            print("🎉 ALL TESTS PASSED! Ready to run main experiment.")
        else:
            print("❌ SOME TESTS FAILED! Please fix issues before running main experiment.")
            print("\nFailed tests:")
            for test_name, result in self.test_results.items():
                if result["status"] == "FAIL":
                    print(f"  - {test_name}: {result['message']}")
        
        print("=" * 70)

class PerformanceTracker:
    """Tracks and visualizes model performance"""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name, trainer: ModelTrainer):
        """Add results for a model"""
        self.results[model_name] = {
            'train_losses': trainer.train_losses.copy(),
            'train_accuracies': trainer.train_accuracies.copy(),
            'test_accuracies': trainer.test_accuracies.copy(),
            'iteration_times': trainer.iteration_times.copy()
        }
    
    def plot_performance(self, iterations: List[int]):
        """Plot performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = {'Pure CNN': 'blue', 'Hybrid QNN': 'red'}
        markers = {'Pure CNN': 'o', 'Hybrid QNN': 's'}
        
        # Plot training loss
        for model_name, results in self.results.items():
            if results['train_losses']:
                ax1.plot(iterations[:len(results['train_losses'])], 
                        results['train_losses'], 
                        label=model_name, color=colors[model_name], 
                        marker=markers[model_name], linewidth=2)
                
                # Add trend line
                if len(results['train_losses']) > 1:
                    z = np.polyfit(iterations[:len(results['train_losses'])], 
                                  results['train_losses'], 1)
                    p = np.poly1d(z)
                    ax1.plot(iterations, p(iterations), 
                            color=colors[model_name], linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss vs Iterations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training accuracy
        for model_name, results in self.results.items():
            if results['train_accuracies']:
                ax2.plot(iterations[:len(results['train_accuracies'])], 
                        results['train_accuracies'], 
                        label=model_name, color=colors[model_name], 
                        marker=markers[model_name], linewidth=2)
                
                # Add trend line
                if len(results['train_accuracies']) > 1:
                    z = np.polyfit(iterations[:len(results['train_accuracies'])], 
                                  results['train_accuracies'], 1)
                    p = np.poly1d(z)
                    ax2.plot(iterations, p(iterations), 
                            color=colors[model_name], linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Training Accuracy (%)')
        ax2.set_title('Training Accuracy vs Iterations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot test accuracy
        for model_name, results in self.results.items():
            if results['test_accuracies']:
                ax3.plot(iterations[:len(results['test_accuracies'])], 
                        results['test_accuracies'], 
                        label=model_name, color=colors[model_name], 
                        marker=markers[model_name], linewidth=2)
                
                # Add trend line
                if len(results['test_accuracies']) > 1:
                    z = np.polyfit(iterations[:len(results['test_accuracies'])], 
                                  results['test_accuracies'], 1)
                    p = np.poly1d(z)
                    ax3.plot(iterations, p(iterations), 
                            color=colors[model_name], linestyle='--', alpha=0.7)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Test Accuracy (%)')
        ax3.set_title('Test Accuracy vs Iterations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot iteration times
        for model_name, results in self.results.items():
            if results['iteration_times']:
                ax4.plot(iterations[:len(results['iteration_times'])], 
                        results['iteration_times'], 
                        label=model_name, color=colors[model_name], 
                        marker=markers[model_name], linewidth=2)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Training Time per Iteration')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            if results['train_accuracies']:
                print(f"  Final Training Accuracy: {results['train_accuracies'][-1]:.2f}%")
            if results['test_accuracies']:
                print(f"  Final Test Accuracy: {results['test_accuracies'][-1]:.2f}%")
            if results['train_losses']:
                print(f"  Final Training Loss: {results['train_losses'][-1]:.4f}")
            if results['iteration_times']:
                avg_time = np.mean(results['iteration_times'])
                print(f"  Average Iteration Time: {avg_time:.2f}s")


class ExperimentRunner:
    """Main class to run the comparative experiment"""
    
    def __init__(self, num_iterations=3, batch_size=4):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Data loaders
        self.train_loader, self.test_loader = self._prepare_data()
        
        # Models
        self.cnn_model = PureCNN()
        # self.hybrid_model = HybridDensityQNN(num_sub_unitaries=2)
        self.hybrid_model = HybridDensityQNN(num_sub_unitaries=2)

        
        # Trackers
        self.cnn_trainer = ModelTrainer(self.cnn_model, "Pure CNN", self.device)
        self.hybrid_trainer = ModelTrainer(self.hybrid_model, "Hybrid QNN", self.device)
        self.tracker = PerformanceTracker()
    
    def _prepare_data(self):
        """Prepare CIFAR-10 data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Use smaller subset for faster testing
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        
        # Use only 100 samples for dry run
        if len(trainset) > 100:
            indices = torch.randperm(len(trainset))[:100]
            trainset = torch.utils.data.Subset(trainset, indices)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
        # Use only 20 test samples
        if len(testset) > 20:
            indices = torch.randperm(len(testset))[:20]
            testset = torch.utils.data.Subset(testset, indices)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Training samples: {len(trainset)}")
        print(f"Test samples: {len(testset)}")
        print(f"Batch size: {self.batch_size}")
        
        return train_loader, test_loader
    
    def run_experiment(self):
        """Run the main experiment"""
        print(f"\nStarting experiment with {self.num_iterations} iterations")
        print("="*60)
        
        iterations = list(range(1, self.num_iterations + 1))
        
        for iteration in iterations:
            print(f"\n--- Iteration {iteration}/{self.num_iterations} ---")
            
            # Train CNN
            print("Training Pure CNN...")
            cnn_loss, cnn_acc = self.cnn_trainer.train_epoch(self.train_loader)
            cnn_test_acc = self.cnn_trainer.evaluate(self.test_loader)
            print(f"CNN - Loss: {cnn_loss:.4f}, Train Acc: {cnn_acc:.2f}%, Test Acc: {cnn_test_acc:.2f}%")
            
            # Train Hybrid Model
            print("Training Hybrid QNN...")
            hybrid_loss, hybrid_acc = self.hybrid_trainer.train_epoch(self.train_loader)
            hybrid_test_acc = self.hybrid_trainer.evaluate(self.test_loader)
            print(f"Hybrid - Loss: {hybrid_loss:.4f}, Train Acc: {hybrid_acc:.2f}%, Test Acc: {hybrid_test_acc:.2f}%")
        
        # Collect results
        self.tracker.add_model_results("Pure CNN", self.cnn_trainer)
        self.tracker.add_model_results("Hybrid QNN", self.hybrid_trainer)
        
        # Display results
        self.tracker.print_summary()
        self.tracker.plot_performance(iterations)
        
        return self.tracker


# Main execution
if __name__ == "__main__":
    # Run comprehensive dry-run tests
    tester = DryRunTester()
    all_passed = tester.run_all_tests()
    
    if all_passed:
        print("\n" + "🎉" * 20)
        print("ALL TESTS PASSED! Starting main experiment...")
        print("🎉" * 20)
        
        # # Run the main experiment
        # runner = ExperimentRunner(num_iterations=3, batch_size=4)
        # results = runner.run_experiment()
        
        # print("\n🏁 EXPERIMENT COMPLETED SUCCESSFULLY!")
    else:
        print("\n" + "❌" * 20)
        print("TESTS FAILED! Please fix the issues above before running the main experiment.")
        print("❌" * 20)

