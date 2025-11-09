"""
Performance tracking utility for classical vs quantum CNN comparison
"""
import json
import time
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    def __init__(self, model_type: str, description: str):
        self.model_type = model_type  # 'classical' or 'quantum'
        self.description = description
        self.start_time = None
        self.metrics = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'model_type': model_type,
            'description': description,
            'hyperparameters': {},
            'results': {
                'train_loss': [],
                'val_loss': [],
                'test_accuracy': 0.0,
                'training_time_seconds': 0.0,
                'inference_time_ms': 0.0
            },
            'notes': ''
        }
    
    def set_hyperparameters(self, **kwargs):
        self.metrics['hyperparameters'] = kwargs
    
    def start_training(self):
        self.start_time = time.time()
    
    def end_training(self):
        if self.start_time:
            self.metrics['results']['training_time_seconds'] = time.time() - self.start_time
    
    def log_epoch(self, train_loss: float, val_loss: float):
        self.metrics['results']['train_loss'].append(train_loss)
        self.metrics['results']['val_loss'].append(val_loss)
    
    def set_test_accuracy(self, accuracy: float):
        self.metrics['results']['test_accuracy'] = accuracy
    
    def set_inference_time(self, time_ms: float):
        self.metrics['results']['inference_time_ms'] = time_ms
    
    def add_note(self, note: str):
        self.metrics['notes'] += note + '\n'
    
    def save(self):
        # Create experiments directory structure
        exp_dir = Path('experiments') / self.model_type
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        filename = f"{timestamp}_{self.model_type}_{self.description}.json"
        filepath = exp_dir / filename
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Experiment saved to: {filepath}")
        return filepath

# Example usage:
if __name__ == '__main__':
    # Classical CNN example
    tracker = ExperimentTracker('classical', 'baseline_15epochs')
    tracker.set_hyperparameters(
        batch_size=64,
        epochs=15,
        learning_rate=0.001,
        optimizer='Adam'
    )
    tracker.start_training()
    # ... training loop ...
    tracker.log_epoch(train_loss=0.5, val_loss=0.6)
    tracker.end_training()
    tracker.set_test_accuracy(75.5)
    tracker.add_note('First baseline run')
    tracker.save()
