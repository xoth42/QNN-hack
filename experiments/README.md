# Experiment Tracking

## Folder Structure
```
experiments/
├── classical/          # Classical CNN runs
├── quantum/            # Quantum hybrid runs
└── comparison/         # Side-by-side comparisons
```

## Naming Convention
`YYYY-MM-DD_model-type_description.json`

Example: `2025-11-09_classical_baseline_15epochs.json`

## Experiment Log Format
```json
{
  "date": "2025-11-09",
  "model_type": "classical|quantum",
  "architecture": "description",
  "hyperparameters": {
    "batch_size": 64,
    "epochs": 15,
    "learning_rate": 0.001
  },
  "results": {
    "train_loss": [],
    "val_loss": [],
    "test_accuracy": 0.0,
    "training_time_seconds": 0.0
  },
  "notes": "Any observations"
}
```
