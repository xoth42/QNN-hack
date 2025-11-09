# Quick Start Guide

## Setup (First Time)

1. **Configure Git** (if not done):
```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

2. **Setup Environment**:
```bash
bash setup_cnn_env.sh
conda activate cnn
```

3. **Install Quantum Dependencies**:
```bash
pip install pennylane pennylane-braket amazon-braket-sdk
```

4. **Configure AWS Braket** (for quantum layer):
```bash
aws configure
# Enter your AWS credentials
```

---

## Running Experiments

### Classical Baseline
```bash
python cifar10_tinycnn.py
```
Results saved to `best_model.pth`

### Quantum Hybrid
```bash
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4
```

Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32, smaller for quantum to reduce cost)
- `--quantum-qubits`: Number of qubits (default: 4)
- `--quantum-layers`: Quantum circuit depth (default: 2)

---

## Tracking Results

All experiments auto-save to `experiments/` folder:
- `experiments/classical/` - Classical CNN runs
- `experiments/quantum/` - Quantum hybrid runs

Each run creates a JSON file with:
- Hyperparameters
- Training/validation loss per epoch
- Test accuracy
- Training time
- Notes

---

## Branch Workflow

**Current branch**: `quantum-layer-development`

### Making Changes
```bash
# Create your feature branch
git checkout -b your-feature-name

# Make changes, then:
git add .
git commit -m "Description of changes"
git push origin your-feature-name
```

### Switching Branches
```bash
git checkout main                      # Go to main
git checkout quantum-layer-development # Go to quantum dev
```

---

## Task Checklist

### Classical Team
- [ ] Run baseline and verify it works
- [ ] Generate performance metrics
- [ ] Create visualizations (loss curves, confusion matrix)
- [ ] Document results

### Quantum Team
- [ ] Test quantum hybrid runs successfully
- [ ] Experiment with different qubit counts (2, 4, 8)
- [ ] Try different quantum layer depths
- [ ] Compare performance vs classical
- [ ] Track AWS costs

---

## Troubleshooting

**Import errors**: Make sure conda environment is activated
```bash
conda activate cnn
```

**AWS Braket errors**: Check credentials
```bash
aws sts get-caller-identity
```

**Out of memory**: Reduce batch size
```bash
python quantum_hybrid_cnn.py --batch-size 16
```

**Quantum layer too slow**: Use local simulator for testing
Edit `quantum_hybrid_cnn.py` line 23:
```python
dev = qml.device("default.qubit", wires=n_qubits)  # Local simulator
```
