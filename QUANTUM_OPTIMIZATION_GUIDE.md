# Quantum CNN Optimization Guide

## 🚀 New Optimized Implementation

### What's Been Improved

**Original `quantum_hybrid_cnn.py` Issues:**
- ❌ Only 4 qubits (very limited)
- ❌ Linear entanglement only (weak)
- ❌ 2 variational layers (too shallow)
- ❌ Only RY+RZ gates
- ❌ No data re-uploading
- ❌ 34 minutes for 1 epoch with batch_size=4
- ❌ 27.28% accuracy

**New `quantum_hybrid_cnn_optimized.py` Features:**
- ✅ Configurable qubits (4, 8, 16+)
- ✅ Circular entanglement (stronger connectivity)
- ✅ 4+ variational layers (deeper circuits)
- ✅ RX+RY+RZ gates (more expressive)
- ✅ Data re-uploading (better feature encoding)
- ✅ Progress bars with tqdm
- ✅ Better hyperparameter defaults

## 📊 Running Experiments

### Quick Start

```bash
# 1. Classical baseline (if not run yet)
python cifar10_tinycnn.py

# 2. Optimized quantum experiments
python quantum_hybrid_cnn_optimized.py --local --epochs 5 --quantum-qubits 8 --quantum-layers 4

# 3. Generate comparison report
python compare_and_visualize.py
```

### Systematic Experiments

```bash
# Run all experiments automatically
python run_all_experiments.py
```

This will:
1. Check if classical baseline exists
2. Run quantum experiments with different configurations
3. Generate comprehensive comparison report

### Manual Experiments

```bash
# Small quantum (fast, baseline)
python quantum_hybrid_cnn_optimized.py --local --epochs 5 --quantum-qubits 4 --quantum-layers 2 --batch-size 16

# Medium quantum (balanced)
python quantum_hybrid_cnn_optimized.py --local --epochs 5 --quantum-qubits 8 --quantum-layers 4 --batch-size 16

# Large quantum (best performance, slower)
python quantum_hybrid_cnn_optimized.py --local --epochs 5 --quantum-qubits 16 --quantum-layers 6 --batch-size 8
```

## 🔬 Quantum Circuit Design

### Original Circuit
```
Input → RY encoding
     → RY + RZ rotations (2 layers)
     → Linear CNOT chain
     → Measure PauliZ
```

### Optimized Circuit
```
Input → RY encoding
     → Data re-uploading (RY * 0.5)
     → RX + RY + RZ rotations (4+ layers)
     → Circular CNOT chain
     → Additional every-other-qubit CNOTs
     → Measure PauliZ
```

**Key Improvements:**
1. **Data Re-uploading**: Encodes data multiple times through the circuit
2. **Circular Entanglement**: Last qubit connects to first (no dead ends)
3. **Additional Entanglement**: Every-other-qubit connections for richer correlations
4. **3 Rotation Gates**: RX, RY, RZ per qubit (more expressive)
5. **Deeper Circuits**: 4-6 layers instead of 2

## 📈 Expected Results

### Classical Baseline
- **Accuracy**: 65-75% (small architecture: 16→32→32)
- **Training Time**: 15-20 minutes (15 epochs)
- **Parameters**: ~81,000

### Quantum Hybrid (Original)
- **Accuracy**: 27-35% (very poor)
- **Training Time**: 30-60 minutes (slow)
- **Parameters**: ~16,000 quantum + classical

### Quantum Hybrid (Optimized)
- **Accuracy**: 50-65% (target: 70-80% of classical)
- **Training Time**: 20-40 minutes (depends on qubits/layers)
- **Parameters**: More quantum parameters = better expressivity

## 🎯 Optimization Strategies

### For Speed
```bash
--batch-size 32          # Larger batches (but slower per batch)
--epochs 3               # Fewer epochs for quick tests
--quantum-qubits 4       # Fewer qubits
--quantum-layers 2       # Shallower circuits
```

### For Accuracy
```bash
--batch-size 16          # Smaller batches (better gradients)
--epochs 10              # More training
--quantum-qubits 8-16    # More qubits
--quantum-layers 4-6     # Deeper circuits
```

### For Balanced
```bash
--batch-size 16
--epochs 5
--quantum-qubits 8
--quantum-layers 4
```

## 📊 Analysis Tools

### Compare All Experiments
```bash
python compare_and_visualize.py
```

Generates:
- `experiments/analysis/loss_curves.png` - Training/validation loss
- `experiments/analysis/accuracy_comparison.png` - Bar chart of all models
- `experiments/analysis/training_time_comparison.png` - Time comparison
- `experiments/analysis/quantum_analysis.png` - Qubits vs accuracy, layers vs accuracy
- `experiments/analysis/summary_report.txt` - Text summary

## 🔧 Troubleshooting

### "Training too slow"
- Reduce `--batch-size` to 8 or 4
- Reduce `--quantum-qubits` to 4
- Reduce `--quantum-layers` to 2
- Use `--epochs 3` for quick tests

### "Accuracy too low"
- Increase `--quantum-qubits` to 8 or 16
- Increase `--quantum-layers` to 4 or 6
- Train for more `--epochs`
- Check if classical baseline is good (should be 65-75%)

### "Out of memory"
- Reduce `--batch-size`
- Reduce `--quantum-qubits`
- Close other applications

### "PennyLane errors"
- Make sure you're using `--local` flag
- Check PennyLane version: `pip show pennylane`
- Reinstall if needed: `pip install --upgrade pennylane`

## 📝 Experiment Tracking

All experiments auto-save to:
```
experiments/
├── classical/           # Classical CNN results
├── quantum/             # Original quantum results
├── quantum_optimized/   # Optimized quantum results
└── analysis/            # Comparison plots and reports
```

Each experiment saves:
- Hyperparameters (qubits, layers, batch size, etc.)
- Training/validation loss per epoch
- Final test accuracy
- Training time
- Notes and configuration

## 🎓 Understanding the Results

### Good Results
- Quantum achieves 70-80% of classical accuracy
- Training time is reasonable (< 1 hour)
- Loss curves show convergence
- Validation accuracy improves over epochs

### Bad Results
- Quantum < 50% of classical accuracy
- Loss curves don't converge
- Validation accuracy doesn't improve
- Training takes > 2 hours

### What to Try
1. **If accuracy is low**: Increase qubits and layers
2. **If training is slow**: Decrease batch size or use fewer qubits
3. **If overfitting**: Add dropout or reduce epochs
4. **If underfitting**: Train longer or use deeper circuits

## 🚀 Next Steps

1. **Run classical baseline** (if not done)
2. **Run 3-5 quantum experiments** with different configurations
3. **Generate comparison report**
4. **Analyze results** and identify best configuration
5. **Document findings** in your report

## 📚 References

- PennyLane Documentation: https://pennylane.ai
- Quantum Machine Learning: https://pennylane.ai/qml/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
