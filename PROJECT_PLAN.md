# QNN Hackathon Project Plan

## Goal
Compare classical CNN vs quantum-hybrid CNN for CIFAR-10 image classification

---

## Task 1: Classical CNN Baseline (Manager: @681130121284943877)

### Status: In Progress
- [x] 1a. Basic code structure exists (`cifar10_tinycnn.py`)
- [ ] 1a. Verify code runs end-to-end without errors
- [ ] 1b. Generate training runs with metrics:
  - Training/validation loss curves
  - Test accuracy
  - Training time
  - Model size/parameters
  - Confusion matrix
- [ ] 1b. Create presentable output (plots, tables, summary stats)

### Deliverables
- Working classical CNN baseline
- Performance metrics document
- Visualization of results

---

## Task 2: Quantum Hybrid Layer (Manager: You)

### Status: Partially Complete
- [x] 2a. Basic quantum layer input (4 features via FC layer)
- [x] 2b. Initial quantum circuit (4 qubits, RY/RZ/CNOT gates)
- [ ] 2a. Optimize feature reduction for quantum input
- [ ] 2b. Experiment with different quantum circuits:
  - Vary number of qubits
  - Try different gate combinations
  - Test entanglement strategies
- [ ] 2c. Verify quantum→classical output integration
- [ ] 2d. Implement performance tracking:
  - Training time comparison
  - Inference time per batch
  - AWS Braket costs/usage
  - Accuracy metrics

### Current Architecture
```
Input (3x32x32) 
→ Conv layers (feature extraction)
→ FC layer (reduce to 4 dims)
→ QUANTUM LAYER (4 qubits)
→ FC layer (4 → 10 classes)
→ Output
```

### Experiments to Try
1. Single quantum layer (current)
2. Multiple quantum layers
3. Different quantum layer positions
4. Vary quantum circuit depth

---

## Task 3: Comparison & Analysis

### Metrics to Compare
- [ ] Test accuracy (classical vs quantum)
- [ ] Training time
- [ ] Inference time
- [ ] Model complexity (parameters)
- [ ] Convergence speed (epochs to target accuracy)
- [ ] Resource usage (memory, compute)

### Deliverables
- Comparison table
- Performance plots
- Analysis writeup
- Presentation slides

---

## Current Files
- `cifar10_tinycnn.py` - Classical CNN baseline
- `Hybrid-model.ipynb` - Quantum hybrid implementation
- `setup_cnn_env.sh` - Environment setup
- `install_prereqs.py` - Dependency installer

## Next Steps
1. Run classical baseline and collect metrics
2. Complete quantum layer integration
3. Run experiments with different quantum architectures
4. Compare results
