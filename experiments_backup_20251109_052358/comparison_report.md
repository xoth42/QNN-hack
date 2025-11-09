# Classical vs Quantum CNN Comparison Report

**Generated:** 2025-11-09 05:23:46

## Executive Summary

[WARN] **No classical experiments found.** Found 1 quantum experiments.

## Statistical Significance Testing

[WARN] Insufficient samples for statistical testing (need at least 2 of each type)

## Individual Experiment Details

### Quantum Hybrid CNN Experiments

| Date | Description | Qubits | Layers | Test Acc (%) | Train Time (s) |
|------|-------------|--------|--------|--------------|----------------|
| 2025-11-09 | 4qubits_2layers_local | 4 | 2 | 27.28 | 2032.6 |

## Recommendations

- Run experiments for both classical and quantum models to enable comparison
- Use `python cifar10_tinycnn.py` for classical baseline
- Use `python quantum_hybrid_cnn.py --local` for quantum experiments

## Methodology

**Dataset:** CIFAR-10 (60,000 32x32 color images, 10 classes)

**Classical CNN Architecture:**
- 3 convolutional layers (3→32→64→128 channels)
- BatchNorm + ReLU + MaxPool after each conv layer
- Fully connected classifier with dropout

**Quantum Hybrid CNN Architecture:**
- Same convolutional backbone as classical
- Quantum layer with parameterized circuit (configurable qubits/depth)
- Linear classifier on quantum expectation values

**Statistical Testing:**
- Independent t-test for comparing means
- Significance level: α = 0.05

