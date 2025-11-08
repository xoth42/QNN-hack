# Contributing Guide

## Branch Structure
- `main` - Stable code
- `quantum-layer-development` - Active quantum layer work
- `classical-baseline` - Classical CNN improvements

## Getting Started

### 1. Setup Environment
```bash
bash setup_cnn_env.sh
conda activate cnn
```

### 2. For Classical CNN Work
```bash
python cifar10_tinycnn.py
```

### 3. For Quantum Layer Work
- Ensure AWS Braket credentials configured
- Open `Hybrid-model.ipynb` in Jupyter
- Install PennyLane: `pip install pennylane pennylane-braket`

## Task Assignment

### Classical CNN Team
- Verify baseline runs
- Generate metrics and visualizations
- Document performance

### Quantum Layer Team
- Optimize quantum circuit design
- Track performance vs classical
- Experiment with architectures

## Submitting Work
1. Create feature branch from `quantum-layer-development`
2. Make changes
3. Test thoroughly
4. Create pull request with description of changes
5. Tag relevant team members for review

## Performance Tracking
Log all experiments in `experiments/` folder with:
- Configuration used
- Metrics achieved
- Runtime/cost
- Notes/observations
