# Benchmarks

This folder contains performance benchmarks and training demonstrations.

## Files

### benchmark_decomposition.py
Performance benchmarks for unitary decomposition methods.

**Usage:**
```bash
python benchmark_decomposition.py
```

**Measures:**
- Decomposition time for Walsh vs PennyLane
- Gate count for different matrix sizes
- Memory usage during decomposition
- Reconstruction accuracy

**Results:**
- Walsh (4-qubit diagonal): ~1-6ms, 29 gates
- PennyLane (4-qubit non-diagonal): ~0.06ms, 1 gate
- Memory: ~0.02MB (minimal)

### run_training_demo.py
Demonstration of actual QNN training on CIFAR-10.

**Usage:**
```bash
python run_training_demo.py
```

**Features:**
- Loads CIFAR-10 dataset
- Trains QNN for 2 epochs
- Shows loss decreasing
- Validates accuracy
- Demonstrates gradient flow

**Configuration:**
- Batch size: 4
- Epochs: 2
- Batches per epoch: 10
- Learning rate: 0.001

## Running Benchmarks

All benchmarks can be run from the benchmarks folder:

```bash
cd benchmarks

# Run decomposition benchmarks
python benchmark_decomposition.py

# Run training demo
python run_training_demo.py
```

## Expected Results

### Decomposition Benchmarks
- Walsh decomposition: Fast for diagonal matrices
- PennyLane decomposition: Exact for all matrices
- Total overhead: ~1ms per forward pass

### Training Demo
- Model creates successfully
- Forward pass works
- Backward pass computes gradients
- Loss decreases over iterations
- No NaN or Inf values
