# Walsh Circuit Decomposition - Usage Guide

## Overview

The Walsh circuit decomposition is used to efficiently implement diagonal unitary matrices in quantum circuits using CNOT and RZ gates.

## Requirements

**IMPORTANT**: Walsh decomposition only works with **DIAGONAL** unitary matrices where all diagonal elements have magnitude 1 (are on the unit circle).

## Common Error

```
AssertionError at line 35: assert np.allclose(matrix, np.diag(d))
```

**Cause**: You're passing a non-diagonal matrix to `Walsh_coefficients()`.

## Solutions

### Solution 1: Ensure Your Matrix is Diagonal

```python
import numpy as np
from walsh_circuit_decomposition import Walsh_coefficients, build_optimal_walsh_circuit

# Create a diagonal unitary matrix
phases = np.array([0, np.pi/4, np.pi/2, np.pi])
diagonal_matrix = np.diag(np.exp(1j * phases))

# This will work
coeffs = Walsh_coefficients(diagonal_matrix)
circuit = build_optimal_walsh_circuit(diagonal_matrix)
```

### Solution 2: Diagonalize Your Matrix First

```python
from walsh_circuit_decomposition import diagonalize_unitary, build_optimal_walsh_circuit

# If you have a non-diagonal unitary
non_diagonal_unitary = ...  # Your matrix

# Diagonalize it first
diagonal_matrix, transformation = diagonalize_unitary(non_diagonal_unitary)

# Now you can use Walsh decomposition
circuit = build_optimal_walsh_circuit(diagonal_matrix)
```

### Solution 3: Check Your Matrix

```python
import numpy as np

# Check if matrix is diagonal
def is_diagonal(matrix):
    d = np.diag(matrix)
    return np.allclose(matrix, np.diag(d), atol=1e-10)

# Check if diagonal elements are on unit circle
def is_unitary_diagonal(matrix):
    d = np.diag(matrix)
    return np.allclose(np.abs(d), 1.0, atol=1e-12)

# Before calling Walsh_coefficients
if not is_diagonal(your_matrix):
    print("ERROR: Matrix must be diagonal!")
    # Diagonalize it or fix your code
    
if not is_unitary_diagonal(your_matrix):
    print("ERROR: Diagonal elements must have magnitude 1!")
    # Normalize or fix your code
```

## Complete Example

```python
import numpy as np
import torch
from walsh_circuit_decomposition import (
    Walsh_coefficients, 
    build_optimal_walsh_circuit,
    diagonalize_unitary
)

# Example 1: Diagonal matrix (works directly)
print("Example 1: Diagonal Matrix")
phases = np.array([0, np.pi/4, np.pi/2, np.pi])
diag_matrix = np.diag(np.exp(1j * phases))

# Convert to torch tensor if needed
diag_tensor = torch.tensor(diag_matrix)

# Get Walsh coefficients
coeffs = Walsh_coefficients(diag_tensor)
print(f"Walsh coefficients: {coeffs}")

# Build quantum circuit
circuit = build_optimal_walsh_circuit(diag_tensor)
print(f"Circuit gates: {circuit}")

# Example 2: Non-diagonal matrix (needs diagonalization)
print("\nExample 2: Non-Diagonal Matrix")
# Hadamard matrix (non-diagonal)
hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# Diagonalize first
diag_h, transform = diagonalize_unitary(hadamard)
print(f"Diagonalized: {np.diag(diag_h)}")

# Now can use Walsh decomposition
circuit_h = build_optimal_walsh_circuit(torch.tensor(diag_h))
print(f"Circuit gates: {circuit_h}")
```

## Understanding the Output

The `build_optimal_walsh_circuit()` function returns a list of tuples:

```python
[
    ('RZ', [angle, qubit_index]),  # Rotation around Z-axis
    ('CNOT', [control, target]),    # CNOT gate
    ...
]
```

Example:
```python
[
    ('RZ', [1.178, 0]),      # RZ(1.178) on qubit 0
    ('CNOT', [0, 1]),        # CNOT from qubit 0 to qubit 1
    ('RZ', [-0.393, 1]),     # RZ(-0.393) on qubit 1
]
```

## Integration with PennyLane

```python
import pennylane as qml

# Get circuit from Walsh decomposition
circuit_gates = build_optimal_walsh_circuit(your_diagonal_matrix)

# Apply to PennyLane circuit
@qml.qnode(dev)
def quantum_circuit():
    for gate_type, params in circuit_gates:
        if gate_type == 'RZ':
            angle, qubit = params
            qml.RZ(angle, wires=qubit)
        elif gate_type == 'CNOT':
            control, target = params
            qml.CNOT(wires=[control, target])
    return qml.state()
```

## Troubleshooting

### Error: "Matrix must be square"
- Your matrix is not square (e.g., 3x4)
- Fix: Ensure matrix is NxN

### Error: "Matrix size must be power of 2"
- Matrix size is not 2, 4, 8, 16, 32, etc.
- Fix: Pad matrix or use different size

### Error: "Matrix has off-diagonal elements"
- Your matrix is not diagonal
- Fix: Use `diagonalize_unitary()` first

### Error: "Diagonal elements must have magnitude 1"
- Your diagonal values are not on the unit circle
- Fix: Normalize: `matrix = np.diag(np.exp(1j * np.angle(np.diag(matrix))))`

## References

- Original code from: https://github.com/morik04/quantum-non-markovian-dynamics
- Walsh-Hadamard Transform: https://en.wikipedia.org/wiki/Hadamard_transform
