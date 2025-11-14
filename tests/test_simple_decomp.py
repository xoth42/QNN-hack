"""Simple test of decomposition with known matrix"""
import numpy as np
import torch
from walsh_circuit_decomposition import build_optimal_walsh_circuit

# Test with simple 2-qubit diagonal matrix
print("Testing simple 2-qubit diagonal matrix...")
phases = np.array([0, np.pi/4, np.pi/2, np.pi])
D = np.diag(np.exp(1j * phases))

print(f"Original D:\n{D}")

# Build circuit
circuit = build_optimal_walsh_circuit(torch.tensor(D))
print(f"\nCircuit: {len(circuit)} gates")
for gate in circuit:
    print(f"  {gate}")

# The issue: Walsh decomposition expects the matrix to represent
# a diagonal unitary in the computational basis
# But our RBS matrices are NOT diagonal!

print("\n" + "="*70)
print("INSIGHT: The problem is that RBS networks create NON-DIAGONAL unitaries!")
print("We MUST diagonalize them first, then apply V @ D @ V†")
print("But our current implementation tries to use Walsh on non-diagonal V!")
print("="*70)
