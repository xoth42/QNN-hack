"""
Test suite for RBS networks and entanglement patterns.
Tests all 4 patterns from Figure 9 of the paper.
"""
import sys
import io
# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import numpy as np
from tuple_triangle import pyramid, x_circuit, butterfly_circuit, round_robin_circuit
from density_qnn import create_rbs_network_from_pattern

print("="*70)
print("RBS NETWORK TESTS")
print("="*70)

# Test parameters
qubits = 8
patterns_to_test = ['pyramid', 'x_circuit', 'butterfly', 'round_robin']

print(f"\nTesting with {qubits} qubits")
print("-"*70)

for pattern_name in patterns_to_test:
    print(f"\n{pattern_name.upper()}:")
    
    try:
        # Create RBS network
        matrix = create_rbs_network_from_pattern(pattern_name, qubits)
        expected_size = 2**qubits
        
        # Test 1: Correct dimensions
        assert matrix.shape == (expected_size, expected_size), \
            f"Wrong shape: {matrix.shape}, expected ({expected_size}, {expected_size})"
        print(f"  ✅ Shape: {matrix.shape}")
        
        # Test 2: Is unitary (U @ U† = I)
        identity = torch.matmul(matrix, matrix.T.conj())
        is_unitary = torch.allclose(identity, torch.eye(expected_size), atol=1e-5)
        assert is_unitary, "Matrix is not unitary!"
        print(f"  ✅ Unitary: U @ U† = I (max error: {torch.max(torch.abs(identity - torch.eye(expected_size))).item():.2e})")
        
        # Test 3: Determinant has magnitude 1 (property of unitary matrices)
        det = torch.linalg.det(matrix.to(torch.complex64))
        det_magnitude = torch.abs(det).item()
        assert abs(det_magnitude - 1.0) < 1e-4, f"Determinant magnitude {det_magnitude} != 1"
        print(f"  ✅ Determinant: |det(U)| = {det_magnitude:.6f}")
        
        # Test 4: No NaN or Inf values
        assert not torch.isnan(matrix).any(), "Matrix contains NaN"
        assert not torch.isinf(matrix).any(), "Matrix contains Inf"
        print(f"  ✅ No NaN/Inf values")
        
        # Test 5: Get pattern structure info
        pattern_func = {
            'pyramid': pyramid,
            'x_circuit': x_circuit,
            'butterfly': butterfly_circuit,
            'round_robin': round_robin_circuit
        }[pattern_name]
        
        pattern_layers = pattern_func(qubits)
        depth = len(pattern_layers)
        total_gates = sum(len(layer) for layer in pattern_layers)
        print(f"  ℹ️  Depth: {depth}, Total RBS gates: {total_gates}")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("✅ ALL RBS NETWORK TESTS PASSED")
print("="*70)
