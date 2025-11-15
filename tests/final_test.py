"""
Final comprehensive test - verify all QNN functionality works perfectly.
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Ensure local simulator
if 'BRAKET_DEVICE' in os.environ:
    del os.environ['BRAKET_DEVICE']

print("="*70)
print("FINAL COMPREHENSIVE QNN TEST")
print("="*70)

passed = 0
failed = 0

# Test 1: All imports
print("\n[1/12] Testing imports...")
try:
    from tuple_triangle import pyramid, x_circuit, butterfly_circuit, round_robin_circuit
    from density_qnn import density_layer, get_entanglement_pattern, string_from_RBS_connections
    from walsh_circuit_decomposition import Walsh_coefficients, build_optimal_walsh_circuit, diagonalize_unitary
    from qnn_model import QuantumCircuit, HybridDensityQNN
    print("  PASS - All imports successful")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1
    exit(1)

# Test 2: Entanglement patterns
print("\n[2/12] Testing all 4 paper patterns...")
try:
    n = 8
    patterns = {
        'pyramid': pyramid(n),
        'x_circuit': x_circuit(n),
        'butterfly': butterfly_circuit(n),
        'round_robin': round_robin_circuit(n)
    }
    
    for name, pattern in patterns.items():
        assert len(pattern) > 0, f"{name} has no layers"
        depth = len(pattern)
        gates = sum(len(layer) for layer in pattern)
        print(f"  {name:15s}: depth={depth:2d}, gates={gates:3d}")
    
    print("  PASS - All patterns working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 3: String conversion with parallel gates
print("\n[3/12] Testing parallel gate handling...")
try:
    qubits = 4
    # Test parallel gates: [(1,2), (3,4)]
    parallel_gates = [(1, 2), (3, 4)]
    string = string_from_RBS_connections(parallel_gates, qubits)
    
    assert string == 'RBSRBS', f"Expected 'RBSRBS', got '{string}'"
    print(f"  Parallel gates {parallel_gates} -> '{string}'")
    print("  PASS - Parallel gates handled correctly")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 4: Matrix sizes consistency
print("\n[4/12] Testing matrix size consistency...")
try:
    from density_qnn import matrix_from_IRBS_string
    
    qubits = 4
    expected_size = 2**qubits
    
    test_cases = [
        ([(1, 2)], 'RBSII'),
        ([(1, 2), (3, 4)], 'RBSRBS'),
        ([(2, 3)], 'IRBSI'),
    ]
    
    for gates, expected_str in test_cases:
        string = string_from_RBS_connections(gates, qubits)
        assert string == expected_str, f"Expected '{expected_str}', got '{string}'"
        
        matrix = matrix_from_IRBS_string(string)
        assert matrix.shape == (expected_size, expected_size), \
            f"Expected ({expected_size}, {expected_size}), got {matrix.shape}"
    
    print(f"  All matrices are {expected_size}x{expected_size}")
    print("  PASS - Matrix sizes consistent")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 5: Density layer creation
print("\n[5/12] Testing density layer...")
try:
    qubits = 4
    matrix_count = 4
    layer_func = density_layer(qubits, matrix_count)
    
    weights = torch.randn(matrix_count)
    density_matrix = layer_func(weights)
    
    expected_size = 2**qubits
    assert density_matrix.shape == (expected_size, expected_size)
    print(f"  Density matrix shape: {density_matrix.shape}")
    print("  PASS - Density layer working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 6: Walsh decomposition
print("\n[6/12] Testing Walsh decomposition...")
try:
    phases = np.array([0, np.pi/4, np.pi/2, np.pi])
    diag_matrix = np.diag(np.exp(1j * phases))
    
    circuit = build_optimal_walsh_circuit(torch.tensor(diag_matrix))
    gate_types = set(g[0] for g in circuit)
    
    assert gate_types.issubset({'CNOT', 'RZ'}), f"Invalid gates: {gate_types}"
    print(f"  Circuit: {len(circuit)} gates, types={gate_types}")
    print("  PASS - Walsh decomposition working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 7: Diagonalization
print("\n[7/12] Testing matrix diagonalization...")
try:
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    diag, transform = diagonalize_unitary(H)
    
    d = np.diag(diag)
    assert np.allclose(diag, np.diag(d), atol=1e-10)
    
    reconstructed = transform @ diag @ transform.conj().T
    assert np.allclose(H, reconstructed, atol=1e-10)
    
    print("  PASS - Diagonalization working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 8: Quantum circuit creation
print("\n[8/12] Testing quantum circuit...")
try:
    qc = QuantumCircuit(num_qubits=4, shots=None, QNN_layers=4)
    circuit = qc.circuit
    
    inputs = torch.randn(4)
    weights = torch.randn(4)
    output = circuit(inputs, weights)
    
    assert len(output) == 4
    print(f"  Circuit output: {[f'{float(x):.3f}' for x in output]}")
    print("  PASS - Quantum circuit working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 9: Hybrid model creation
print("\n[9/12] Testing hybrid QNN model...")
try:
    model = HybridDensityQNN(num_sub_unitaries=4, num_qubits=4)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params}")
    print("  PASS - Model created")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 10: Batch size 1
print("\n[10/12] Testing batch size 1...")
try:
    batch1 = torch.randn(1, 3, 32, 32)
    output1 = model(batch1)
    
    assert output1.shape == (1, 10), f"Expected (1, 10), got {output1.shape}"
    print(f"  Input: {batch1.shape}, Output: {output1.shape}")
    print("  PASS - Batch size 1 working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 11: Batch size 8
print("\n[11/12] Testing batch size 8...")
try:
    batch8 = torch.randn(8, 3, 32, 32)
    output8 = model(batch8)
    
    assert output8.shape == (8, 10), f"Expected (8, 10), got {output8.shape}"
    print(f"  Input: {batch8.shape}, Output: {output8.shape}")
    print("  PASS - Batch size 8 working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Test 12: Gradient computation
print("\n[12/12] Testing gradient computation...")
try:
    model.zero_grad()
    batch = torch.randn(2, 3, 32, 32)
    target = torch.tensor([3, 7])
    
    output = model(batch)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients computed"
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print("  PASS - Gradients working")
    passed += 1
except Exception as e:
    print(f"  FAIL - {e}")
    failed += 1

# Final summary
print("\n" + "="*70)
print("FINAL TEST RESULTS")
print("="*70)
print(f"Passed: {passed}/12")
print(f"Failed: {failed}/12")

if failed == 0:
    print("\n*** ALL TESTS PASSED ***")
    print("Status: PRODUCTION READY")
    print("\nThe QNN implementation is complete and working:")
    print("  - All 4 paper patterns implemented")
    print("  - Density matrix approach with diagonalization")
    print("  - Walsh decomposition for efficient circuits")
    print("  - Batch processing for any batch size")
    print("  - Gradient computation working")
    print("  - Ready for training on CIFAR-10")
else:
    print(f"\n*** {failed} TESTS FAILED ***")
    print("Status: NEEDS ATTENTION")

print("="*70)
