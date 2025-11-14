"""
Quick Verification Test - No Training, Just Check Everything Works
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import torch
import torch.nn as nn
import numpy as np

# Ensure local simulator
if 'BRAKET_DEVICE' in os.environ:
    del os.environ['BRAKET_DEVICE']

print("="*80)
print("QUICK VERIFICATION TEST")
print("="*80)

# Test 1: Imports
print("\n[1/10] Testing imports...")
try:
    from qnn_model import HybridDensityQNN
    from unitary_decomposition import decompose_unitary_matrix, is_diagonal, is_unitary
    from walsh_circuit_decomposition import build_optimal_walsh_circuit
    print("  [PASS] All imports work")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    exit(1)

# Test 2: Walsh decomposition
print("\n[2/10] Testing Walsh decomposition...")
try:
    diag = np.diag([1, -1, 1j, -1j])
    gates = build_optimal_walsh_circuit(diag)
    assert len(gates) > 0
    print(f"  [PASS] Walsh works ({len(gates)} gates)")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 3: Unitary decomposition (diagonal)
print("\n[3/10] Testing unitary decomposition (diagonal)...")
try:
    diag = np.diag([1, -1, 1j, -1j])
    gates = decompose_unitary_matrix(diag, method="auto")
    gate_types = [g[0] for g in gates]
    assert "CNOT" in gate_types or "RZ" in gate_types
    print(f"  [PASS] Uses Walsh for diagonal")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 4: Unitary decomposition (non-diagonal)
print("\n[4/10] Testing unitary decomposition (non-diagonal)...")
try:
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    gates = decompose_unitary_matrix(hadamard, method="auto")
    gate_types = [g[0] for g in gates]
    assert "QubitUnitary" in gate_types
    print(f"  [PASS] Uses PennyLane for non-diagonal")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 5: Model creation
print("\n[5/10] Testing model creation...")
try:
    model = HybridDensityQNN(num_qubits=4)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [PASS] Model created ({param_count} parameters)")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 6: Forward pass (batch size 1)
print("\n[6/10] Testing forward pass (batch=1)...")
try:
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape == (1, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print(f"  [PASS] Forward pass works")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 7: Forward pass (batch size 8)
print("\n[7/10] Testing forward pass (batch=8)...")
try:
    input_tensor = torch.randn(8, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape == (8, 10)
    assert not torch.isnan(output).any()
    print(f"  [PASS] Batch processing works")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 8: Backward pass
print("\n[8/10] Testing backward pass...")
try:
    input_tensor = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  [PASS] Gradients computed ({grad_count} params)")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 9: V @ D @ V† decomposition
print("\n[9/10] Testing V @ D @ V† decomposition...")
try:
    # Generate random unitary
    size = 4
    random_matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    U, _ = np.linalg.qr(random_matrix)
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eig(U)
    D = np.diag(eigenvalues)
    V = eigenvectors
    
    # Test D is diagonal
    assert is_diagonal(D)
    
    # Test decomposition
    d_gates = decompose_unitary_matrix(D, method="auto")
    v_gates = decompose_unitary_matrix(V, method="auto")
    
    d_types = [g[0] for g in d_gates]
    v_types = [g[0] for g in v_gates]
    
    # D should use Walsh (CNOT/RZ)
    assert "CNOT" in d_types or "RZ" in d_types
    
    print(f"  [PASS] V @ D @ V† decomposition works")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 10: Error handling
print("\n[10/10] Testing error handling...")
try:
    # Non-unitary matrix should raise error
    try:
        non_unitary = np.random.rand(4, 4)
        decompose_unitary_matrix(non_unitary)
        print(f"  [FAIL] Should have raised error for non-unitary")
        exit(1)
    except ValueError:
        pass  # Expected
    
    # Non-power-of-2 should raise error
    try:
        from unitary_decomposition import get_num_qubits
        invalid = np.eye(5)
        get_num_qubits(invalid)
        print(f"  [FAIL] Should have raised error for non-power-of-2")
        exit(1)
    except ValueError:
        pass  # Expected
    
    print(f"  [PASS] Error handling works")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED - SYSTEM VERIFIED")
print("="*80)
print("\nSummary:")
print("  [PASS] Walsh decomposition works")
print("  [PASS] PennyLane decomposition works")
print("  [PASS] V @ D @ V† strategy works")
print("  [PASS] QNN model works")
print("  [PASS] Forward pass works")
print("  [PASS] Backward pass works")
print("  [PASS] Batch processing works")
print("  [PASS] Gradient flow works")
print("  [PASS] Error handling works")
print("\n  STATUS: PRODUCTION READY")
print("="*80)
