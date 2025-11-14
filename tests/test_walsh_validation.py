"""
Test script to verify Walsh decomposition validation.

This script tests that build_optimal_walsh_circuit() correctly:
1. Accepts diagonal matrices
2. Rejects non-diagonal matrices with clear error messages
3. Works with various matrix sizes
"""

import numpy as np
import torch
from walsh_circuit_decomposition import build_optimal_walsh_circuit


def test_diagonal_matrix_accepted():
    """Test that diagonal matrices are accepted."""
    print("Test 1: Diagonal matrix (should succeed)")
    
    # Create a diagonal unitary matrix
    diag_matrix = np.diag([1, -1, 1j, -1j])
    
    try:
        gates = build_optimal_walsh_circuit(diag_matrix)
        print(f"  ✓ Success: Generated {len(gates)} gates")
        return True
    except ValueError as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_non_diagonal_matrix_rejected():
    """Test that non-diagonal matrices are rejected."""
    print("\nTest 2: Non-diagonal matrix (should raise ValueError)")
    
    # Create a non-diagonal unitary matrix (Hadamard)
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    try:
        gates = build_optimal_walsh_circuit(hadamard)
        print(f"  ✗ Failed: Should have raised ValueError but got {len(gates)} gates")
        return False
    except ValueError as e:
        print(f"  ✓ Success: Raised ValueError as expected")
        print(f"  Error message: {str(e)[:100]}...")
        return True


def test_identity_matrix_accepted():
    """Test that identity matrix (diagonal) is accepted."""
    print("\nTest 3: Identity matrix (should succeed)")
    
    identity = np.eye(4)
    
    try:
        gates = build_optimal_walsh_circuit(identity)
        print(f"  ✓ Success: Generated {len(gates)} gates")
        return True
    except ValueError as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_torch_tensor_diagonal():
    """Test that diagonal torch tensors are accepted."""
    print("\nTest 4: Diagonal torch tensor (should succeed)")
    
    diag_matrix = torch.diag(torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.complex64))
    
    try:
        gates = build_optimal_walsh_circuit(diag_matrix)
        print(f"  ✓ Success: Generated {len(gates)} gates")
        return True
    except ValueError as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_torch_tensor_non_diagonal():
    """Test that non-diagonal torch tensors are rejected."""
    print("\nTest 5: Non-diagonal torch tensor (should raise ValueError)")
    
    # Create a non-diagonal matrix
    matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.complex64)
    
    try:
        gates = build_optimal_walsh_circuit(matrix)
        print(f"  ✗ Failed: Should have raised ValueError but got {len(gates)} gates")
        return False
    except ValueError as e:
        print(f"  ✓ Success: Raised ValueError as expected")
        return True


def test_larger_diagonal_matrix():
    """Test with larger diagonal matrix (8x8)."""
    print("\nTest 6: Larger diagonal matrix 8x8 (should succeed)")
    
    # Create 8x8 diagonal matrix (3 qubits)
    phases = np.random.rand(8) * 2 * np.pi
    diag_matrix = np.diag(np.exp(1j * phases))
    
    try:
        gates = build_optimal_walsh_circuit(diag_matrix)
        print(f"  ✓ Success: Generated {len(gates)} gates")
        return True
    except ValueError as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_error_message_content():
    """Test that error message provides helpful guidance."""
    print("\nTest 7: Error message content (should mention decompose_unitary_matrix)")
    
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    try:
        gates = build_optimal_walsh_circuit(hadamard)
        print(f"  ✗ Failed: Should have raised ValueError")
        return False
    except ValueError as e:
        error_msg = str(e)
        has_diagonal_mention = "diagonal" in error_msg.lower()
        has_alternative = "decompose_unitary_matrix" in error_msg
        
        if has_diagonal_mention and has_alternative:
            print(f"  ✓ Success: Error message is helpful")
            print(f"    - Mentions 'diagonal': {has_diagonal_mention}")
            print(f"    - Suggests alternative: {has_alternative}")
            return True
        else:
            print(f"  ✗ Failed: Error message missing key information")
            print(f"    - Mentions 'diagonal': {has_diagonal_mention}")
            print(f"    - Suggests alternative: {has_alternative}")
            return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Walsh Decomposition Validation Tests")
    print("=" * 70)
    
    tests = [
        test_diagonal_matrix_accepted,
        test_non_diagonal_matrix_rejected,
        test_identity_matrix_accepted,
        test_torch_tensor_diagonal,
        test_torch_tensor_non_diagonal,
        test_larger_diagonal_matrix,
        test_error_message_content,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
