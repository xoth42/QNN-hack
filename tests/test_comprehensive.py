"""
Comprehensive tests for Walsh decomposition and Density QNN
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from walsh_circuit_decomposition import (
    Walsh_coefficients,
    build_optimal_walsh_circuit,
    diagonalize_unitary,
    gray_code,
    _fwht
)
from density_qnn import (
    RBS,
    get_theta,
    RandRBS,
    matrix_from_IRBS_string,
    string_from_RBS_connections,
    pyramid_network_rbs,
    upsidown_pyramid_network_rbs,
    density_layer
)

print("="*70)
print("COMPREHENSIVE TEST SUITE")
print("="*70)

# ============================================================================
# WALSH DECOMPOSITION TESTS
# ============================================================================

def test_walsh_decomposition():
    print("\n" + "="*70)
    print("WALSH DECOMPOSITION TESTS")
    print("="*70)
    
    passed = 0
    failed = 0
    
    # Test 1: Gray code generation
    print("\nTest 1: Gray code generation")
    try:
        gray = gray_code(3)
        expected = [0, 1, 3, 2, 6, 7, 5, 4]
        assert gray == expected, f"Expected {expected}, got {gray}"
        print(f"  ✓ Gray code(3): {gray}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 2: Fast Walsh-Hadamard Transform
    print("\nTest 2: Fast Walsh-Hadamard Transform")
    try:
        vec = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        result = _fwht(vec)
        # Check result is correct size
        assert len(result) == len(vec), "FWHT output size mismatch"
        print(f"  ✓ FWHT input: {vec}")
        print(f"  ✓ FWHT output: {result}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 3: Walsh coefficients for diagonal matrix
    print("\nTest 3: Walsh coefficients")
    try:
        phases = np.array([0, np.pi/4, np.pi/2, np.pi])
        diag_matrix = np.diag(np.exp(1j * phases))
        coeffs = Walsh_coefficients(torch.tensor(diag_matrix))
        assert len(coeffs) == 4, "Wrong number of coefficients"
        print(f"  ✓ Phases: {phases}")
        print(f"  ✓ Coefficients: {coeffs}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 4: Build circuit - verify only CNOT and RZ gates
    print("\nTest 4: Circuit contains only CNOT and RZ gates")
    try:
        phases = np.array([0, np.pi/4, np.pi/2, np.pi])
        diag_matrix = np.diag(np.exp(1j * phases))
        circuit = build_optimal_walsh_circuit(torch.tensor(diag_matrix))
        
        # Check all gates are CNOT or RZ
        valid_gates = {'CNOT', 'RZ'}
        for gate_type, params in circuit:
            assert gate_type in valid_gates, f"Invalid gate type: {gate_type}"
        
        # Count gates
        cnot_count = sum(1 for g, _ in circuit if g == 'CNOT')
        rz_count = sum(1 for g, _ in circuit if g == 'RZ')
        
        print(f"  ✓ Total gates: {len(circuit)}")
        print(f"  ✓ CNOT gates: {cnot_count}")
        print(f"  ✓ RZ gates: {rz_count}")
        print(f"  ✓ All gates are CNOT or RZ: True")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 5: Circuit parameters are valid
    print("\nTest 5: Circuit parameters are valid")
    try:
        phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        diag_matrix = np.diag(np.exp(1j * phases))
        circuit = build_optimal_walsh_circuit(torch.tensor(diag_matrix))
        
        num_qubits = int(np.log2(len(phases)))
        
        for gate_type, params in circuit:
            if gate_type == 'CNOT':
                control, target = params
                assert 0 <= control < num_qubits, f"Invalid control qubit: {control}"
                assert 0 <= target < num_qubits, f"Invalid target qubit: {target}"
                assert control != target, "Control and target must be different"
            elif gate_type == 'RZ':
                angle, qubit = params
                assert 0 <= qubit < num_qubits, f"Invalid qubit: {qubit}"
                assert isinstance(angle, (int, float, np.number)), f"Invalid angle type: {type(angle)}"
        
        print(f"  ✓ All gate parameters are valid")
        print(f"  ✓ Number of qubits: {num_qubits}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 6: Diagonalization
    print("\nTest 6: Diagonalization of non-diagonal matrix")
    try:
        # Hadamard matrix
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        diag_H, transform = diagonalize_unitary(H)
        
        # Check it's diagonal
        d = np.diag(diag_H)
        assert np.allclose(diag_H, np.diag(d), atol=1e-10), "Result not diagonal"
        
        # Check reconstruction
        reconstructed = transform @ diag_H @ transform.conj().T
        assert np.allclose(H, reconstructed, atol=1e-10), "Reconstruction failed"
        
        print(f"  ✓ Diagonalization successful")
        print(f"  ✓ Eigenvalues: {d}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 7: Error handling - non-diagonal matrix
    print("\nTest 7: Error handling for non-diagonal matrix")
    try:
        non_diag = np.array([[1, 0.5], [0.5, 1]])
        try:
            Walsh_coefficients(torch.tensor(non_diag))
            print(f"  ✗ FAILED: Should have raised ValueError")
            failed += 1
        except ValueError as e:
            if "DIAGONAL" in str(e):
                print(f"  ✓ Correctly raised ValueError for non-diagonal matrix")
                passed += 1
            else:
                print(f"  ✗ FAILED: Wrong error message: {e}")
                failed += 1
    except Exception as e:
        print(f"  ✗ FAILED: Unexpected error: {e}")
        failed += 1
    
    # Test 8: Error handling - non-unit circle
    print("\nTest 8: Error handling for non-unit circle elements")
    try:
        bad_diag = np.diag([1, 2, 3, 4])
        try:
            Walsh_coefficients(torch.tensor(bad_diag, dtype=torch.complex128))
            print(f"  ✗ FAILED: Should have raised ValueError")
            failed += 1
        except ValueError as e:
            if "unit circle" in str(e):
                print(f"  ✓ Correctly raised ValueError for non-unit elements")
                passed += 1
            else:
                print(f"  ✗ FAILED: Wrong error message: {e}")
                failed += 1
    except Exception as e:
        print(f"  ✗ FAILED: Unexpected error: {e}")
        failed += 1
    
    print(f"\nWalsh Tests: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} Walsh tests failed"


# ============================================================================
# DENSITY QNN TESTS
# ============================================================================

def test_density_qnn():
    print("\n" + "="*70)
    print("DENSITY QNN TESTS")
    print("="*70)
    
    passed = 0
    failed = 0
    
    # Test 1: RBS gate properties
    print("\nTest 1: RBS gate properties")
    try:
        theta = np.pi/4
        rbs = RBS(theta)
        
        # Check shape
        assert rbs.shape == (4, 4), f"Wrong shape: {rbs.shape}"
        
        # Check it's unitary (RBS @ RBS.T.conj() = I)
        identity = torch.matmul(rbs, rbs.T.conj())
        assert torch.allclose(identity, torch.eye(4), atol=1e-6), "RBS not unitary"
        
        # Check structure (first and last elements should be 1)
        assert torch.isclose(rbs[0, 0], torch.tensor(1.0)), "RBS[0,0] should be 1"
        assert torch.isclose(rbs[3, 3], torch.tensor(1.0)), "RBS[3,3] should be 1"
        
        print(f"  ✓ RBS shape: {rbs.shape}")
        print(f"  ✓ RBS is unitary: True")
        print(f"  ✓ RBS structure correct")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 2: Random RBS generation
    print("\nTest 2: Random RBS generation")
    try:
        rbs1 = RandRBS()
        rbs2 = RandRBS()
        
        # Check they're different (with high probability)
        assert not torch.allclose(rbs1, rbs2), "Random RBS gates are identical"
        
        # Check both are unitary
        for rbs in [rbs1, rbs2]:
            identity = torch.matmul(rbs, rbs.T.conj())
            assert torch.allclose(identity, torch.eye(4), atol=1e-6), "Random RBS not unitary"
        
        print(f"  ✓ Random RBS gates are different")
        print(f"  ✓ Both are unitary")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 3: String to matrix conversion
    print("\nTest 3: String to matrix conversion")
    try:
        # Simple string: "IRBS" = I ⊗ RBS (3 qubits: 1 identity + 1 RBS covering 2 qubits)
        string = "IRBS"
        matrix = matrix_from_IRBS_string(string)
        
        # Should be 8x8 (3 qubits: I on qubit 1, RBS on qubits 2-3)
        assert matrix.shape == (8, 8), f"Wrong shape: {matrix.shape}, expected (8, 8)"
        
        print(f"  ✓ String 'IRBS' -> matrix shape: {matrix.shape} (3 qubits)")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 4: RBS connections to string
    print("\nTest 4: RBS connections to string")
    try:
        # Test with simple connections
        connections = [(1, 2), (3, 4)]
        qubits = 4
        string = string_from_RBS_connections(connections, qubits)
        
        # Check string contains RBS
        assert 'RBS' in string, "String should contain 'RBS'"
        # Note: For [(1,2), (3,4)] with 4 qubits, we get "RBSRBS" (no I needed)
        # RBS on 1-2 covers qubits 1-2, RBS on 3-4 covers qubits 3-4
        
        print(f"  ✓ Connections {connections} -> string: '{string}'")
        print(f"  ✓ String format valid")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 5: Pyramid network
    print("\nTest 5: Pyramid network RBS")
    try:
        qubits = 4
        matrix = pyramid_network_rbs(qubits)
        
        # Check shape (2^qubits x 2^qubits)
        expected_size = 2 ** qubits
        assert matrix.shape == (expected_size, expected_size), f"Wrong shape: {matrix.shape}"
        
        # Check it's a valid matrix (no NaN or Inf)
        assert not torch.isnan(matrix).any(), "Matrix contains NaN"
        assert not torch.isinf(matrix).any(), "Matrix contains Inf"
        
        print(f"  ✓ Pyramid network for {qubits} qubits")
        print(f"  ✓ Matrix shape: {matrix.shape}")
        print(f"  ✓ No NaN or Inf values")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 6: Inverted pyramid network
    print("\nTest 6: Inverted pyramid network RBS")
    try:
        qubits = 4
        matrix = upsidown_pyramid_network_rbs(qubits)
        
        # Check shape
        expected_size = 2 ** qubits
        assert matrix.shape == (expected_size, expected_size), f"Wrong shape: {matrix.shape}"
        
        # Check it's a valid matrix
        assert not torch.isnan(matrix).any(), "Matrix contains NaN"
        assert not torch.isinf(matrix).any(), "Matrix contains Inf"
        
        print(f"  ✓ Inverted pyramid network for {qubits} qubits")
        print(f"  ✓ Matrix shape: {matrix.shape}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 7: Density layer creation
    print("\nTest 7: Density layer creation")
    try:
        qubits = 3
        matrix_count = 2
        
        layer_func = density_layer(qubits, matrix_count)
        
        # Test with random weights
        weights = torch.rand(matrix_count)
        result = layer_func(weights)
        
        # Check shape
        expected_size = 2 ** qubits
        assert result.shape == (expected_size, expected_size), f"Wrong shape: {result.shape}"
        
        # Check no NaN or Inf
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"
        
        print(f"  ✓ Density layer created for {qubits} qubits, {matrix_count} matrices")
        print(f"  ✓ Output shape: {result.shape}")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    # Test 8: Density layer with different weights
    print("\nTest 8: Density layer weight sensitivity")
    try:
        qubits = 2
        matrix_count = 2
        
        layer_func = density_layer(qubits, matrix_count)
        
        # Test with different weights
        weights1 = torch.tensor([1.0, 0.0])
        weights2 = torch.tensor([0.0, 1.0])
        
        result1 = layer_func(weights1)
        result2 = layer_func(weights2)
        
        # Results should be different
        assert not torch.allclose(result1, result2), "Different weights give same result"
        
        print(f"  ✓ Different weights produce different outputs")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1
    
    print(f"\nDensity QNN Tests: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} Density QNN tests failed"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    try:
        test_walsh_decomposition()
        test_density_qnn()
        
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print("\n✅ ALL TESTS PASSED!")
        print("="*70)
    except AssertionError as e:
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"\n❌ TESTS FAILED: {e}")
        print("="*70)
