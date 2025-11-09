import numpy as np
import pytest
from walsh_circuit_decomposition import gray_code, Walsh_coefficients, build_optimal_walsh_circuit

def test_gray_code():
    """Test gray code generation for different bit lengths"""
    # Test for 2-bit gray code
    assert gray_code(2) == [0, 1, 3, 2]
    # Test for 3-bit gray code
    assert gray_code(3) == [0, 1, 3, 2, 6, 7, 5, 4]

def test_walsh_coefficients():
    """Test Walsh coefficient calculation for diagonal matrices"""
    # Test with identity matrix (2x2)
    identity = np.array([[1, 0], [0, 1]])
    coeffs = Walsh_coefficients(identity)
    assert len(coeffs) == 2
    assert np.allclose(coeffs, 0, atol=1e-12)  # Identity matrix should have zero coefficients

    # Test with diagonal phase matrix
    phase_mat = np.array([[1, 0], [0, -1]])
    coeffs = Walsh_coefficients(phase_mat)
    assert len(coeffs) == 2
    # The coefficients should be computed correctly (sign may vary based on convention)
    assert np.allclose(np.abs(coeffs[1]), np.pi/2, atol=1e-12)

def test_build_optimal_walsh_circuit():
    """Test circuit construction for simple diagonal matrices"""
    # Test with a simple phase gate
    phase_mat = np.array([[1, 0], [0, -1]])
    circuit = build_optimal_walsh_circuit(phase_mat)
    
    # The circuit should contain a single RZ gate
    assert any(gate[0] == "RZ" for gate in circuit)
    
    # Test with identity matrix (should produce empty or minimal circuit)
    identity = np.array([[1, 0], [0, 1]])
    circuit = build_optimal_walsh_circuit(identity)
    # Should have minimal or no gates due to optimization
    assert len(circuit) == 0 or all(abs(gate[1][0]) < 1e-12 for gate in circuit if gate[0] == "RZ")

def test_invalid_input_walsh_coefficients():
    """Test error handling for invalid inputs in Walsh coefficient calculation"""
    # Test non-square matrix
    with pytest.raises(AssertionError):
        Walsh_coefficients(np.array([[1, 0], [0, 1], [0, 0]]))
    
    # Test non-power-of-2 sized matrix
    with pytest.raises(AssertionError):
        Walsh_coefficients(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    
    # Test non-diagonal matrix
    with pytest.raises(AssertionError):
        Walsh_coefficients(np.array([[1, 1], [0, 1]]))
    
    # Test non-unitary diagonal elements
    with pytest.raises(AssertionError):
        Walsh_coefficients(np.array([[2, 0], [0, 1]]))

def test_larger_circuit_construction():
    """Test circuit construction for 4x4 diagonal matrix"""
    # Create a 4x4 diagonal matrix with different phases
    matrix = np.diag([1, -1, 1j, -1j])
    circuit = build_optimal_walsh_circuit(matrix)
    
    # Verify circuit structure
    assert len(circuit) > 0
    # Check that gates are either CNOT or RZ
    for gate in circuit:
        assert gate[0] in ["CNOT", "RZ"]
        if gate[0] == "CNOT":
            assert len(gate[1]) == 2  # CNOT should have 2 qubits
        else:  # RZ gate
            assert len(gate[1]) == 2  # RZ should have angle and target qubit