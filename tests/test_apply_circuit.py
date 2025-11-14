"""
Test script for apply_decomposed_circuit function.

This script verifies that:
1. CNOT gates are applied correctly (self-inverse)
2. RZ gates are applied correctly (angle negation for inverse)
3. QubitUnitary gates are applied correctly (conjugate transpose for inverse)
4. Forward and inverse operations correctly reconstruct identity
"""

import sys
import io
# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pennylane as qml
from unitary_decomposition import decompose_unitary_matrix, apply_decomposed_circuit


def test_cnot_self_inverse():
    """Test that CNOT is self-inverse: CNOT @ CNOT = I"""
    print("\n=== Test 1: CNOT Self-Inverse ===")
    
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def circuit():
        # Start with |01⟩ state
        qml.PauliX(wires=1)
        
        # Apply CNOT twice (should return to |01⟩)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        
        return qml.state()
    
    state = circuit()
    expected = np.array([0, 1, 0, 0])  # |01⟩
    
    if np.allclose(state, expected, atol=1e-6):
        print("✓ CNOT self-inverse test PASSED")
        return True
    else:
        print(f"✗ CNOT self-inverse test FAILED")
        print(f"  Expected: {expected}")
        print(f"  Got: {state}")
        return False


def test_rz_inverse():
    """Test that RZ(-θ) is the inverse of RZ(θ)"""
    print("\n=== Test 2: RZ Inverse ===")
    
    dev = qml.device('default.qubit', wires=1)
    
    @qml.qnode(dev)
    def circuit(angle):
        # Start with |+⟩ state (superposition)
        qml.Hadamard(wires=0)
        
        # Apply RZ(θ) then RZ(-θ)
        qml.RZ(angle, wires=0)
        qml.RZ(-angle, wires=0)
        
        # Should still be in |+⟩
        qml.Hadamard(wires=0)  # Transform back to |0⟩
        
        return qml.probs(wires=0)
    
    probs = circuit(np.pi / 4)
    
    # Should be in |0⟩ state (probability [1, 0])
    if np.allclose(probs[0], 1.0, atol=1e-6):
        print("✓ RZ inverse test PASSED")
        return True
    else:
        print(f"✗ RZ inverse test FAILED")
        print(f"  Expected P(|0⟩) = 1.0")
        print(f"  Got P(|0⟩) = {probs[0]}")
        return False


def test_qubit_unitary_inverse():
    """Test that QubitUnitary(U†) is the inverse of QubitUnitary(U)"""
    print("\n=== Test 3: QubitUnitary Inverse ===")
    
    dev = qml.device('default.qubit', wires=2)
    
    # Create a random unitary matrix (2-qubit)
    from scipy.stats import unitary_group
    U = unitary_group.rvs(4)  # Random 4×4 unitary
    
    @qml.qnode(dev)
    def circuit():
        # Start with |10⟩ state
        qml.PauliX(wires=0)
        
        # Apply U then U†
        qml.QubitUnitary(U, wires=[0, 1])
        qml.QubitUnitary(np.conj(U.T), wires=[0, 1])
        
        return qml.state()
    
    state = circuit()
    expected = np.array([0, 0, 1, 0])  # |10⟩
    
    if np.allclose(state, expected, atol=1e-6):
        print("✓ QubitUnitary inverse test PASSED")
        return True
    else:
        print(f"✗ QubitUnitary inverse test FAILED")
        print(f"  Expected: {expected}")
        print(f"  Got: {state}")
        print(f"  Error: {np.linalg.norm(state - expected)}")
        return False


def test_apply_decomposed_circuit_forward():
    """Test apply_decomposed_circuit with forward operation"""
    print("\n=== Test 4: Apply Decomposed Circuit (Forward) ===")
    
    dev = qml.device('default.qubit', wires=2)
    
    # Create a simple diagonal matrix
    D = np.diag([1, -1, 1j, -1j])
    
    # Decompose it
    gates = decompose_unitary_matrix(D, method="walsh")
    
    @qml.qnode(dev)
    def circuit_with_apply():
        qml.PauliX(wires=1)  # Start with |01⟩
        apply_decomposed_circuit(gates, inverse=False)
        return qml.state()
    
    @qml.qnode(dev)
    def circuit_direct():
        qml.PauliX(wires=1)  # Start with |01⟩
        qml.QubitUnitary(D, wires=[0, 1])
        return qml.state()
    
    state_apply = circuit_with_apply()
    state_direct = circuit_direct()
    
    if np.allclose(state_apply, state_direct, atol=1e-6):
        print("✓ Apply decomposed circuit (forward) test PASSED")
        return True
    else:
        print(f"✗ Apply decomposed circuit (forward) test FAILED")
        print(f"  Error: {np.linalg.norm(state_apply - state_direct)}")
        return False


def test_apply_decomposed_circuit_inverse():
    """Test apply_decomposed_circuit with inverse operation"""
    print("\n=== Test 5: Apply Decomposed Circuit (Inverse) ===")
    
    dev = qml.device('default.qubit', wires=2)
    
    # Create a simple diagonal matrix
    D = np.diag([1, -1, 1j, -1j])
    
    # Decompose it
    gates = decompose_unitary_matrix(D, method="walsh")
    
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=1)  # Start with |01⟩
        
        # Apply D then D†
        apply_decomposed_circuit(gates, inverse=False)
        apply_decomposed_circuit(gates, inverse=True)
        
        return qml.state()
    
    state = circuit()
    expected = np.array([0, 1, 0, 0])  # Should return to |01⟩
    
    if np.allclose(state, expected, atol=1e-6):
        print("✓ Apply decomposed circuit (inverse) test PASSED")
        return True
    else:
        print(f"✗ Apply decomposed circuit (inverse) test FAILED")
        print(f"  Expected: {expected}")
        print(f"  Got: {state}")
        print(f"  Error: {np.linalg.norm(state - expected)}")
        return False


def test_vdv_decomposition():
    """Test full VDV† decomposition using apply_decomposed_circuit"""
    print("\n=== Test 6: Full VDV† Decomposition ===")
    
    dev = qml.device('default.qubit', wires=2)
    
    # Create a random unitary and diagonalize it
    from scipy.stats import unitary_group
    U = unitary_group.rvs(4)
    
    # Diagonalize: U = V @ D @ V†
    eigenvalues, V = np.linalg.eig(U)
    D = np.diag(eigenvalues)
    
    # Decompose V and D
    v_gates = decompose_unitary_matrix(V, method="pennylane")
    d_gates = decompose_unitary_matrix(D, method="walsh")
    
    @qml.qnode(dev)
    def circuit_decomposed():
        qml.PauliX(wires=0)  # Start with |10⟩
        
        # Apply V @ D @ V†
        apply_decomposed_circuit(v_gates, inverse=False)
        apply_decomposed_circuit(d_gates, inverse=False)
        apply_decomposed_circuit(v_gates, inverse=True)
        
        return qml.state()
    
    @qml.qnode(dev)
    def circuit_direct():
        qml.PauliX(wires=0)  # Start with |10⟩
        qml.QubitUnitary(U, wires=[0, 1])
        return qml.state()
    
    state_decomposed = circuit_decomposed()
    state_direct = circuit_direct()
    
    error = np.linalg.norm(state_decomposed - state_direct)
    
    if error < 1e-6:
        print(f"✓ VDV† decomposition test PASSED (error: {error:.2e})")
        return True
    else:
        print(f"✗ VDV† decomposition test FAILED")
        print(f"  Reconstruction error: {error:.2e}")
        print(f"  Expected error < 1e-6")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing apply_decomposed_circuit Implementation")
    print("=" * 60)
    
    tests = [
        test_cnot_self_inverse,
        test_rz_inverse,
        test_qubit_unitary_inverse,
        test_apply_decomposed_circuit_forward,
        test_apply_decomposed_circuit_inverse,
        test_vdv_decomposition,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
