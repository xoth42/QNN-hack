"""
Final verification script - Quick sanity checks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from walsh_circuit_decomposition import build_optimal_walsh_circuit, Walsh_coefficients
from density_qnn import RBS, pyramid_network_rbs, upsidown_pyramid_network_rbs, density_layer

print("="*70)
print("FINAL VERIFICATION - QUICK SANITY CHECKS")
print("="*70)

all_passed = True

# Test 1: Walsh decomposition outputs only CNOT and RZ
print("\n1. Walsh decomposition gate types...")
try:
    phases = np.array([0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
    matrix = np.diag(np.exp(1j * phases))
    circuit = build_optimal_walsh_circuit(torch.tensor(matrix))
    
    gate_types = set(g[0] for g in circuit)
    valid_gates = {'CNOT', 'RZ'}
    
    if gate_types.issubset(valid_gates):
        print(f"   ✅ PASS - Only CNOT and RZ gates: {gate_types}")
    else:
        print(f"   ❌ FAIL - Invalid gates found: {gate_types}")
        all_passed = False
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    all_passed = False

# Test 2: RBS is unitary
print("\n2. RBS gate is unitary...")
try:
    rbs = RBS(np.pi/4)
    identity = torch.matmul(rbs, rbs.T.conj())
    is_unitary = torch.allclose(identity, torch.eye(4), atol=1e-6)
    
    if is_unitary:
        print(f"   ✅ PASS - RBS is unitary")
    else:
        print(f"   ❌ FAIL - RBS is not unitary")
        all_passed = False
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    all_passed = False

# Test 3: Pyramid networks produce correct dimensions
print("\n3. Pyramid networks dimensions...")
try:
    for qubits in [2, 4, 8]:
        m1 = pyramid_network_rbs(qubits)
        m2 = upsidown_pyramid_network_rbs(qubits)
        expected_size = 2**qubits
        
        if m1.shape == (expected_size, expected_size) and m2.shape == (expected_size, expected_size):
            print(f"   ✅ PASS - {qubits} qubits: {m1.shape}")
        else:
            print(f"   ❌ FAIL - {qubits} qubits: expected ({expected_size}, {expected_size}), got {m1.shape}")
            all_passed = False
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    all_passed = False

# Test 4: Density layer works
print("\n4. Density layer functionality...")
try:
    layer = density_layer(3, 2)
    weights = torch.tensor([0.3, 0.7])
    result = layer(weights)
    expected_size = 2**3
    
    if result.shape == (expected_size, expected_size):
        print(f"   ✅ PASS - Density layer output: {result.shape}")
    else:
        print(f"   ❌ FAIL - Expected ({expected_size}, {expected_size}), got {result.shape}")
        all_passed = False
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    all_passed = False

# Test 5: Walsh coefficients computation
print("\n5. Walsh coefficients computation...")
try:
    phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    matrix = np.diag(np.exp(1j * phases))
    coeffs = Walsh_coefficients(torch.tensor(matrix))
    
    if len(coeffs) == 4 and not np.isnan(coeffs).any():
        print(f"   ✅ PASS - Coefficients computed: {coeffs}")
    else:
        print(f"   ❌ FAIL - Invalid coefficients")
        all_passed = False
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    all_passed = False

# Test 6: Error handling
print("\n6. Error handling for invalid inputs...")
try:
    # Non-diagonal matrix should raise error
    non_diag = np.array([[1, 0.5], [0.5, 1]])
    try:
        Walsh_coefficients(torch.tensor(non_diag))
        print(f"   ❌ FAIL - Should have raised error for non-diagonal matrix")
        all_passed = False
    except ValueError:
        print(f"   ✅ PASS - Correctly rejects non-diagonal matrix")
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    all_passed = False

# Final summary
print("\n" + "="*70)
if all_passed:
    print("✅ ALL VERIFICATION CHECKS PASSED!")
    print("\nSystem Status: READY FOR PRODUCTION")
else:
    print("❌ SOME CHECKS FAILED")
    print("\nSystem Status: NEEDS ATTENTION")
print("="*70)
