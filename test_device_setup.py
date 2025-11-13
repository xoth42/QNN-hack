"""Test that quantum device setup works without AWS credentials"""
import os
import sys

# Make sure BRAKET_DEVICE is not set
if 'BRAKET_DEVICE' in os.environ:
    del os.environ['BRAKET_DEVICE']

from qnn_model import QuantumCircuit

print("="*70)
print("Testing Quantum Device Setup (Local Simulator)")
print("="*70)

try:
    # Test with default settings (should use local simulator)
    qc = QuantumCircuit(num_qubits=4, shots=100)
    print(f"\n✅ SUCCESS: Quantum circuit initialized")
    print(f"   Device: {qc.device}")
    print(f"   Qubits: {qc.num_qubits}")
    print(f"   Shots: {qc.shots}")
    
    # Test circuit creation
    circuit = qc.circuit
    print(f"\n✅ SUCCESS: Quantum circuit callable created")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Local simulator working correctly")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
