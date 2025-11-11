"""
Simple test to verify the quantum layer works end-to-end.
This test ensures the QNN can be created and run a forward pass.
"""
import os
import torch

# Ensure we use local simulator
if 'BRAKET_DEVICE' in os.environ:
    del os.environ['BRAKET_DEVICE']

print("="*70)
print("Testing Quantum Neural Network Layer")
print("="*70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from qnn_model import QuantumCircuit, HybridDensityQNN
    print("   ✅ Imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 2: Create quantum circuit
print("\n2. Testing quantum circuit creation...")
try:
    qc = QuantumCircuit(num_qubits=4, shots=None, QNN_layers=2)
    print("   ✅ Quantum circuit created")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Get circuit callable
print("\n3. Testing circuit callable...")
try:
    circuit = qc.circuit
    print("   ✅ Circuit callable obtained")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Run circuit with sample data
print("\n4. Testing circuit execution...")
try:
    inputs = torch.randn(4)
    weights = torch.randn(2)  # 2 QNN layers
    output = circuit(inputs, weights)
    print(f"   ✅ Circuit executed")
    print(f"   Output length: {len(output)}")
    print(f"   Output values: {[float(x) for x in output]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Create full hybrid model
print("\n5. Testing full hybrid QNN model...")
try:
    model = HybridDensityQNN(num_sub_unitaries=2, num_qubits=4)
    print("   ✅ Hybrid model created")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Forward pass with batch
print("\n6. Testing forward pass with batch...")
try:
    batch = torch.randn(2, 3, 32, 32)  # 2 CIFAR-10 images
    output = model(batch)
    print(f"   ✅ Forward pass successful")
    print(f"   Input shape: {batch.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Gradient computation
print("\n7. Testing gradient computation...")
try:
    batch = torch.randn(1, 3, 32, 32)
    target = torch.tensor([5])  # Class 5
    
    output = model(batch)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check if gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients computed"
    
    print(f"   ✅ Gradients computed")
    print(f"   Loss: {loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - QNN Layer is working!")
print("="*70)
print("\nThe quantum neural network layer is ready for training.")
