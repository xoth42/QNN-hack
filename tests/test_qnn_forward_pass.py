"""
Integration test for complete QNN forward pass.

Tests QNN initialization and forward pass with the new decomposition approach.
Verifies that the model produces valid outputs without NaN or Inf values.
"""
import os
import torch
import numpy as np

# Ensure local simulator for testing
if 'BRAKET_DEVICE' in os.environ:
    del os.environ['BRAKET_DEVICE']

print("="*70)
print("QNN FORWARD PASS INTEGRATION TEST")
print("="*70)

# Import QNN model
from qnn_model import HybridDensityQNN


def test_qnn_initialization(num_qubits=4, num_sub_unitaries=10):
    """Test QNN model initialization with new decomposition."""
    print(f"\n1. Testing QNN initialization (qubits={num_qubits}, sub_unitaries={num_sub_unitaries})...")
    
    try:
        model = HybridDensityQNN(
            num_sub_unitaries=num_sub_unitaries,
            num_qubits=num_qubits
        )
        
        # Verify model structure
        assert hasattr(model, 'conv1'), "Missing conv1 layer"
        assert hasattr(model, 'conv2'), "Missing conv2 layer"
        assert hasattr(model, 'fc1'), "Missing fc1 layer"
        assert hasattr(model, 'qlayer'), "Missing quantum layer"
        assert hasattr(model, 'fc2'), "Missing fc2 layer"
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model initialized successfully")
        print(f"      - Total parameters: {num_params}")
        print(f"      - Qubits: {num_qubits}")
        print(f"      - Sub-unitaries: {num_sub_unitaries}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_forward_pass_single_sample(model):
    """Test forward pass with single sample."""
    print(f"\n2. Testing forward pass (single sample)...")
    
    try:
        # Create single CIFAR-10 sized image
        # Note: batch_size=1 is required due to known batching limitation
        single = torch.randn(1, 3, 32, 32)
        
        # Forward pass
        output = model(single)
        
        # Verify output shape
        expected_shape = (1, 10)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        print(f"   ✅ Forward pass successful")
        print(f"      - Input shape: {single.shape}")
        print(f"      - Output shape: {output.shape}")
        
        return output
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_forward_pass_multiple_samples(model, num_samples=2):
    """Test forward pass with multiple samples (processed individually)."""
    print(f"\n3. Testing forward pass (multiple samples, processed individually)...")
    
    try:
        # Process multiple samples one at a time
        # This is the current working approach due to batching limitation
        outputs = []
        
        for i in range(num_samples):
            single = torch.randn(1, 3, 32, 32)
            output = model(single)
            outputs.append(output)
        
        # Concatenate outputs
        all_outputs = torch.cat(outputs, dim=0)
        
        # Verify shape
        expected_shape = (num_samples, 10)
        assert all_outputs.shape == expected_shape, f"Expected {expected_shape}, got {all_outputs.shape}"
        
        print(f"   ✅ Multiple samples processed successfully")
        print(f"      - Samples processed: {num_samples}")
        print(f"      - Combined output shape: {all_outputs.shape}")
        
        return all_outputs
        
    except Exception as e:
        print(f"   ❌ Multiple sample processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_output_validity(output):
    """Verify outputs are valid (no NaN, no Inf)."""
    print(f"\n4. Testing output validity...")
    
    try:
        # Check for NaN
        has_nan = torch.isnan(output).any().item()
        assert not has_nan, "Output contains NaN values"
        
        # Check for Inf
        has_inf = torch.isinf(output).any().item()
        assert not has_inf, "Output contains Inf values"
        
        # Check output range is reasonable
        output_min = output.min().item()
        output_max = output.max().item()
        output_mean = output.mean().item()
        output_std = output.std().item()
        
        print(f"   ✅ Output is valid")
        print(f"      - No NaN: ✓")
        print(f"      - No Inf: ✓")
        print(f"      - Range: [{output_min:.4f}, {output_max:.4f}]")
        print(f"      - Mean: {output_mean:.4f}, Std: {output_std:.4f}")
        
    except Exception as e:
        print(f"   ❌ Output validation failed: {e}")
        raise


def test_different_qubit_counts():
    """Test with different numbers of qubits."""
    print(f"\n5. Testing different qubit counts...")
    
    qubit_configs = [4, 7]
    
    for num_qubits in qubit_configs:
        try:
            print(f"\n   Testing {num_qubits} qubits...")
            
            # Initialize model
            model = HybridDensityQNN(
                num_sub_unitaries=4,  # Use fewer sub-unitaries for speed
                num_qubits=num_qubits
            )
            
            # Test forward pass (single sample due to batching limitation)
            single = torch.randn(1, 3, 32, 32)
            output = model(single)
            
            # Verify shape
            expected_shape = (1, 10)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            # Verify validity
            assert not torch.isnan(output).any(), f"NaN in output for {num_qubits} qubits"
            assert not torch.isinf(output).any(), f"Inf in output for {num_qubits} qubits"
            
            print(f"      ✅ {num_qubits} qubits: PASSED")
            
        except Exception as e:
            print(f"      ❌ {num_qubits} qubits: FAILED - {e}")
            raise


def test_gradient_flow(model):
    """Test that gradients flow through the model."""
    print(f"\n6. Testing gradient flow...")
    
    try:
        model.zero_grad()
        
        # Create input and target (single sample)
        single = torch.randn(1, 3, 32, 32)
        target = torch.tensor([3])
        
        # Forward pass
        output = model(single)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_grads = False
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                grad_norm = param.grad.norm().item()
                grad_norms.append((name, grad_norm))
        
        assert has_grads, "No gradients computed"
        
        print(f"   ✅ Gradients computed successfully")
        print(f"      - Loss: {loss.item():.4f}")
        print(f"      - Parameters with gradients: {len(grad_norms)}")
        
        # Show a few gradient norms
        print(f"      - Sample gradient norms:")
        for name, norm in grad_norms[:3]:
            print(f"        {name}: {norm:.6f}")
        
    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_deterministic_output(model):
    """Test that same input produces same output (deterministic)."""
    print(f"\n7. Testing deterministic output...")
    
    try:
        # Create a fixed input
        torch.manual_seed(42)
        single = torch.randn(1, 3, 32, 32)
        
        # Run twice
        model.eval()  # Set to eval mode to ensure deterministic behavior
        with torch.no_grad():
            output1 = model(single)
            output2 = model(single)
        
        # Compare
        max_diff = (output1 - output2).abs().max().item()
        
        # Should be identical (or very close due to floating point)
        assert max_diff < 1e-6, f"Non-deterministic output: max diff = {max_diff}"
        
        print(f"   ✅ Output is deterministic")
        print(f"      - Max difference between runs: {max_diff:.10f}")
        
    except Exception as e:
        print(f"   ❌ Deterministic test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_all_tests():
    """Run all integration tests."""
    print("\nRunning all integration tests...\n")
    
    try:
        # Test 1: Initialize model with 4 qubits
        model_4q = test_qnn_initialization(num_qubits=4, num_sub_unitaries=10)
        
        # Test 2: Forward pass with single sample
        output = test_forward_pass_single_sample(model_4q)
        
        # Test 3: Forward pass with multiple samples (processed individually)
        multi_output = test_forward_pass_multiple_samples(model_4q, num_samples=2)
        
        # Test 4: Validate output
        test_output_validity(output)
        
        # Test 5: Different qubit counts
        test_different_qubit_counts()
        
        # Test 6: Gradient flow
        test_gradient_flow(model_4q)
        
        # Test 7: Deterministic output
        test_deterministic_output(model_4q)
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nSummary:")
        print("  ✅ QNN initialization works correctly")
        print("  ✅ Forward pass produces valid outputs (single sample)")
        print("  ✅ Multiple samples can be processed individually")
        print("  ✅ No NaN or Inf values in outputs")
        print("  ✅ Works with 4 and 7 qubits")
        print("  ✅ Gradients flow correctly")
        print("  ✅ Output is deterministic")
        print("\nNote: Batch processing requires processing samples individually")
        print("Status: QNN is ready for training!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("TESTS FAILED ❌")
        print("="*70)
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
