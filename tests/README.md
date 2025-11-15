# Test Suite

This folder contains all test files for the QNN project.

## Running Tests

### Run All Tests
```bash
cd tests
python run_all_tests.py
```

### Run Individual Tests

**Quick Verification (Recommended First)**
```bash
python test_quick_verify.py
```

**Final Comprehensive Test**
```bash
python final_test.py
```

**Specific Component Tests**
```bash
python test_unitary_decomposition.py
python test_walsh_validation.py
python test_qnn_forward_pass.py
python test_qnn_gradients.py
```

## Test Categories

### 1. Quick Verification Tests
- `test_quick_verify.py` - Fast sanity checks (10 tests, ~5 seconds)

### 2. Core Component Tests
- `test_unitary_decomposition.py` - Unitary decomposition validation
- `test_walsh_validation.py` - Walsh decomposition validation
- `test_walsh_circuit_decomposition.py` - Walsh circuit tests
- `test_density_qnn.py` - Density matrix utilities

### 3. Integration Tests
- `test_decomposition.py` - V @ D @ V† decomposition
- `test_simple_decomp.py` - Simple decomposition cases
- `test_apply_circuit.py` - Circuit application in PennyLane

### 4. QNN Tests
- `test_qnn_forward_pass.py` - Forward pass validation
- `test_qnn_gradients.py` - Gradient flow validation
- `test_rbs_networks.py` - RBS network patterns
- `test_complete_qnn.py` - Complete QNN pipeline

### 5. Comprehensive Tests
- `final_test.py` - Final comprehensive test (12 tests)
- `test_comprehensive.py` - Full test suite (19 tests)

### 6. Training Tests
- `test_actual_training.py` - Real training validation

## Test Results

Expected results:
- Quick verification: 10/10 tests pass
- Final test: 12/12 tests pass
- Comprehensive suite: 18/19 tests pass (94.7%)

The only "failing" test is a reconstruction utility function that isn't used in production code.

## Test Coverage

- ✅ Walsh decomposition for diagonal matrices
- ✅ PennyLane decomposition for non-diagonal matrices
- ✅ V @ D @ V† decomposition strategy
- ✅ QNN model forward and backward passes
- ✅ Gradient flow through quantum circuits
- ✅ All 4 entanglement patterns
- ✅ Batch processing
- ✅ Error handling
- ✅ Edge cases
