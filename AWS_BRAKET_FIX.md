# AWS Braket S3 Error Fix

## Problem

The code was throwing this error:
```
botocore.errorfactory.NoSuchKey: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
```

## Root Cause

In `qnn_model.py`, the `_setup_device()` method had:
1. **Hardcoded AWS environment variables** that forced AWS Braket usage:
   ```python
   os.environ['AWS_DEFAULT_REGION'] = "us-west-1"
   os.environ['BRAKET_DEVICE'] = 'arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3'
   ```

2. **Commented out local simulator fallback code**, preventing the system from using the local PennyLane simulator when AWS credentials weren't available.

## Solution

### Changes Made to `qnn_model.py`:

1. **Removed hardcoded AWS environment variables**
   - No longer forces AWS Braket device usage
   - Respects user's environment configuration

2. **Uncommented local simulator fallback**
   ```python
   # Local simulator default. If shots is None, use analytic/statevector where available.
   if self.shots is None:
       dev = qml.device("default.qubit", wires=self.num_qubits)
   else:
       dev = qml.device("default.qubit", wires=self.num_qubits, shots=self.shots)
   ```

3. **Proper fallback behavior**
   - If `BRAKET_DEVICE` environment variable is set → tries AWS Braket
   - If AWS Braket fails or not configured → falls back to local simulator
   - No errors when AWS credentials are missing

## How to Use

### Option 1: Local Simulator (Default)
Just run your code normally without setting any environment variables:
```bash
python your_script.py
```

The system will automatically use PennyLane's `default.qubit` local simulator.

### Option 2: AWS Braket (Optional)
If you want to use AWS Braket quantum hardware/simulators:

1. Set up your AWS credentials in `.env` file:
   ```bash
   AWS_DEFAULT_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_key
   BRAKET_DEVICE=arn:aws:braket:::device/quantum-simulator/amazon/sv1
   BRAKET_SHOTS=1000
   ```

2. Make sure you have proper S3 bucket permissions configured in AWS

3. Run your code - it will automatically use the Braket device

## Testing

Added `test_device_setup.py` to verify the fix:
```bash
python QNN-hack/test_device_setup.py
```

This test confirms:
- ✅ Local simulator initializes without AWS credentials
- ✅ Quantum circuit can be created
- ✅ No S3/botocore errors

## Verification

All existing tests still pass:
```bash
python -m pytest QNN-hack/tests/ -v
# Result: 13/13 tests passing
```

## Benefits

1. **Works out of the box** - No AWS setup required for local development
2. **Graceful fallback** - Automatically uses local simulator if AWS fails
3. **Flexible** - Easy to switch between local and cloud quantum devices
4. **No breaking changes** - All existing functionality preserved

## Related Files

- `qnn_model.py` - Main fix applied here
- `test_device_setup.py` - New test to verify local simulator works
- `example.env.txt` - Template for AWS configuration (optional)
