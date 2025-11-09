# qnn_model.py
import torch
# qnn_model.py
"""Quantum models and helpers for the hybrid experiment.

Features:
- QuantumCircuit: creates a PennyLane device (Braket if BRAKET_DEVICE set, otherwise local simulator)
- ping_layer: small TorchLayer used to estimate client-side/communication overhead
- HybridDensityQNN: hybrid CNN + quantum network that estimates comm overhead and reports
  estimated device execution time (T_call - T_comm)
- get_braket_task_metadata: optional helper to fetch Braket quantum task metadata via boto3
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import pennylane as qml

# Optional AWS helper dependencies
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False


class QuantumCircuit:
    """Quantum circuit definition for the hybrid model and small ping-layer generator."""

    def __init__(self, num_qubits: int = 4, shots: Optional[int] = None):
        self.num_qubits = num_qubits
        # Allow override with env var BRAKET_SHOTS or default to None (analytic/statevector) for local
        env_shots = os.getenv('BRAKET_SHOTS')
        if env_shots is not None:
            try:
                self.shots = int(env_shots)
            except Exception:
                self.shots = shots
        else:
            self.shots = shots

        self.device = self._setup_device()

    def _setup_device(self):
        """Initialize a PennyLane device.

        If BRAKET_DEVICE is set in the environment, try to use the remote Braket device.
        Otherwise fall back to the local default.qubit simulator.
        """
        braket_arn = os.getenv('BRAKET_DEVICE', '').strip()
        if braket_arn:
            try:
                dev = qml.device(
                    "braket.aws.qubit",
                    device_arn=braket_arn,
                    wires=self.num_qubits,
                    shots=self.shots,
                )
                print(f"Using remote Braket device: {braket_arn} (wires={self.num_qubits}, shots={self.shots})")
                return dev
            except Exception as e:
                print(f"Warning: failed to initialize Braket device ({braket_arn}): {e}. Falling back to local simulator.")

        # Local simulator default. If shots is None, use analytic/statevector where available.
        if self.shots is None:
            dev = qml.device("default.qubit", wires=self.num_qubits)
        else:
            dev = qml.device("default.qubit", wires=self.num_qubits, shots=self.shots)
        print(f"Using local simulator: default.qubit (wires={self.num_qubits}, shots={self.shots})")
        return dev

    @property
    def circuit(self):
        """Return a callable QNode (quantum sub-circuit) 
        suitable for wrapping by TorchLayer."""

        @qml.qnode(self.device, interface="torch")
        def quantum_sub_circuit(inputs, weights):
            # inputs shape: (num_qubits,), weights shape: (2*num_qubits,)
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            for i in range(self.num_qubits):
                qml.RZ(weights[i], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.num_qubits):
                qml.RY(weights[i + self.num_qubits], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return quantum_sub_circuit

    def ping_layer(self):
        """Return a tiny TorchLayer used to estimate TorchLayer/device overhead.

        The ping layer uses a 1-qubit tiny circuit. We create a device similar to the
        chosen execution path (remote/local) but with a single wire so calls exercise
        the same API / communication plumbing while keeping runtime minimal.
        """
        # Use a single-qubit device for ping; if BRAKET_DEVICE is set, try remote, else local
        braket_arn = os.getenv('BRAKET_DEVICE', '').strip()
        ping_shots = 2000
        if braket_arn:
            try:
                ping_dev = qml.device("braket.aws.qubit", device_arn=braket_arn, wires=1, shots=ping_shots)
            except Exception:
                ping_dev = qml.device("default.qubit", wires=1, shots=ping_shots)
        else:
            ping_dev = qml.device("default.qubit", wires=1, shots=ping_shots)

        @qml.qnode(ping_dev, interface="torch")
        def ping_qnode(x, weights=None):
            qml.RY(x[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        # Wrap as TorchLayer (no trainable weights)
        try:
            return qml.qnn.TorchLayer(ping_qnode, {})
        except Exception:
            # In case the qnn API isn't available or fails, return a callable wrapper
            def fallback(x):
                return ping_qnode(x)

            return fallback


class HybridDensityQNN(nn.Module):
    """Hybrid CNN + Quantum Neural Network with comm-overhead estimation."""

    def __init__(self, num_sub_unitaries: int = 2, num_qubits: int = 4):
        super(HybridDensityQNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_qubits)

        self.K = num_sub_unitaries
        self.num_qubits = num_qubits

        # Build main quantum circuit and TorchLayers
        qc = QuantumCircuit(num_qubits=num_qubits)
        quantum_circuit = qc.circuit
        self.quantum_layers = nn.ModuleList([
            qml.qnn.TorchLayer(quantum_circuit, {"weights": (num_qubits * 2,)})
            for _ in range(self.K)
        ])

        self.alpha = nn.Parameter(torch.ones(self.K))
        self.fc2 = nn.Linear(num_qubits, 10)

        # Estimate client-side/communication overhead using a tiny ping circuit
        try:
            ping_qc = QuantumCircuit(num_qubits=1)
            ping_layer = ping_qc.ping_layer()
            # Warm-up and measure a few times
            dummy = torch.zeros(1)
            runs = 5
            times = []
            for _ in range(runs):
                t0 = time.time()
                # call ping_layer; some fallbacks may be plain callables
                _ = ping_layer(dummy)
                times.append(time.time() - t0)
            self._comm_overhead = float(sum(times) / len(times))
            print(f"Estimated TorchLayer comm overhead per call: {self._comm_overhead:.4f}s")
        except Exception as e:
            self._comm_overhead = 0.0
            print(f"Warning: ping overhead estimation failed: {e}; using comm_overhead=0")

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))

        quantum_outputs = []
        quantum_time_total = 0.0
        comm_est_total = 0.0

        for i in range(batch_size):
            sample = x[i]
            t0 = time.time()
            circuit_outputs = [self.quantum_layers[k](sample) for k in range(self.K)]
            call_time = time.time() - t0

            # estimate device time by subtracting comm overhead (floor at 0)
            comm_est = self._comm_overhead
            device_time_est = max(0.0, call_time - comm_est)

            quantum_time_total += call_time
            comm_est_total += comm_est

            alpha_norm = torch.softmax(self.alpha, dim=0)
            weighted_out = sum(alpha_norm[k] * circuit_outputs[k] for k in range(self.K))
            quantum_outputs.append(weighted_out)

        quantum_out = torch.stack(quantum_outputs)
        out = self.fc2(quantum_out)

        # Diagnostic summary once per forward
        try:
            avg_call = quantum_time_total / batch_size
            avg_device_est = (quantum_time_total - comm_est_total) / batch_size
        except ZeroDivisionError:
            avg_call = 0.0
            avg_device_est = 0.0

        print(
            f"Hybrid forward: batch_size={batch_size}, total_call={quantum_time_total:.3f}s, "
            f"avg_call={avg_call:.3f}s, est_device_total={(quantum_time_total - comm_est_total):.3f}s, "
            f"est_device_avg={avg_device_est:.3f}s, est_comm_avg={self._comm_overhead:.3f}s",
        )

        return out


def get_braket_task_metadata(quantum_task_arn: str, region: Optional[str] = None) -> dict:
    """Fetch Braket Quantum Task metadata using boto3.

    Args:
        quantum_task_arn: The ARN of the quantum task to query.
        region: Optional AWS region (falls back to AWS_DEFAULT_REGION env var).

    Returns:
        A dict with the raw boto3 response. If boto3 is not available or the call fails,
        returns an empty dict.
    """
    if not _BOTO3_AVAILABLE:
        print("boto3 not available in the environment. Install boto3 to query Braket metadata.")
        return {}

    region = region or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
    client = boto3.client('braket', region_name=region)

    try:
        resp = client.get_quantum_task(quantumTaskArn=quantum_task_arn)
    except ClientError as e:
        print(f"AWS ClientError while fetching quantum task metadata: {e}")
        return {}
    except BotoCoreError as e:
        print(f"AWS BotoCoreError while fetching quantum task metadata: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error when fetching quantum task metadata: {e}")
        return {}

    # Print top-level keys for visibility
    print("Braket quantum task metadata keys:", list(resp.keys()))

    # Try to compute execution durations if timestamp fields are present
    def _parse_iso(ts):
        try:
            # boto3 returns datetimes as datetime objects in many cases; handle both
            if ts is None:
                return None
            if isinstance(ts, str):
                # Attempt ISO parse
                from datetime import datetime

                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return ts
        except Exception:
            return None

    parsed = {}
    # Common fields that may exist: 'createdAt', 'startedAt', 'endedAt', 'deviceExecutionStartTime', 'deviceExecutionEndTime'
    for k in ['createdAt', 'startedAt', 'endedAt', 'deviceExecutionStartTime', 'deviceExecutionEndTime']:
        if k in resp:
            parsed[k] = _parse_iso(resp.get(k))

    # Compute durations where possible
    try:
        if 'deviceExecutionStartTime' in parsed and 'deviceExecutionEndTime' in parsed:
            parsed['device_execution_seconds'] = (
                parsed['deviceExecutionEndTime'] - parsed['deviceExecutionStartTime']
            ).total_seconds()
        elif 'startedAt' in parsed and 'endedAt' in parsed:
            parsed['total_runtime_seconds'] = (parsed['endedAt'] - parsed['startedAt']).total_seconds()
    except Exception:
        # Ignore computation errors
        pass

    # Return both raw response and parsed times for convenience
    return {'raw': resp, 'parsed_times': parsed}
