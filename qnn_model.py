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
from numpy import pi
from density_qnn import density_layer
from walsh_circuit_decomposition import build_optimal_walsh_circuit
# Optional AWS helper dependencies
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False

class QuantumCircuit:
    """Quantum circuit definition for the hybrid model and small ping-layer generator."""

    def __init__(self, num_qubits: int = 7, shots: Optional[int] = 10,QNN_layers=10):
        print(f"Init Quantum circuit, QNN_layers: {QNN_layers}, qubits: {num_qubits}")
        self.num_qubits = num_qubits
        self.QNN_layers = QNN_layers
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
                print(f"Successful! Using remote Braket device: {braket_arn} (wires={self.num_qubits}, shots={self.shots})")
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

        # prepare the quantum layer
        quantum_layer = density_layer(self.num_qubits,self.QNN_layers)
        
        @qml.qnode(self.device, interface="torch")
        def quantum_sub_circuit(inputs, weights):
            # inputs shape: (num_qubits,), 
            # old version: weights shape: (2*num_qubits,)
            # new version: weights shape: (self.QNN_layers,) - one weight per sub-unitary
            
            # Embed the incoming values from the CNN into quantum states
            # set to Y(pi*xi/2) from literature
            for i in range(self.num_qubits):
                qml.RY(inputs[i] * pi/2, wires=i)
                
            # Create the internal quantum layer from the weights
            quantum_layer_decomp = quantum_layer(weights)
            # Decompose to build optimal circuit to run 
            quantum_layer_decomp = build_optimal_walsh_circuit(quantum_layer_decomp)
            # interpret circuit output
            for gate in quantum_layer_decomp:
                if gate[0] == "CNOT":
                    qml.CNOT(wires=gate[1])
                elif gate[0] == "RZ":
                    qml.RZ(gate[1][0],wires=gate[1][1]) # first is theta0, second is target
                else:
                    assert "Layer decomposition error"
            # Measure outputs
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return quantum_sub_circuit


class HybridDensityQNN(nn.Module):
    """Hybrid CNN + Quantum Neural Network with comm-overhead estimation."""

    def __init__(self, num_sub_unitaries: int = 10, num_qubits: int = 7):
        super(HybridDensityQNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_qubits)

        self.num_qubits = num_qubits

        # Build main quantum circuit and TorchLayers
        qc = QuantumCircuit(num_qubits=num_qubits,QNN_layers=num_sub_unitaries)
        quantum_circuit = qc.circuit
        # start with 1 qnn-subunitarys layer (with num_sub_unitaries for amount of sub unitaries)
        # https://docs.pennylane.ai/en/stable/code/api/pennylane.qnn.TorchLayer.html
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {"weights": num_sub_unitaries})
           

        # self.quantum_layers = nn.ModuleList([
        #     qml.qnn.TorchLayer(quantum_circuit, {"weights": (num_qubits * 2,)})
        #     for _ in range(self.K)
        # ])

        # self.alpha = nn.Parameter(torch.ones(self.K))
        # self.alpha = nn.Parameter(torch.ones(1)) # trying to stich the old version together

        self.fc2 = nn.Linear(num_qubits, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = self.qlayer(x) # run the quantum layer
        x = self.fc2(x)
        return x


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
