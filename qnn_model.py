# qnn_model.py
import torch
"""Quantum models and helpers for the hybrid experiment.

Features:
- QuantumCircuit: creates a PennyLane device (Braket if BRAKET_DEVICE set, otherwise local simulator)
- HybridDensityQNN: hybrid CNN + quantum network with density matrix approach
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
from walsh_circuit_decomposition import build_optimal_walsh_circuit, diagonalize_unitary

# Optional AWS helper dependencies
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False


class QuantumCircuit:
    """Quantum circuit definition for the hybrid model."""

    def __init__(self, num_qubits: int = 7, shots: Optional[int] = None, QNN_layers=10):
        print(f"Init Quantum circuit, QNN_layers: {QNN_layers}, qubits: {num_qubits}")
        self.num_qubits = num_qubits
        self.QNN_layers = QNN_layers
        
        # Allow override with env var BRAKET_SHOTS
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

        # Local simulator default
        if self.shots is None:
            dev = qml.device("default.qubit", wires=self.num_qubits)
        else:
            dev = qml.device("default.qubit", wires=self.num_qubits, shots=self.shots)
        print(f"Using local simulator: default.qubit (wires={self.num_qubits}, shots={self.shots})")
        return dev

    @property
    def circuit(self):
        """Return a callable QNode (quantum sub-circuit) suitable for wrapping by TorchLayer."""

        # Prepare the density quantum layer with all 4 paper patterns (Figure 9)
        # This uses: pyramid, X-circuit, butterfly, and round-robin
        quantum_layer = density_layer(
            self.num_qubits, 
            self.QNN_layers,
            patterns=None  # Uses all 4 paper patterns by default
        )
        
        @qml.qnode(self.device, interface="torch")
        def quantum_sub_circuit(inputs, weights):
            """
            Density matrix quantum circuit with proper diagonalization.
            
            This implements the full density QNN approach:
            1. Encode classical data into quantum state
            2. Create density matrix from weighted RBS networks
            3. Diagonalize the density matrix
            4. Apply using Walsh decomposition
            
            Args:
                inputs: shape (num_qubits,) - feature vector from CNN
                weights: shape (QNN_layers,) - mixing coefficients for density matrices
            
            Returns:
                List of expectation values, one per qubit
            """
            # Step 1: Data encoding - embed CNN features into quantum states
            # Using amplitude encoding with RY rotations
            for i in range(self.num_qubits):
                qml.RY(inputs[i] * pi/2, wires=i)
            
            # Step 2: Create density matrix from weighted sum of RBS networks
            density_matrix = quantum_layer(weights)
            
            # Step 3: Diagonalize the density matrix
            # This gives us: density_matrix = U @ D @ U†
            # where D is diagonal and U is the transformation
            diag_matrix, transform_matrix = diagonalize_unitary(density_matrix)
            
            # Step 4: Apply the unitary transformation
            # We need to apply: U @ D @ U†
            
            # 4a: Apply U (transformation to diagonal basis)
            transform_circuit = build_optimal_walsh_circuit(torch.tensor(transform_matrix, dtype=torch.complex64))
            for gate in transform_circuit:
                if gate[0] == "CNOT":
                    qml.CNOT(wires=gate[1])
                elif gate[0] == "RZ":
                    qml.RZ(gate[1][0], wires=gate[1][1])
            
            # 4b: Apply D (diagonal unitary using Walsh decomposition)
            diag_circuit = build_optimal_walsh_circuit(diag_matrix)
            for gate in diag_circuit:
                if gate[0] == "CNOT":
                    qml.CNOT(wires=gate[1])
                elif gate[0] == "RZ":
                    qml.RZ(gate[1][0], wires=gate[1][1])
            
            # 4c: Apply U† (inverse transformation)
            for gate in reversed(transform_circuit):
                if gate[0] == "CNOT":
                    qml.CNOT(wires=gate[1])  # CNOT is self-inverse
                elif gate[0] == "RZ":
                    qml.RZ(-gate[1][0], wires=gate[1][1])  # Inverse rotation
            
            # Step 5: Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return quantum_sub_circuit


class HybridDensityQNN(nn.Module):
    """Hybrid CNN + Quantum Neural Network using density matrix approach."""

    def __init__(self, num_sub_unitaries: int = 10, num_qubits: int = 7):
        super(HybridDensityQNN, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_qubits)

        self.num_qubits = num_qubits
        self.num_sub_unitaries = num_sub_unitaries

        # Build quantum circuit
        qc = QuantumCircuit(num_qubits=num_qubits, QNN_layers=num_sub_unitaries, shots=None)
        self.quantum_circuit = qc.circuit
        
        # Create trainable weights for quantum layer
        self.quantum_weights = torch.nn.Parameter(torch.randn(num_sub_unitaries) * 0.1)

        # Final classifier
        self.fc2 = nn.Linear(num_qubits, 10)

    def forward(self, x):
        """
        Forward pass through hybrid CNN-QNN.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        batch_size = x.shape[0]
        
        # CNN feature extraction
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))  # Shape: (batch_size, num_qubits)
        
        # Quantum layer: process each sample individually
        quantum_outputs = []
        for i in range(batch_size):
            sample = x[i]  # Shape: (num_qubits,)
            qout = self.quantum_circuit(sample, self.quantum_weights)
            quantum_outputs.append(torch.stack(qout).float())  # Ensure float32
        
        x = torch.stack(quantum_outputs)  # Shape: (batch_size, num_qubits)
        
        # Final classification
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
            if ts is None:
                return None
            if isinstance(ts, str):
                from datetime import datetime
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return ts
        except Exception:
            return None

    parsed = {}
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
        pass

    return {'raw': resp, 'parsed_times': parsed}
