"""
Unitary Decomposition Module

This module provides utilities for decomposing unitary matrices into quantum circuits.
It supports both diagonal matrices (using Walsh decomposition) and arbitrary unitary
matrices (using PennyLane's built-in decomposition).

Mathematical Background:
- A unitary matrix U satisfies: U† @ U = I (where † denotes conjugate transpose)
- Diagonal matrices have non-zero elements only on the main diagonal
- Any unitary can be decomposed as U = V @ D @ V† where D is diagonal
"""

import numpy as np
import torch
import pennylane as qml
from typing import Union, Tuple, List, Any
from walsh_circuit_decomposition import build_optimal_walsh_circuit


def is_diagonal(matrix: Union[torch.Tensor, np.ndarray], tol: float = 1e-10) -> bool:
    """
    Check if a matrix has only diagonal elements (all off-diagonal elements are zero).
    
    A matrix M is diagonal if |M[i,j]| < tol for all i ≠ j.
    
    Mathematical Definition:
        M is diagonal ⟺ M[i,j] = 0 for all i ≠ j
    
    Args:
        matrix: Input matrix to check (can be real or complex)
        tol: Tolerance for considering off-diagonal elements as zero (default: 1e-10)
    
    Returns:
        bool: True if matrix is diagonal within tolerance, False otherwise
    
    Examples:
        >>> diag_matrix = np.diag([1, 2, 3])
        >>> is_diagonal(diag_matrix)
        True
        
        >>> non_diag = np.array([[1, 0.1], [0, 2]])
        >>> is_diagonal(non_diag, tol=1e-2)
        False
    """
    # Convert to numpy if torch tensor
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Get matrix dimensions
    n, m = matrix.shape
    
    # Matrix must be square
    if n != m:
        return False
    
    # Check all off-diagonal elements
    for i in range(n):
        for j in range(m):
            if i != j:
                # Check if off-diagonal element exceeds tolerance
                if np.abs(matrix[i, j]) > tol:
                    return False
    
    return True


def is_unitary(matrix: Union[torch.Tensor, np.ndarray], tol: float = 1e-6) -> bool:
    """
    Verify that a matrix is unitary by checking if U† @ U = I.
    
    A matrix U is unitary if it satisfies:
        U† @ U = I
    where U† is the conjugate transpose and I is the identity matrix.
    
    Mathematical Definition:
        U is unitary ⟺ U† @ U = I ⟺ ||U† @ U - I||_F < tol
    where ||·||_F is the Frobenius norm.
    
    Args:
        matrix: Input matrix to verify (must be square, can be complex)
        tol: Tolerance for unitarity check (default: 1e-6)
    
    Returns:
        bool: True if matrix is unitary within tolerance, False otherwise
    
    Examples:
        >>> # Identity is unitary
        >>> I = np.eye(4)
        >>> is_unitary(I)
        True
        
        >>> # Hadamard gate is unitary
        >>> H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        >>> is_unitary(H)
        True
        
        >>> # Random matrix is typically not unitary
        >>> random_matrix = np.random.rand(4, 4)
        >>> is_unitary(random_matrix)
        False
    """
    # Convert to numpy if torch tensor
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Matrix must be square
    n, m = matrix.shape
    if n != m:
        return False
    
    # Compute U† @ U
    u_dagger = np.conj(matrix.T)
    product = u_dagger @ matrix
    
    # Compare with identity matrix
    identity = np.eye(n, dtype=matrix.dtype)
    
    # Compute Frobenius norm of difference: ||U† @ U - I||_F
    diff = product - identity
    frobenius_norm = np.linalg.norm(diff, ord='fro')
    
    return frobenius_norm < tol


def get_num_qubits(matrix: Union[torch.Tensor, np.ndarray]) -> int:
    """
    Extract the number of qubits from a unitary matrix's dimensions.
    
    For a quantum system with n qubits, the unitary matrix has dimensions 2^n × 2^n.
    This function computes n = log₂(dim).
    
    Mathematical Relationship:
        dim = 2^n ⟹ n = log₂(dim)
    
    Args:
        matrix: Unitary matrix with dimensions 2^n × 2^n
    
    Returns:
        int: Number of qubits n
    
    Raises:
        ValueError: If matrix is not square or dimensions are not a power of 2
    
    Examples:
        >>> # 2-qubit system: 4×4 matrix
        >>> matrix_2q = np.eye(4)
        >>> get_num_qubits(matrix_2q)
        2
        
        >>> # 4-qubit system: 16×16 matrix
        >>> matrix_4q = np.eye(16)
        >>> get_num_qubits(matrix_4q)
        4
        
        >>> # Invalid: 5×5 is not 2^n
        >>> invalid = np.eye(5)
        >>> get_num_qubits(invalid)
        ValueError: Matrix dimensions must be a power of 2
    """
    # Convert to numpy if torch tensor
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Get matrix dimensions
    n, m = matrix.shape
    
    # Matrix must be square
    if n != m:
        raise ValueError(
            f"Matrix must be square. Got dimensions {n}×{m}"
        )
    
    # Check if dimension is a power of 2
    dim = n
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(
            f"Matrix dimensions must be a power of 2 (2^n). Got {dim}×{dim}"
        )
    
    # Compute number of qubits: n = log₂(dim)
    num_qubits = int(np.log2(dim))
    
    return num_qubits



def decompose_unitary_matrix(
    matrix: Union[torch.Tensor, np.ndarray],
    method: str = "auto"
) -> List[Tuple[str, Any]]:
    """
    Decompose a unitary matrix into quantum gates.
    
    This function provides a unified interface for decomposing both diagonal and
    non-diagonal unitary matrices into quantum circuits. It automatically selects
    the appropriate decomposition method based on the matrix structure:
    - Diagonal matrices: Walsh decomposition (optimal for diagonal unitaries)
    - Non-diagonal matrices: PennyLane QubitUnitary (works for arbitrary unitaries)
    
    Mathematical Background:
        For diagonal matrices, Walsh decomposition provides an optimal circuit
        using only CNOT and RZ gates with O(n²) gate count.
        
        For non-diagonal matrices, PennyLane's QubitUnitary gate handles the
        decomposition automatically and maintains gradient flow for backpropagation.
    
    Args:
        matrix: Unitary matrix to decompose (2^n × 2^n)
        method: Decomposition method to use
            - "auto": Automatically choose based on matrix structure (default)
            - "walsh": Force Walsh decomposition (only works for diagonal matrices)
            - "pennylane": Force PennyLane QubitUnitary decomposition
    
    Returns:
        List[Tuple[str, Any]]: List of (gate_type, params) tuples:
            - ("CNOT", [control, target]): CNOT gate
            - ("RZ", [angle, qubit]): RZ rotation gate
            - ("QubitUnitary", [matrix, wires]): Arbitrary unitary gate
    
    Raises:
        ValueError: If matrix is not unitary
        ValueError: If matrix dimensions are not a power of 2
        ValueError: If method="walsh" but matrix is not diagonal
        ValueError: If method is not one of "auto", "walsh", "pennylane"
    
    Examples:
        >>> # Diagonal matrix - uses Walsh decomposition
        >>> diag_matrix = np.diag([1, -1, 1j, -1j])
        >>> gates = decompose_unitary_matrix(diag_matrix)
        >>> # Returns list of CNOT and RZ gates
        
        >>> # Non-diagonal matrix - uses PennyLane
        >>> hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        >>> gates = decompose_unitary_matrix(hadamard)
        >>> # Returns [("QubitUnitary", [hadamard, [0]])]
        
        >>> # Force specific method
        >>> gates = decompose_unitary_matrix(diag_matrix, method="walsh")
    """
    # Validate method parameter
    valid_methods = ["auto", "walsh", "pennylane"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {valid_methods}"
        )
    
    # Convert to numpy if torch tensor
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = np.array(matrix)
    
    # Validate matrix is square
    n, m = matrix_np.shape
    if n != m:
        raise ValueError(
            f"Matrix must be square. Got dimensions {n}×{m}"
        )
    
    # Validate matrix dimensions are power of 2
    try:
        num_qubits = get_num_qubits(matrix_np)
    except ValueError as e:
        raise ValueError(f"Invalid matrix dimensions: {e}")
    
    # Validate matrix is unitary
    if not is_unitary(matrix_np):
        raise ValueError(
            "Matrix is not unitary. A unitary matrix must satisfy U† @ U = I"
        )
    
    # Determine if matrix is diagonal
    matrix_is_diagonal = is_diagonal(matrix_np)
    
    # Select decomposition method
    if method == "auto":
        # Automatically choose based on matrix structure
        selected_method = "walsh" if matrix_is_diagonal else "pennylane"
    elif method == "walsh":
        # User forced Walsh - validate it's diagonal
        if not matrix_is_diagonal:
            raise ValueError(
                "Walsh decomposition requires a diagonal matrix. "
                "Use method='auto' or method='pennylane' for non-diagonal matrices."
            )
        selected_method = "walsh"
    else:  # method == "pennylane"
        selected_method = "pennylane"
    
    # Perform decomposition based on selected method
    if selected_method == "walsh":
        # Use Walsh decomposition for diagonal matrices
        gates = build_optimal_walsh_circuit(matrix_np)
        return gates
    
    else:  # selected_method == "pennylane"
        # Use PennyLane QubitUnitary for non-diagonal matrices
        # PennyLane will handle the decomposition automatically
        wires = list(range(num_qubits))
        
        # Return as QubitUnitary gate
        # The matrix will be applied directly in the quantum circuit
        gates = [("QubitUnitary", [matrix_np, wires])]
        return gates


def apply_decomposed_circuit(
    gates: List[Tuple[str, Any]],
    inverse: bool = False
) -> None:
    """
    Apply a decomposed quantum circuit to the current PennyLane quantum context.
    
    This function takes a list of gates (as returned by decompose_unitary_matrix)
    and applies them to the quantum circuit. It supports inverse operations for
    implementing U† (conjugate transpose) operations.
    
    Mathematical Background:
        For inverse operations:
        - CNOT is self-inverse: CNOT† = CNOT
        - RZ(θ)† = RZ(-θ): Negate the rotation angle
        - QubitUnitary(U)† = QubitUnitary(U†): Conjugate transpose the matrix
    
    Gate Handling:
        - CNOT gates: Applied as qml.CNOT(wires=[control, target])
          Self-inverse, so same operation for forward and inverse
        
        - RZ gates: Applied as qml.RZ(angle, wires=qubit)
          For inverse: negate the angle to get RZ(-angle)
        
        - QubitUnitary gates: Applied as qml.QubitUnitary(matrix, wires=wires)
          For inverse: conjugate transpose the matrix (U†)
    
    Args:
        gates: List of (gate_type, params) tuples from decompose_unitary_matrix
            - ("CNOT", [control, target])
            - ("RZ", [angle, qubit])
            - ("QubitUnitary", [matrix, wires])
        inverse: If True, apply the inverse (conjugate transpose) of the circuit
            Default: False (apply circuit as-is)
    
    Returns:
        None: Gates are applied to the current PennyLane quantum context
    
    Raises:
        ValueError: If gate type is not recognized
        RuntimeError: If called outside a PennyLane quantum context
    
    Examples:
        >>> # Inside a PennyLane QNode
        >>> @qml.qnode(dev)
        >>> def circuit():
        >>>     # Apply forward circuit
        >>>     gates = decompose_unitary_matrix(U)
        >>>     apply_decomposed_circuit(gates, inverse=False)
        >>>     
        >>>     # Apply inverse circuit (U†)
        >>>     apply_decomposed_circuit(gates, inverse=True)
        >>>     
        >>>     return qml.state()
        
        >>> # For U = VDV† decomposition:
        >>> @qml.qnode(dev)
        >>> def circuit():
        >>>     v_gates = decompose_unitary_matrix(V)
        >>>     d_gates = decompose_unitary_matrix(D)
        >>>     
        >>>     apply_decomposed_circuit(v_gates)      # Apply V
        >>>     apply_decomposed_circuit(d_gates)      # Apply D
        >>>     apply_decomposed_circuit(v_gates, inverse=True)  # Apply V†
        >>>     
        >>>     return qml.expval(qml.PauliZ(0))
    
    Notes:
        - This function must be called within a PennyLane QNode context
        - Gates are applied in the order they appear in the list
        - For inverse=True, gates are still applied in the same order
          (the caller should reverse the order if needed for full circuit inverse)
        - PennyLane automatically handles gradient flow through these operations
    """
    # Process gates in order
    for gate_type, params in gates:
        
        if gate_type == "CNOT":
            # CNOT gate: self-inverse (CNOT† = CNOT)
            # params = [control, target]
            control, target = params
            qml.CNOT(wires=[control, target])
        
        elif gate_type == "RZ":
            # RZ rotation gate: RZ(θ)† = RZ(-θ)
            # params = [angle, qubit]
            angle, qubit = params
            
            if inverse:
                # For inverse, negate the angle
                qml.RZ(-angle, wires=qubit)
            else:
                # Forward: apply as-is
                qml.RZ(angle, wires=qubit)
        
        elif gate_type == "QubitUnitary":
            # Arbitrary unitary gate: U† = conjugate transpose
            # params = [matrix, wires]
            matrix, wires = params
            
            if inverse:
                # For inverse, apply conjugate transpose
                # U† = (U*)^T where * is complex conjugate and T is transpose
                matrix_dagger = np.conj(matrix.T)
                qml.QubitUnitary(matrix_dagger, wires=wires)
            else:
                # Forward: apply as-is
                qml.QubitUnitary(matrix, wires=wires)
        
        else:
            # Unknown gate type
            raise ValueError(
                f"Unknown gate type '{gate_type}'. "
                f"Supported types: CNOT, RZ, QubitUnitary"
            )



def reconstruct_unitary_from_gates(
    gates: List[Tuple[str, Any]],
    matrix_size: int
) -> np.ndarray:
    """
    Reconstruct a unitary matrix from its gate decomposition.
    
    This function is primarily used for testing and validation. It takes a list
    of gates and reconstructs the full unitary matrix they represent by computing
    the matrix product of all gate operations.
    
    Mathematical Background:
        For gates G₁, G₂, ..., Gₙ applied in sequence, the total unitary is:
        U = Gₙ @ ... @ G₂ @ G₁
        
        Each gate is converted to its matrix representation:
        - CNOT: 4×4 controlled-NOT matrix (for 2-qubit subsystem)
        - RZ(θ): diag(1, e^(iθ)) rotation matrix
        - QubitUnitary: The matrix itself
    
    Args:
        gates: List of (gate_type, params) tuples from decompose_unitary_matrix
        matrix_size: Size of the full unitary matrix (2^n for n qubits)
    
    Returns:
        np.ndarray: Reconstructed unitary matrix of shape (matrix_size, matrix_size)
    
    Examples:
        >>> # Decompose and reconstruct
        >>> original = np.diag([1, -1, 1j, -1j])
        >>> gates = decompose_unitary_matrix(original)
        >>> reconstructed = reconstruct_unitary_from_gates(gates, 4)
        >>> error = np.linalg.norm(original - reconstructed)
        >>> print(f"Reconstruction error: {error:.2e}")
        Reconstruction error: 1.23e-15
    """
    # Start with identity matrix
    num_qubits = int(np.log2(matrix_size))
    result = np.eye(matrix_size, dtype=complex)
    
    # Apply each gate
    for gate_type, params in gates:
        
        if gate_type == "CNOT":
            # Build CNOT matrix for the full system
            control, target = params
            gate_matrix = _build_cnot_matrix(num_qubits, control, target)
            result = gate_matrix @ result
        
        elif gate_type == "RZ":
            # Build RZ matrix for the full system
            angle, qubit = params
            gate_matrix = _build_rz_matrix(num_qubits, angle, qubit)
            result = gate_matrix @ result
        
        elif gate_type == "QubitUnitary":
            # The matrix is already provided
            matrix, wires = params
            
            # If it's a full-system unitary, apply directly
            if len(wires) == num_qubits:
                result = matrix @ result
            else:
                # For subsystem unitaries, embed in full system
                gate_matrix = _embed_subsystem_unitary(matrix, wires, num_qubits)
                result = gate_matrix @ result
    
    return result


def _build_cnot_matrix(num_qubits: int, control: int, target: int) -> np.ndarray:
    """Build CNOT gate matrix for full quantum system."""
    size = 2 ** num_qubits
    cnot = np.zeros((size, size), dtype=complex)
    
    # CNOT flips target qubit when control is |1⟩
    for i in range(size):
        # Check if control qubit is 1
        if (i >> control) & 1:
            # Flip target qubit
            j = i ^ (1 << target)
            cnot[i, j] = 1
        else:
            # Control is 0, identity operation
            cnot[i, i] = 1
    
    return cnot


def _build_rz_matrix(num_qubits: int, angle: float, qubit: int) -> np.ndarray:
    """Build RZ rotation gate matrix for full quantum system."""
    size = 2 ** num_qubits
    rz = np.eye(size, dtype=complex)
    
    # RZ(θ) applies phase e^(iθ) when qubit is |1⟩
    for i in range(size):
        if (i >> qubit) & 1:
            rz[i, i] = np.exp(1j * angle)
    
    return rz


def _embed_subsystem_unitary(
    matrix: np.ndarray,
    wires: List[int],
    num_qubits: int
) -> np.ndarray:
    """Embed a subsystem unitary into the full quantum system."""
    # For simplicity, if the subsystem is the full system, return as-is
    # Otherwise, this would require tensor product embedding
    size = 2 ** num_qubits
    if matrix.shape[0] == size:
        return matrix
    
    # For partial system, we'd need to implement tensor product embedding
    # This is complex, so for now we assume full-system unitaries
    raise NotImplementedError(
        "Subsystem unitary embedding not yet implemented. "
        "Only full-system unitaries are currently supported."
    )
