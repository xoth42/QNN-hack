# Code from morik04
# https://github.com/morik04/quantum-non-markovian-dynamics/blob/main/optimal_walsh_update.ipynb

import numpy as np
# from qiskit import QuantumCircuit
# from qiskit.circuit.library import RZGate
from collections import defaultdict
import pennylane as qml
import torch


# helper functions
def gray_code(n: int):
    return [i ^ (i >> 1) for i in range(1 << n)]

def _fwht(vec: np.ndarray) -> np.ndarray:
    
    y = vec.astype(float).copy()
    h, N = 1, y.size
    while h < N:
        for i in range(0, N, h*2):
            for j in range(i, i+h):
                u, v = y[j], y[j+h]
                y[j], y[j+h] = u+v, u-v
        h <<= 1
    return y

def Walsh_coefficients(matrix: np.ndarray) -> np.ndarray:
    # fix for err (RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.)
    try:
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().numpy()
    except Exception:
        pass
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    N = matrix.shape[0]
    assert N & (N-1) == 0 and N > 0
    
    d = np.diag(matrix)
    # assert np.allclose(matrix, np.diag(d))
    # assert np.allclose(np.abs(d), 1.0, atol=1e-12)

    f = np.angle(d)   
    a = _fwht(f) / N
    return a


def diagonalize_unitary(matrix):
    """
    Diagonalize a unitary matrix.
    
    Args:
        matrix: Unitary matrix to diagonalize (can be torch.Tensor or numpy array)
        
    Returns:
        Tuple of (diagonal_matrix, transformation_matrix)
        where: matrix = transformation @ diagonal_matrix @ transformation.conj().T
    """
    # Convert to numpy if needed
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().numpy()
    elif not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Check if already diagonal
    d = np.diag(matrix)
    if np.allclose(matrix, np.diag(d), atol=1e-10):
        return matrix, np.eye(matrix.shape[0])
    
    # Diagonalize using eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    diagonal_matrix = np.diag(eigenvalues)
    
    return diagonal_matrix, eigenvectors


def _is_diagonal(matrix, tol=1e-10):
    """
    Check if a matrix has only diagonal elements (all off-diagonal elements are zero).
    
    Args:
        matrix: Input matrix to check (can be torch.Tensor or numpy array)
        tol: Tolerance for considering off-diagonal elements as zero
    
    Returns:
        bool: True if matrix is diagonal within tolerance
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
                if np.abs(matrix[i, j]) > tol:
                    return False
    
    return True


EPS = 1e-12        # ignore small rotations
# replace qiskit with pennylane backend
def build_optimal_walsh_circuit(matrix):
    """
    Build Walsh circuit for DIAGONAL unitary matrices only.
    
    The Walsh decomposition algorithm is specifically designed for diagonal unitary
    matrices and will produce incorrect results for non-diagonal matrices. This
    function uses the Fast Walsh-Hadamard Transform (FWHT) to compute optimal
    decomposition into CNOT and RZ gates.
    
    Mathematical Background:
        For a diagonal unitary matrix D = diag(e^(iφ₀), e^(iφ₁), ..., e^(iφₙ)),
        Walsh decomposition finds coefficients aⱼ such that:
            φₖ = Σⱼ aⱼ · wⱼ(k)
        where wⱼ(k) are Walsh functions. The circuit implements these rotations
        using Gray code ordering to minimize CNOT gates.
    
    Args:
        matrix: DIAGONAL unitary matrix (2^n × 2^n) where all off-diagonal
                elements are zero. Can be torch.Tensor or numpy array.
    
    Returns:
        List of (gate_type, params) tuples:
            - ("CNOT", [control, target]): CNOT gate
            - ("RZ", [angle, qubit]): RZ rotation gate
    
    Raises:
        ValueError: If matrix is not diagonal. Walsh decomposition only works
                   for diagonal matrices. Use decompose_unitary_matrix() from
                   unitary_decomposition.py for non-diagonal matrices.
    
    Examples:
        >>> # Diagonal matrix - works correctly
        >>> diag_matrix = np.diag([1, -1, 1j, -1j])
        >>> gates = build_optimal_walsh_circuit(diag_matrix)
        
        >>> # Non-diagonal matrix - raises ValueError
        >>> hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        >>> gates = build_optimal_walsh_circuit(hadamard)
        ValueError: Walsh decomposition requires diagonal matrix...
    
    References:
        - Original implementation: https://github.com/morik04/quantum-non-markovian-dynamics
        - Walsh functions and quantum circuits: arXiv:quant-ph/0406176
    """
    # VALIDATION: Check if matrix is diagonal
    if not _is_diagonal(matrix):
        raise ValueError(
            "Walsh decomposition requires a diagonal matrix. "
            "The input matrix has non-zero off-diagonal elements. "
            "For non-diagonal unitary matrices, use decompose_unitary_matrix() "
            "from unitary_decomposition.py with method='pennylane' or method='auto'."
        )
    aj = np.real(Walsh_coefficients(matrix)) # get aj's for the 2**n diagonal
    n = int(np.log2(len(aj)))
    
    # qc = QuantumCircuit(n)
    qc = []
    gray_seq = gray_code(n)
    groups = defaultdict(list)
    for j in gray_seq:
        if j == 0:
            continue  # skip a0
        target = j.bit_length() - 1
        groups[target].append(j)

    def controls_of(j, target):
        return [q for q in range(n) if ((j >> q) & 1) and q != target]

    for target in sorted(groups):
        seq = groups[target]
        if not seq:
            continue

    
        j0 = seq[0]
        theta0 = -2.0 * aj[j0]
        current = set(controls_of(j0, target))  
      
        for c in current:
            qc.append(("CNOT",[c, target]))
            #qml.CNOT(wires=[c, target])
            # qc.cx(c, target)
        if abs(theta0) >= EPS:
            qc.append(("RZ",[theta0, target]))
            # qml.RZ(theta0, wires=target)
            # qc.append(RZGate(theta0), [target])

        prev = j0

        
        for j in seq[1:]:
            theta = -2.0 * aj[j]
            diff = prev ^ j                    
            flip = diff.bit_length() - 1        
            if flip != target:
                qc.append(("CNOT",[flip, target]))
                # qml.CNOT(wires=[flip, target])
                # qc.cx(flip, target)
               
                if flip in current:
                    current.remove(flip)
                else:
                    current.add(flip)

            if abs(theta) >= EPS:
                qc.append(("RZ",[theta, target]))
                # qml.RZ(theta, wires=target)
                # qc.append(RZGate(theta), [target])

            prev = j

        for c in sorted(current, reverse=True):
            qc.append(("CNOT",[c, target]))
            # qml.CNOT(wires=[c, target])
            # qc.cx(c, target)

    return qc


# example test 
# phases_2q = np.array([0, np.pi/4, np.pi/2, np.pi])
# diagU_2q = np.diag(np.exp(1j * phases_2q))
# qc=build_optimal_walsh_circuit(diagU_2q)
# qc.draw('mpl')