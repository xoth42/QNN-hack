from tuple_triangle import inverted_pyramid, pyramid

import torch
from numpy import pi,cos,sin
from random import random
from torch import kron
from functools import reduce
twopi: float = 2 * pi # my computational physics teacher told me do this

I = torch.eye(2, dtype=torch.float32)
# "hamming weight preserving unitary"
# "Reconfigurable beam splitter"
def RBS(theta):
    """
    Reconfigurable Beam Splitter gate.
    
    Args:
        theta: Rotation angle
        
    Returns:
        4x4 RBS unitary matrix
    """
    c = float(cos(theta))
    s = float(sin(theta))
    return torch.tensor([
        [1.0,  0.0,  0.0, 0.0],
        [0.0,    c,   -s, 0.0],
        [0.0,    s,    c, 0.0],
        [0.0,  0.0,  0.0, 1.0]
    ], dtype=torch.float32)

# random theta from 0 to 2pi
def get_theta():
    return random() * twopi


# RBS with random theta
def RandRBS():
    return RBS(get_theta())
 
 
def matrix_from_IRBS_string(string):
    """
    Convert a string of 'I' and 'RBS' tokens to a matrix.
    
    String format: 'I' for identity, 'RBS' for random beam splitter
    Example: 'IRBS' = I ⊗ RBS (2 qubits)
             'IRBSIRBS' = I ⊗ RBS ⊗ I ⊗ RBS (4 qubits)
    
    Args:
        string: String containing 'I' and 'RBS' tokens
        
    Returns:
        Tensor product of gates
    """
    if not string:
        raise ValueError("String cannot be empty")
    
    # Parse string into tokens
    tokens = []
    i = 0
    while i < len(string):
        if string[i:i+3] == 'RBS':
            tokens.append('RBS')
            i += 3
        elif string[i] == 'I':
            tokens.append('I')
            i += 1
        elif string[i] in ['B', 'S']:
            # Skip B and S if they appear separately (legacy parsing)
            i += 1
        else:
            raise ValueError(f"Invalid character '{string[i]}' at position {i}")
    
    if not tokens:
        raise ValueError("No valid tokens found in string")
    
    # Build matrix from tokens
    if tokens[0] == 'I':
        matrix = I
    elif tokens[0] == 'RBS':
        matrix = RandRBS()
    else:
        raise ValueError(f"Invalid first token: {tokens[0]}")
    
    # Process remaining tokens
    for token in tokens[1:]:
        if token == 'I':
            matrix = torch.kron(matrix, I)
        elif token == 'RBS':
            matrix = torch.kron(matrix, RandRBS())
        else:
            raise ValueError(f"Invalid token: {token}")
    
    return matrix

 
# def pyramid_network_rbs(qubits):
#     assert qubits > 1
#     # generate the network for a pyramid configuration (the paper figure 9.a)
#     connections = pyram
#     string = string_from_RBS_connections(connections)
#     return matrix_from_IRBS_string(string)
    # RBS connections should look like: ((1,2)) or ((1,2),(3,4)), ...
# Tuple of tuples
def string_from_RBS_connections(RBS_tuples, qubits):
    """
    Convert RBS connection tuples to a string representation.
    
    RBS gates in the same layer act SIMULTANEOUSLY (in parallel), not sequentially.
    We need to build a tensor product of all gates acting at once.
    
    Args:
        RBS_tuples: List of tuples indicating which qubits are connected
                   e.g., [(1,2), (3,4)] means RBS on qubits 1-2 AND 3-4 simultaneously
        qubits: Total number of qubits
        
    Returns:
        String with 'I' for identity and 'RBS' for beam splitter
        
    Example:
        [(1,2)] with 4 qubits -> "RBSII" (RBS on 1-2, I on 3, I on 4)
        [(1,2), (3,4)] with 4 qubits -> "RBSRBS" (RBS on 1-2, RBS on 3-4, both at once)
        [(3,4), (1,2)] with 4 qubits -> "RBSRBS" (same as above, order doesn't matter for parallel gates)
    """
    if not RBS_tuples:
        # No connections, all identities
        return "I" * qubits
    
    # Create a map of which qubits are covered by RBS gates
    # qubit_map[i] = 'RBS_start' if qubit i is the start of an RBS
    # qubit_map[i] = 'RBS_end' if qubit i is the end of an RBS
    # qubit_map[i] = 'I' if qubit i has identity
    qubit_map = ['I'] * (qubits + 1)  # 1-indexed, so need qubits+1
    
    for connection in RBS_tuples:
        qubit_start = connection[0]
        qubit_end = connection[1]
        
        # Mark these qubits as part of an RBS
        qubit_map[qubit_start] = 'RBS_start'
        qubit_map[qubit_end] = 'RBS_end'
    
    # Build string from qubit map
    gates = []
    i = 1  # Start from qubit 1 (1-indexed)
    while i <= qubits:
        if qubit_map[i] == 'RBS_start':
            gates.append('RBS')
            i += 2  # RBS covers 2 qubits
        elif qubit_map[i] == 'I':
            gates.append('I')
            i += 1
        elif qubit_map[i] == 'RBS_end':
            # This shouldn't happen if RBS gates are properly formed
            # Skip this qubit as it's part of a previous RBS
            i += 1
        else:
            i += 1
    
    return ''.join(gates)


def pyramid_network_rbs(qubits) :
    tuples_list = pyramid(qubits)
    string = string_from_RBS_connections(tuples_list[0],qubits)

    matrix = matrix_from_IRBS_string(string)
    for i in range(1,len(tuples_list)):
        string = string_from_RBS_connections(tuples_list[i],qubits)
        matrix = torch.matmul(matrix,matrix_from_IRBS_string(string))
    return matrix

def upsidown_pyramid_network_rbs(qubits) :
    tuples_list = inverted_pyramid(qubits)
    string = string_from_RBS_connections(tuples_list[0],qubits)

    matrix = matrix_from_IRBS_string(string)
    for i in range(1,len(tuples_list)):
        string = string_from_RBS_connections(tuples_list[i],qubits)
        matrix = torch.matmul(matrix,matrix_from_IRBS_string(string))
    return matrix


# Create a density QNN matrix
# The theta's are initially set, and do not change throughout the neural network steps
# the weights "alpha's" will vary and be randomly mutated, evaluated, and edited to do gradient decent and solve the neural network. 
# # This is the matrix for the QNN step based on 
# 'Training-efficient density quantum machine learning'
# If there are sufficiently enough internal matrixies, and enough training, one or more will be 'selected' and it their weights will be increased, speeding up training and gradient decent.
# will return a function which makes the layer based on the 'alphas'
def density_layer(qubits,matrix_count):
    print(f"Init density layer, qubits: {qubits}")
    # get_one_rbs_network = lambda :  pyramid_network_rbs(qubits) if random() > .5 else upsidown_pyramid_network_rbs(qubits)
    get_one_rbs_network = lambda :  upsidown_pyramid_network_rbs(qubits)


    # the list of internal RBS matricies, not weighted
    RBS_networks = [
        # generate an upsidown or rightside up pyramid rbs ladder
        get_one_rbs_network() for _ in range(matrix_count)        
    ]
    
    def density_layer_function(weights):
        assert len(weights) == matrix_count
        assert len(weights) > 1
        # sum over each of the weighted RBS networks
        sum = torch.mul(RBS_networks[0], weights[0]) 
        for i in range(1,matrix_count):
            sum += torch.mul(RBS_networks[i], weights[i])
        return sum
            
    return density_layer_function
    
# Decompose non-unitary into circuit
