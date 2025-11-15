from tuple_triangle import (
    pyramid, inverted_pyramid, x_circuit, butterfly_circuit, round_robin_circuit,
    brickwork, linear_chain, circular
)

import torch
from numpy import pi,cos,sin
from random import random
from torch import kron
from functools import reduce
twopi: float = 2 * pi # my computational physics teacher told me do this

I = torch.eye(2)
# TODO TESTS FOR THIS SHYTTE
# "hamming weight preserving unitary"
# "Reconfigurable beam splitter"
def RBS(theta):
    # ensure we use the same dtype as the global identity matrix `I` to avoid dtype mismatches
    return torch.tensor([
        [1,     0,          0,          0],
        [0, cos(theta), -sin(theta),    0],
        [0, sin(theta), cos(theta),     0],
        [0,     0,          0,          1]
    ], dtype=I.dtype)

# random theta from 0 to 2pi
def get_theta():
    return random() * twopi


# RBS with random theta
def RandRBS():
    return RBS(get_theta())
 
 
def matrix_from_IRBS_string(string):
    l = len(string)
    assert l > 0
    
    matrix = I
    #first
    i = 0
    if string[i] == 'I':
        matrix = I
    elif string[i] == 'R':
        matrix= RandRBS()
    
    i += 1

    while i < l:
        if string[i] == 'I':
            matrix = torch.kron(matrix,I)
            i += 1
        elif string[i] == 'R':
            matrix= torch.kron(matrix,RandRBS())
            i += 1
        elif (string[i] == 'B') or  (string[i] == 'S'):
            i += 1
        else:
            # invalid string creation
            assert False
    return matrix

 
# def pyramid_network_rbs(qubits):
#     assert qubits > 1
#     # generate the network for a pyramid configuration (the paper figure 9.a)
#     connections = pyram
#     string = string_from_RBS_connections(connections)
#     return matrix_from_IRBS_string(string)
    # RBS connections should look like: ((1,2)) or ((1,2),(3,4)), ...
# Tuple of tuples
# qubits is the total number of qubits, rbs tuples should not go past the num qubits
# When calling the RBS gate, we will asign a random theta. Since it should not change anywhere else, this is fine.
def string_from_RBS_connections(RBS_tuples, qubits):
    """
    Convert RBS connection tuples to string representation.
    
    Handles parallel gates correctly - gates in same layer act simultaneously.
    
    Args:
        RBS_tuples: List of tuples like [(1,2), (3,4)] for parallel RBS gates
        qubits: Total number of qubits
        
    Returns:
        String with 'I' for identity and 'RBS' for beam splitter
    """
    if not RBS_tuples:
        return "I" * qubits
    
    # Create a map of which qubits are covered by RBS gates
    qubit_coverage = {}  # qubit_num -> 'RBS_start' or 'RBS_end'
    
    for connection in RBS_tuples:
        q1, q2 = connection[0], connection[1]
        # Ensure q1 < q2 and they're adjacent
        if q1 > q2:
            q1, q2 = q2, q1
        
        qubit_coverage[q1] = 'RBS_start'
        qubit_coverage[q2] = 'RBS_end'
    
    # Build string by scanning through qubits
    result = []
    i = 1
    while i <= qubits:
        if i in qubit_coverage and qubit_coverage[i] == 'RBS_start':
            result.append('RBS')
            i += 2  # RBS covers 2 qubits
        else:
            result.append('I')
            i += 1
    
    return ''.join(result)


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
    string: Unknown = string_from_RBS_connections(tuples_list[0],qubits)

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
def get_entanglement_pattern(pattern_name, qubits):
    """
    Get entanglement pattern by name.
    
    Paper patterns (Figure 9):
    - 'pyramid': Pyramid circuit (depth 2n-1)
    - 'x_circuit': X circuit (depth n-1)  
    - 'butterfly': Butterfly circuit (depth log(n))
    - 'round_robin': Round-robin circuit (depth n-1)
    
    Additional patterns:
    - 'inverted_pyramid': Inverted pyramid variant
    - 'brickwork': Standard brickwork pattern
    - 'linear': Linear chain
    - 'circular': Ring topology
    """
    patterns = {
        # Paper patterns (Figure 9)
        'pyramid': pyramid,
        'x_circuit': x_circuit,
        'butterfly': butterfly_circuit,
        'round_robin': round_robin_circuit,
        # Additional patterns
        'inverted_pyramid': inverted_pyramid,
        'brickwork': brickwork,
        'linear': linear_chain,
        'circular': circular,
    }
    
    if pattern_name not in patterns:
        raise ValueError(f"Unknown pattern: {pattern_name}. Available: {list(patterns.keys())}")
    
    return patterns[pattern_name](qubits)


def create_rbs_network_from_pattern(pattern_name, qubits):
    """Create RBS network matrix from entanglement pattern"""
    tuples_list = get_entanglement_pattern(pattern_name, qubits)
    
    if not tuples_list:
        # Return identity if no connections
        return torch.eye(2**qubits, dtype=torch.float32)
    
    string = string_from_RBS_connections(tuples_list[0], qubits)
    matrix = matrix_from_IRBS_string(string)
    
    for i in range(1, len(tuples_list)):
        string = string_from_RBS_connections(tuples_list[i], qubits)
        matrix = torch.matmul(matrix, matrix_from_IRBS_string(string))
    
    return matrix


def density_layer(qubits, matrix_count, patterns=None):
    """
    Create a density QNN layer with multiple entanglement patterns.
    
    Implements the density matrix approach from:
    "Training-efficient density quantum machine learning"
    
    Args:
        qubits: Number of qubits
        matrix_count: Number of sub-unitaries to mix
        patterns: List of pattern names to use. If None, uses all 4 paper patterns.
                 Paper patterns (Figure 9):
                   - 'pyramid': depth 2n-1, balanced
                   - 'x_circuit': depth n-1, crossing
                   - 'butterfly': depth log(n), most efficient
                   - 'round_robin': depth n-1, most expressive
    
    Returns:
        Function that takes weights and returns weighted density matrix
    """
    print(f"Init density layer, qubits: {qubits}, sub-unitaries: {matrix_count}")
    
    # Default: use all 4 patterns from the paper (Figure 9)
    if patterns is None:
        paper_patterns = ['pyramid', 'x_circuit', 'butterfly', 'round_robin']
        patterns = [paper_patterns[i % len(paper_patterns)] for i in range(matrix_count)]
    elif isinstance(patterns, str):
        # Single pattern for all
        patterns = [patterns] * matrix_count
    elif len(patterns) < matrix_count:
        # Cycle through provided patterns
        patterns = [patterns[i % len(patterns)] for i in range(matrix_count)]
    
    print(f"Using entanglement patterns: {patterns}")
    
    # Generate RBS networks with different entanglement patterns
    RBS_networks = []
    for i, pattern in enumerate(patterns):
        try:
            network = create_rbs_network_from_pattern(pattern, qubits)
            RBS_networks.append(network)
        except Exception as e:
            print(f"Warning: Failed to create pattern '{pattern}': {e}")
            # Fallback to identity
            RBS_networks.append(torch.eye(2**qubits, dtype=torch.float32))
    
    def density_layer_function(weights):
        """
        Compute weighted sum of RBS networks (density matrix).
        
        Args:
            weights: Tensor of shape (matrix_count,) with mixing coefficients
            
        Returns:
            Density matrix of shape (2^qubits, 2^qubits)
        """
        if len(weights) != matrix_count:
            raise ValueError(f"Expected {matrix_count} weights, got {len(weights)}")
        
        # Normalize weights to sum to 1 (convex combination)
        weights_normalized = torch.softmax(weights, dim=0)
        
        # Weighted sum of RBS networks
        density_matrix = torch.mul(RBS_networks[0], weights_normalized[0])
        for i in range(1, matrix_count):
            density_matrix = density_matrix + torch.mul(RBS_networks[i], weights_normalized[i])
        
        return density_matrix
            
    return density_layer_function
    
# Decompose non-unitary into circuit

