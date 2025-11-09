from tuple_triangle import inverted_pyramid, pyramid

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
    return torch.tensor([
        [1,     0,          0,          0],
        [0, cos(theta), -sin(theta),    0],
        [0, sin(theta), cos(theta),     0],
        [0,     0,          0,          1]
    ])

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
def string_from_RBS_connections(RBS_tuples: tuple,qubits):
    i = 0
    l = len(RBS_tuples)
    total_str = ""
    position = 1
    while i < l:
        this_tuple = RBS_tuples[i]
        num = this_tuple[0]
        total_str = total_str + "I"*(num-position) + "RBS"
        position = num + 2 # the position of tensors we have done
        i += 1 # read each pair (we will ignore the second number for now, only do local connections between adjacent qubits, todo add nonadjacent connections)
    total_str = total_str + "I"*(qubits+1-position)
    return total_str


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

