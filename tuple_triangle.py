# -*- coding: utf-8 -*-
"""
Entanglement pattern generators for quantum circuits.
Implements patterns from "Training-efficient density quantum machine learning" paper.

Figure 9 patterns:
a) Pyramid circuit - depth 2n-1
b) X circuit - depth n-1  
c) Butterfly circuit - depth log(n)
d) Round-robin circuit - depth n-1

@author: Lydia_Blackwell_Esslami
"""

def generate_inner_list(z):
    """Helper for pyramid patterns"""
    inner_list = []
    
    if z % 2 == 1:
        max_start = z
        for i in range(1, max_start + 1, 2):
            inner_list.append((i, i + 1))
    elif z % 2 == 0:
        max_start = z
        for i in range(2, max_start + 1, 2):
            inner_list.append((i, i + 1))
            
    return inner_list


def inverted_pyramid(n):
    """
    Inverted pyramid (Figure 9a variant) - starts narrow, expands, contracts.
    Depth: 2n-1
    """
    output_list = []
    for x in range(1, n):
        inner_list = generate_inner_list(x)
        output_list.append(inner_list)
    for y in range(n-2, 0, -1):
        inner_list = generate_inner_list(y)
        output_list.append(inner_list)
    
    return output_list


def pyramid(n):
    """
    Pyramid circuit (Figure 9a) - starts wide, contracts, expands.
    Depth: 2n-1
    
    This is the main pattern from the paper.
    """
    output_list = []
    
    def generate_pyramid_list_ascending(z, max_qubit):
        inner_list = []
        
        if z % 2 == 1:
            start_i = max_qubit - 1 if max_qubit % 2 == 0 else max_qubit - 2
            for i in range(start_i, 0, -2):
                steps_from_top = (start_i - i) // 2
                if z >= 1 + (steps_from_top * 2):
                    inner_list.append((i, i + 1))
        elif z % 2 == 0:
            start_i = max_qubit - 1 if max_qubit % 2 == 1 else max_qubit - 2
            for i in range(start_i, 1, -2):
                steps_from_top = (start_i - i) // 2
                if z >= 2 + (steps_from_top * 2):
                    inner_list.append((i, i + 1))
        
        return inner_list
    
    def generate_pyramid_list_descending(z, max_qubit):
        inner_list = []
        
        if z % 2 == 1:
            start_i = max_qubit - 1 if max_qubit % 2 == 0 else max_qubit - 2
            for i in range(start_i, 2, -2):
                steps_from_top = (start_i - i) // 2
                if z >= 1 + (steps_from_top * 2):
                    inner_list.append((i, i + 1))
        elif z % 2 == 0:
            start_i = max_qubit - 1 if max_qubit % 2 == 1 else max_qubit - 2
            for i in range(start_i, 1, -2):
                steps_from_top = (start_i - i) // 2
                if z >= 2 + (steps_from_top * 2):
                    inner_list.append((i, i + 1))
        
        return inner_list
    
    for x in range(1, n):
        output_list.append(generate_pyramid_list_ascending(x, n))
    
    for y in range(n, 0, -1):
        output_list.append(generate_pyramid_list_descending(y, n))
    
    return output_list


def x_circuit(n):
    """
    X circuit (Figure 9b) - crossing pattern.
    Depth: n-1
    
    Creates an X-shaped connectivity pattern where connections cross.
    More efficient than pyramid with fewer layers.
    """
    output_list = []
    
    # First half: connections move from edges to center
    for layer in range(n // 2):
        connections = []
        # Left side moving right
        if layer + 1 <= n - layer - 1:
            connections.append((layer + 1, layer + 2))
        # Right side moving left  
        if n - layer - 1 > layer + 2:
            connections.append((n - layer - 1, n - layer))
        
        if connections:
            output_list.append(connections)
    
    # Second half: connections move from center to edges
    for layer in range(n // 2 - 1, -1, -1):
        connections = []
        if layer + 2 <= n - layer - 1:
            connections.append((layer + 2, layer + 3))
        if n - layer - 2 > layer + 3:
            connections.append((n - layer - 2, n - layer - 1))
        
        if connections:
            output_list.append(connections)
    
    return output_list


def butterfly_circuit(n):
    """
    Butterfly circuit (Figure 9c) - hierarchical splitting pattern.
    Depth: log(n)
    
    Most efficient pattern with logarithmic depth.
    Uses divide-and-conquer approach like FFT butterfly.
    """
    import math
    output_list = []
    
    # Number of stages is log2(n)
    num_stages = int(math.ceil(math.log2(n)))
    
    for stage in range(num_stages):
        connections = []
        stride = 2 ** (stage + 1)
        half_stride = stride // 2
        
        # In each stage, connect qubits separated by half_stride
        for start in range(1, n + 1, stride):
            for offset in range(half_stride):
                qubit1 = start + offset
                qubit2 = qubit1 + half_stride
                if qubit2 <= n:
                    connections.append((qubit1, qubit2))
        
        if connections:
            output_list.append(connections)
    
    # Reverse butterfly for symmetry
    for stage in range(num_stages - 2, -1, -1):
        connections = []
        stride = 2 ** (stage + 1)
        half_stride = stride // 2
        
        for start in range(1, n + 1, stride):
            for offset in range(half_stride):
                qubit1 = start + offset
                qubit2 = qubit1 + half_stride
                if qubit2 <= n:
                    connections.append((qubit1, qubit2))
        
        if connections:
            output_list.append(connections)
    
    return output_list


def round_robin_circuit(n):
    """
    Round-robin circuit (Figure 9d) - dense all-to-all connections.
    Depth: n-1
    
    Most expressive pattern with maximum connectivity.
    Each qubit eventually connects to every other qubit.
    """
    output_list = []
    
    # Each layer connects qubits with a specific stride
    for stride in range(1, n):
        connections = []
        for i in range(1, n + 1):
            j = i + stride
            if j <= n:
                connections.append((i, j))
        
        if connections:
            output_list.append(connections)
    
    return output_list


# Additional useful patterns for experimentation

def brickwork(n):
    """
    Brickwork pattern - standard in many QML papers.
    Alternates between even and odd pairs.
    """
    output_list = []
    
    layer_odd = [(i, i+1) for i in range(1, n, 2)]
    layer_even = [(i, i+1) for i in range(2, n, 2)]
    
    num_layers = max(2, n // 2)
    for _ in range(num_layers):
        if layer_odd:
            output_list.append(layer_odd)
        if layer_even:
            output_list.append(layer_even)
    
    return output_list


def linear_chain(n):
    """Linear nearest-neighbor chain"""
    output_list = []
    
    # Forward
    layer = [(i, i+1) for i in range(1, n)]
    output_list.append(layer)
    
    # Backward
    layer = [(i, i+1) for i in range(n-1, 0, -1)]
    output_list.append(layer)
    
    return output_list


def circular(n):
    """Ring topology with periodic boundary"""
    output_list = []
    
    # Ring connections
    layer = [(i, i+1) for i in range(1, n)]
    output_list.append(layer)
    
    # Close the ring
    if n > 2:
        output_list.append([(n, 1)])
    
    # Cross-connections
    if n >= 4:
        mid = n // 2
        layer = [(i, i + mid) for i in range(1, mid + 1) if i + mid <= n]
        if layer:
            output_list.append(layer)
    
    return output_list
