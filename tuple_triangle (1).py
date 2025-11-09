# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 03:34:28 2025

@author: Lydia_Blackwell_Esslami
"""
def generate_inner_list(z):
    inner_list = []
    
    # Block for ODD z
    if z % 2 == 1:
        # Since z is odd, the largest odd number <= z is simply z itself.
        max_start = z
        # i goes 1, 3, 5, ...
        for i in range(1, max_start + 1, 2):
            inner_list.append((i, i + 1))
    
    # Block for EVEN z
    elif z % 2 == 0:
        # Since z is even, the largest even number <= z is simply z itself.
        max_start = z
        # i goes 2, 4, 6, ...
        for i in range(2, max_start + 1, 2):
            inner_list.append((i, i + 1))
            
    return inner_list


def inverted_pyramid(n):
    """Gets tuples with the inverted pyramid entanglement algorythm"""
    output_list = []
    for x in range(1,n):
        inner_list = generate_inner_list(x)
        output_list.append(inner_list)
    for y in range(n-2,0, -1):
        inner_list = generate_inner_list(y)
        output_list.append(inner_list)
    
    return output_list

#print(inverted_pyramid(10))

def pyramid(n):
    """Gets tuples with the pyramid entanglemt algorythim"""
    output_list = []

    def generate_inner_list_half_1(z):
        """
        Generates tuples for the first (ascending) half (x loop). 
        Includes the (1,2) tuple.
        """
        inner_list = []
        
        if z % 2 == 1:
            # Odd z: Tuples start at i=7, (7,8), (5,6), ... (1,2).
            # The smallest tuple (1,2) is included when z > 6.
            # We calculate the smallest odd index 'i' that should be included.
            max_index_to_include = z
            
            # Since the original code only checked z>2, z>4, z>6, 
            # the maximum odd starting index is 7.
            max_i = 7 
            
            # The step is 2. The stop condition is when the starting index i exceeds z.
            for i in range(max_i, 0, -2):
                # The pair (i, i+1) is included if the starting index i is <= z's maximum required index.
                # For odd z, the index i should be included if i <= 2*(z//2) + 1.
                # In the original code's fixed 8-bit pattern, we include i if:
                if (i == 7 and z >= 1) or \
                   (i == 5 and z >= 3) or \
                   (i == 3 and z >= 5) or \
                   (i == 1 and z >= 7):
                    inner_list.append((i, i + 1))
                # For a truly scalable version, the logic needs a more complex formula, 
                # but to mimic the *exact* original pattern, we must stick to the fixed pairs:

            # --- Using the parameterizing loop structure for scalability ---
            
            # Find the largest starting index (1, 3, 5, or 7) that the current z supports.
            min_i_to_include = 7
            if z > 6:
                min_i_to_include = 1
            elif z > 4:
                min_i_to_include = 3
            elif z > 2:
                min_i_to_include = 5
            
            stop_i = min_i_to_include - 1 

            for i in range(7, stop_i, -2):
                inner_list.append((i, i + 1))


        elif z % 2 == 0:
            # Even z: Tuples start at i=6, (6,7), (4,5), (2,3).
            min_i_to_include = 6
            if z > 5:
                min_i_to_include = 2
            elif z > 3:
                min_i_to_include = 4

            stop_i = min_i_to_include - 1

            for i in range(6, stop_i, -2):
                inner_list.append((i, i + 1))
                
        return inner_list

    def generate_inner_list_half_2(z):
        """
        Generates tuples for the second (descending) half (y loop).
        EXCLUDES the (1,2) tuple.
        """
        inner_list = []

        if z % 2 == 1:
            # Odd z: Tuples (7,8), (5,6), (3,4). (1,2) is excluded.
            min_i_to_include = 7
            if z > 4:
                min_i_to_include = 3
            elif z > 2:
                min_i_to_include = 5
            
            # Stop condition must ensure i=1 is skipped (stop_i >= 2).
            stop_i = min_i_to_include - 1 
            stop_i = max(2, stop_i) # Forces stop_i >= 2 to exclude (1,2)
            
            for i in range(7, stop_i, -2):
                inner_list.append((i, i + 1))
        
        elif z % 2 == 0:
            # Even z: Tuples (6,7), (4,5), (2,3). (Identical to Half 1 Even logic)
            min_i_to_include = 6
            if z > 5:
                min_i_to_include = 2
            elif z > 3:
                min_i_to_include = 4

            stop_i = min_i_to_include - 1
            
            for i in range(6, stop_i, -2):
                inner_list.append((i, i + 1))

        return inner_list

    # --- Outer Loops (Non-overlapping structure, based on original ranges) ---
    
    # 1. First loop: x = 1 up to n-1 (inclusive) - Uses Half 1 Logic
    for x in range(1, n):
        output_list.append(generate_inner_list_half_1(x))
        
    # 2. Second loop: y = n down to 1 (inclusive) - Uses Half 2 Logic
    for y in range(n, 0, -1):
        output_list.append(generate_inner_list_half_2(y))
        
    return output_list
    
        
print(pyramid(10))



