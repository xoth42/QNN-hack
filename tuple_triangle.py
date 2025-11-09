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
    for x in range(1,n):
        inner_list = []
        if x%2 == 1:
            inner_list.append((9,10))
            if x>2:
                inner_list.append((7,8))
                if x>4:
                    inner_list.append((5,6))
                    if x>6:
                        inner_list.append((3,4))
                        if x>8:
                            inner_list.append((1,2))
        if x%2 == 0:
          inner_list.append((8,9))  
          if x>3:
              inner_list.append((6,7))
              if x>5:
                  inner_list.append((4,5))
                  if x>7:
                      inner_list.append((2,3))
        output_list.append(inner_list)
    for y in range(n,0, -1):
        inner_list = []
        if y%2 == 1:
            inner_list.append((9,10))
            if y>2:
                inner_list.append((7,8))
                if y>4:
                    inner_list.append((5,6))
                    if y>6:
                        inner_list.append((3,4))
                    
        if y%2 == 0:
          inner_list.append((8,9))  
          if y>3:
              inner_list.append((6,7))
              if y>5:
                  inner_list.append((4,5))
                  if y>7:
                      inner_list.append((2,3))
        output_list.append(inner_list)
    
    return output_list
    
        
# print(pyramid(10))



