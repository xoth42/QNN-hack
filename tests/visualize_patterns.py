"""Visualize the entanglement patterns"""
from tuple_triangle import inverted_pyramid, pyramid

def visualize_pattern(pattern_name, pattern_func, n):
    print(f"\n{'='*70}")
    print(f"{pattern_name.upper()} PATTERN (n={n})")
    print(f"{'='*70}")
    
    layers = pattern_func(n)
    print(f"Total layers: {len(layers)}")
    print(f"Total CNOT gates: {sum(len(layer) for layer in layers)}")
    
    print(f"\nLayer structure:")
    for i, layer in enumerate(layers, 1):
        if layer:  # Only show non-empty layers
            print(f"  Layer {i:2d}: {layer}")

# Test with different qubit counts
for n in [4, 8, 10]:
    visualize_pattern("Inverted Pyramid", inverted_pyramid, n)
    visualize_pattern("Pyramid", pyramid, n)
    print()
