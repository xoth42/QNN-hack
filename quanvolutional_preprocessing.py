"""
Quanvolutional Neural Network - Preprocessing Approach
Based on: Henderson et al. (2020) "Quanvolutional Neural Networks"

This approach uses quantum circuits as PREPROCESSING filters, not in the training loop.
MUCH faster and more likely to show quantum advantage!

Usage:
    python quanvolutional_preprocessing.py --num-samples 10000 --local
"""
import torch
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle

def create_quanvolutional_filter(n_qubits: int = 4, use_local: bool = True):
    """
    Create a quantum circuit that acts as a convolutional filter
    
    For a 2x2 patch (4 pixels), we use 4 qubits
    The circuit extracts quantum features from the patch
    """
    if use_local:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("braket.aws.qubit",
                        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                        wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_filter(patch):
        """
        Quantum filter for a 2x2 patch
        patch: flattened 2x2 patch (4 values)
        """
        # Encode patch into quantum state (angle encoding)
        for i in range(n_qubits):
            qml.RY(patch[i] * np.pi, wires=i)
        
        # Quantum feature extraction layers
        # Layer 1: Local rotations
        for i in range(n_qubits):
            qml.RZ(np.pi/4, wires=i)
        
        # Layer 2: Entanglement (creates quantum correlations)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])  # Circular
        
        # Layer 3: More rotations
        for i in range(n_qubits):
            qml.RX(np.pi/4, wires=i)
        
        # Layer 4: More entanglement
        for i in range(0, n_qubits-1, 2):
            qml.CNOT(wires=[i, i+1])
        
        # Measure expectations (these become the filtered values)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_filter

def apply_quanvolutional_filter(image, quantum_filter, stride=4):
    """
    Apply quantum filter to an image using sliding window
    OPTIMIZED: Larger stride (4) for faster processing
    
    Args:
        image: (C, H, W) tensor
        quantum_filter: quantum circuit function
        stride: stride for sliding window (default 4 for speed)
    
    Returns:
        filtered_image: quantum-processed image
    """
    C, H, W = image.shape
    patch_size = 2  # 2x2 patches
    
    # Calculate output dimensions
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1
    
    # Process each channel separately
    filtered_channels = []
    
    for c in range(C):
        channel = image[c]
        filtered_channel = torch.zeros(out_h, out_w)
        
        # Slide window over image
        for i in range(out_h):
            for j in range(out_w):
                # Extract 2x2 patch
                h_start = i * stride
                w_start = j * stride
                patch = channel[h_start:h_start+patch_size, w_start:w_start+patch_size]
                
                # Flatten patch and apply quantum filter
                patch_flat = patch.flatten()
                
                # Quantum circuit returns 4 values, take mean as filtered pixel
                quantum_output = quantum_filter(patch_flat)
                filtered_channel[i, j] = torch.mean(torch.stack(quantum_output))
        
        filtered_channels.append(filtered_channel)
    
    # Stack channels back together
    filtered_image = torch.stack(filtered_channels, dim=0)
    
    return filtered_image

def preprocess_dataset(dataset, quantum_filter, num_samples, output_path):
    """
    Preprocess entire dataset with quantum filter
    """
    print(f"Preprocessing {num_samples} images with quantum filter...")
    print("This is a ONE-TIME cost. Processed images will be saved and reused.")
    
    processed_images = []
    labels = []
    
    for idx in tqdm(range(num_samples), desc="Quantum preprocessing"):
        image, label = dataset[idx]
        
        # Apply quantum filter
        filtered_image = apply_quanvolutional_filter(image, quantum_filter)
        
        processed_images.append(filtered_image)
        labels.append(label)
    
    # Save processed dataset
    processed_data = {
        'images': torch.stack(processed_images),
        'labels': torch.tensor(labels)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed_data, output_path)
    
    print(f"\n✓ Saved quantum-preprocessed dataset to: {output_path}")
    print(f"  Shape: {processed_data['images'].shape}")
    
    return processed_data

def visualize_comparison(original, quantum_filtered, save_path):
    """
    Visualize original vs quantum-filtered images
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Original
        orig_img = original[i].permute(1, 2, 0).numpy()
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Quantum filtered
        quant_img = quantum_filtered[i].permute(1, 2, 0).numpy()
        quant_img = (quant_img - quant_img.min()) / (quant_img.max() - quant_img.min())
        axes[1, i].imshow(quant_img)
        axes[1, i].set_title(f'Quantum Filtered {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Original vs Quantum-Preprocessed Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Quanvolutional Preprocessing for CIFAR-10')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of images to preprocess (default: 10000)')
    parser.add_argument('--local', action='store_true',
                       help='Use local simulator (recommended)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of filtered images')
    args = parser.parse_args()
    
    print("="*80)
    print("QUANVOLUTIONAL PREPROCESSING")
    print("="*80)
    print(f"Samples to process: {args.num_samples}")
    print(f"Simulator: {'Local' if args.local else 'AWS Braket'}")
    print("="*80)
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                           download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    
    # Create quantum filter
    print("\nCreating quantum filter...")
    quantum_filter = create_quanvolutional_filter(n_qubits=4, use_local=args.local)
    print("✓ Quantum filter created (4 qubits, 2x2 patches)")
    
    # Preprocess training set
    train_output = Path('data/quanvolutional_train.pt')
    if not train_output.exists():
        train_data = preprocess_dataset(
            trainset, quantum_filter, 
            min(args.num_samples, len(trainset)),
            train_output
        )
    else:
        print(f"\n✓ Training data already preprocessed: {train_output}")
        train_data = torch.load(train_output)
    
    # Preprocess test set (smaller subset)
    test_output = Path('data/quanvolutional_test.pt')
    test_samples = min(2000, len(testset))
    if not test_output.exists():
        test_data = preprocess_dataset(
            testset, quantum_filter,
            test_samples,
            test_output
        )
    else:
        print(f"\n✓ Test data already preprocessed: {test_output}")
        test_data = torch.load(test_output)
    
    # Visualize if requested
    if args.visualize:
        print("\nCreating visualization...")
        original_images = torch.stack([trainset[i][0] for i in range(5)])
        visualize_comparison(
            original_images,
            train_data['images'][:5],
            Path('experiments/quanvolutional_comparison.png')
        )
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run: python train_quanvolutional_cnn.py")
    print("2. Compare results with classical baseline")
    print("="*80)

if __name__ == '__main__':
    main()
