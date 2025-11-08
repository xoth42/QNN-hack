"""
Setup script for pip-based installation (no conda required)

Usage:
    python setup_pip.py
"""
import subprocess
import sys
from pathlib import Path

def print_colored(msg, color='green'):
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m'
    }
    end = '\033[0m'
    print(f"{colors.get(color, '')}{msg}{end}")

def check_python_version():
    """Ensure Python 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}", 'red')
        return False
    print_colored(f"✓ Python {version.major}.{version.minor}.{version.micro}", 'green')
    return True

def install_dependencies():
    """Install all dependencies from requirements.txt"""
    print_colored("\n📦 Installing dependencies...", 'blue')
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print_colored("✓ All dependencies installed successfully!", 'green')
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Installation failed: {e}", 'red')
        return False

def verify_installation():
    """Quick verification of key imports"""
    print_colored("\n🔍 Verifying installation...", 'blue')
    
    try:
        import torch
        print_colored(f"✓ PyTorch {torch.__version__}", 'green')
        
        import torchvision
        print_colored(f"✓ torchvision {torchvision.__version__}", 'green')
        
        import pennylane as qml
        print_colored(f"✓ PennyLane {qml.__version__}", 'green')
        
        import numpy as np
        print_colored(f"✓ NumPy {np.__version__}", 'green')
        
        return True
    except ImportError as e:
        print_colored(f"❌ Import failed: {e}", 'red')
        return False

def main():
    print("="*60)
    print("  Quantum CNN Comparison - Setup (pip)")
    print("="*60)
    
    if not check_python_version():
        return 1
    
    if not install_dependencies():
        return 1
    
    if not verify_installation():
        return 1
    
    print_colored("\n" + "="*60, 'green')
    print_colored("✓ Setup complete!", 'green')
    print_colored("="*60, 'green')
    
    print("\nNext steps:")
    print("  1. Run verification: python verify_setup.py")
    print("  2. Run classical CNN: python cifar10_tinycnn.py")
    print("  3. Run quantum CNN: python quantum_hybrid_cnn.py --local --epochs 2")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
