"""
Master Experiment Runner
Runs all experiments systematically and generates final report

This script will:
1. Check if classical baseline exists, run if needed
2. Run quantum experiments with different configurations
3. Generate comparison report
"""
import subprocess
import sys
import time
from pathlib import Path
import json

def check_experiments_exist():
    """Check which experiments have been run"""
    exp_dir = Path('experiments')
    
    classical_exists = len(list((exp_dir / 'classical').glob('*.json'))) > 0 if (exp_dir / 'classical').exists() else False
    quantum_count = len(list((exp_dir / 'quantum').glob('*.json'))) if (exp_dir / 'quantum').exists() else 0
    quantum_opt_count = len(list((exp_dir / 'quantum_optimized').glob('*.json'))) if (exp_dir / 'quantum_optimized').exists() else 0
    
    return {
        'classical': classical_exists,
        'quantum_count': quantum_count,
        'quantum_opt_count': quantum_opt_count
    }

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
        return False

def main():
    print("="*80)
    print("QUANTUM CNN COMPARISON - MASTER EXPERIMENT RUNNER")
    print("="*80)
    
    # Check existing experiments
    status = check_experiments_exist()
    print("\nCurrent Status:")
    print(f"  Classical baseline: {'✓ EXISTS' if status['classical'] else '✗ MISSING'}")
    print(f"  Quantum experiments: {status['quantum_count']} found")
    print(f"  Optimized quantum experiments: {status['quantum_opt_count']} found")
    
    experiments_to_run = []
    
    # Classical baseline (if needed)
    if not status['classical']:
        print("\n⚠ Classical baseline missing - will run first")
        experiments_to_run.append({
            'cmd': [sys.executable, 'cifar10_tinycnn.py'],
            'desc': 'Classical CNN Baseline (15 epochs)',
            'priority': 1
        })
    
    # Quantum experiments
    quantum_configs = [
        {'qubits': 4, 'layers': 2, 'epochs': 5, 'batch': 16},
        {'qubits': 8, 'layers': 2, 'epochs': 5, 'batch': 16},
        {'qubits': 8, 'layers': 4, 'epochs': 5, 'batch': 16},
    ]
    
    for config in quantum_configs:
        experiments_to_run.append({
            'cmd': [
                sys.executable, 'quantum_hybrid_cnn_optimized.py',
                '--local',
                '--epochs', str(config['epochs']),
                '--batch-size', str(config['batch']),
                '--quantum-qubits', str(config['qubits']),
                '--quantum-layers', str(config['layers'])
            ],
            'desc': f"Quantum {config['qubits']}q-{config['layers']}l ({config['epochs']} epochs)",
            'priority': 2
        })
    
    # Show plan
    print(f"\n{'='*80}")
    print(f"EXECUTION PLAN: {len(experiments_to_run)} experiments")
    print(f"{'='*80}")
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"{i}. {exp['desc']}")
    
    # Confirm
    print(f"\n{'='*80}")
    response = input("Start experiments? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    print(f"\n{'='*80}")
    print("STARTING EXPERIMENTS")
    print(f"{'='*80}")
    
    completed = 0
    failed = 0
    
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] {exp['desc']}")
        
        if run_command(exp['cmd'], exp['desc']):
            completed += 1
        else:
            failed += 1
            print(f"⚠ Experiment failed, continuing...")
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    run_command([sys.executable, 'compare_and_visualize.py'], 'Comparison and Visualization')
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{len(experiments_to_run)}")
    print(f"Failed: {failed}/{len(experiments_to_run)}")
    print(f"\nResults saved to: experiments/analysis/")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
