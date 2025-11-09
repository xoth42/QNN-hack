"""
Comprehensive Comparison and Visualization Tool
Analyzes all experiments and generates detailed reports
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns

sns.set_style("whitegrid")

def load_all_experiments():
    """Load all experiment JSON files"""
    experiments = {
        'classical': [],
        'quantum': [],
        'quantum_optimized': []
    }
    
    exp_dir = Path('experiments')
    
    for model_type in ['classical', 'quantum', 'quantum_optimized']:
        type_dir = exp_dir / model_type
        if type_dir.exists():
            for json_file in type_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        data['filename'] = json_file.name
                        experiments[model_type].append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    return experiments

def plot_loss_curves(experiments, output_dir):
    """Plot training/validation loss curves for all experiments"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_type, exps) in enumerate(experiments.items()):
        if not exps:
            continue
        
        ax = axes[idx]
        for exp in exps:
            train_loss = exp['results']['train_loss']
            val_loss = exp['results']['val_loss']
            epochs = range(1, len(train_loss) + 1)
            
            label = exp['description'][:30]
            ax.plot(epochs, train_loss, '--', alpha=0.7, label=f'{label} (train)')
            ax.plot(epochs, val_loss, '-', alpha=0.9, label=f'{label} (val)')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{model_type.replace("_", " ").title()} Models', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'loss_curves.png'}")
    plt.close()

def plot_accuracy_comparison(experiments, output_dir):
    """Bar chart comparing test accuracies"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_types = []
    accuracies = []
    colors = []
    labels = []
    
    color_map = {
        'classical': '#2ecc71',
        'quantum': '#3498db',
        'quantum_optimized': '#9b59b6'
    }
    
    for model_type, exps in experiments.items():
        for exp in exps:
            model_types.append(model_type)
            accuracies.append(exp['results']['test_accuracy'])
            colors.append(color_map.get(model_type, '#95a5a6'))
            
            # Create label
            desc = exp['description'][:20]
            if 'quantum_qubits' in exp['hyperparameters']:
                qubits = exp['hyperparameters']['quantum_qubits']
                layers = exp['hyperparameters'].get('quantum_layers', '?')
                desc = f"{qubits}q-{layers}l"
            labels.append(f"{model_type}\n{desc}")
    
    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[k], label=k.replace('_', ' ').title()) 
                      for k in color_map.keys() if experiments[k]]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()

def plot_training_time_comparison(experiments, output_dir):
    """Compare training times"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_types = []
    times = []
    colors = []
    labels = []
    
    color_map = {
        'classical': '#2ecc71',
        'quantum': '#3498db',
        'quantum_optimized': '#9b59b6'
    }
    
    for model_type, exps in experiments.items():
        for exp in exps:
            time_sec = exp['results']['training_time_seconds']
            if time_sec > 0:
                model_types.append(model_type)
                times.append(time_sec / 60)  # Convert to minutes
                colors.append(color_map.get(model_type, '#95a5a6'))
                
                desc = exp['description'][:20]
                if 'quantum_qubits' in exp['hyperparameters']:
                    qubits = exp['hyperparameters']['quantum_qubits']
                    layers = exp['hyperparameters'].get('quantum_layers', '?')
                    desc = f"{qubits}q-{layers}l"
                labels.append(f"{model_type}\n{desc}")
    
    x = np.arange(len(labels))
    bars = ax.bar(x, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, time_min in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_min:.1f}m',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Training Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'training_time_comparison.png'}")
    plt.close()

def plot_quantum_analysis(experiments, output_dir):
    """Analyze quantum experiments: qubits vs accuracy, layers vs accuracy"""
    quantum_exps = experiments['quantum'] + experiments['quantum_optimized']
    
    if not quantum_exps:
        print("No quantum experiments to analyze")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Qubits vs Accuracy
    qubits_data = {}
    for exp in quantum_exps:
        if 'quantum_qubits' in exp['hyperparameters']:
            qubits = exp['hyperparameters']['quantum_qubits']
            acc = exp['results']['test_accuracy']
            if qubits not in qubits_data:
                qubits_data[qubits] = []
            qubits_data[qubits].append(acc)
    
    if qubits_data:
        qubits = sorted(qubits_data.keys())
        avg_accs = [np.mean(qubits_data[q]) for q in qubits]
        std_accs = [np.std(qubits_data[q]) if len(qubits_data[q]) > 1 else 0 for q in qubits]
        
        ax1.errorbar(qubits, avg_accs, yerr=std_accs, marker='o', markersize=10,
                    linewidth=2, capsize=5, capthick=2, color='#3498db')
        ax1.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Qubit Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Layers vs Accuracy
    layers_data = {}
    for exp in quantum_exps:
        if 'quantum_layers' in exp['hyperparameters']:
            layers = exp['hyperparameters']['quantum_layers']
            acc = exp['results']['test_accuracy']
            if layers not in layers_data:
                layers_data[layers] = []
            layers_data[layers].append(acc)
    
    if layers_data:
        layers = sorted(layers_data.keys())
        avg_accs = [np.mean(layers_data[l]) for l in layers]
        std_accs = [np.std(layers_data[l]) if len(layers_data[l]) > 1 else 0 for l in layers]
        
        ax2.errorbar(layers, avg_accs, yerr=std_accs, marker='s', markersize=10,
                    linewidth=2, capsize=5, capthick=2, color='#9b59b6')
        ax2.set_xlabel('Number of Quantum Layers', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy vs Circuit Depth', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quantum_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'quantum_analysis.png'}")
    plt.close()

def generate_summary_report(experiments, output_dir):
    """Generate text summary report"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("QUANTUM CNN COMPARISON - SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")
    
    for model_type, exps in experiments.items():
        if not exps:
            continue
        
        report_lines.append(f"\n{model_type.upper().replace('_', ' ')} MODELS")
        report_lines.append("-"*80)
        
        accuracies = [exp['results']['test_accuracy'] for exp in exps]
        times = [exp['results']['training_time_seconds']/60 for exp in exps if exp['results']['training_time_seconds'] > 0]
        
        report_lines.append(f"Number of experiments: {len(exps)}")
        report_lines.append(f"Average accuracy: {np.mean(accuracies):.2f}% (±{np.std(accuracies):.2f}%)")
        report_lines.append(f"Best accuracy: {np.max(accuracies):.2f}%")
        report_lines.append(f"Worst accuracy: {np.min(accuracies):.2f}%")
        if times:
            report_lines.append(f"Average training time: {np.mean(times):.1f} minutes")
        
        report_lines.append("\nIndividual Experiments:")
        for exp in exps:
            acc = exp['results']['test_accuracy']
            time_min = exp['results']['training_time_seconds'] / 60
            desc = exp['description']
            epochs = exp['hyperparameters'].get('epochs', '?')
            report_lines.append(f"  - {desc}: {acc:.2f}% ({epochs} epochs, {time_min:.1f} min)")
    
    # Comparison
    report_lines.append("\n" + "="*80)
    report_lines.append("CLASSICAL VS QUANTUM COMPARISON")
    report_lines.append("="*80)
    
    if experiments['classical'] and (experiments['quantum'] or experiments['quantum_optimized']):
        classical_acc = np.mean([exp['results']['test_accuracy'] for exp in experiments['classical']])
        quantum_acc = np.mean([exp['results']['test_accuracy'] 
                              for exp in experiments['quantum'] + experiments['quantum_optimized']])
        
        report_lines.append(f"Classical average: {classical_acc:.2f}%")
        report_lines.append(f"Quantum average: {quantum_acc:.2f}%")
        report_lines.append(f"Difference: {quantum_acc - classical_acc:+.2f}%")
        
        if quantum_acc > classical_acc:
            report_lines.append("✓ Quantum shows advantage!")
        elif quantum_acc > classical_acc * 0.9:
            report_lines.append("≈ Quantum competitive (within 10%)")
        else:
            report_lines.append("✗ Classical outperforms quantum")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved: {report_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report_lines))

def main():
    print("Loading experiments...")
    experiments = load_all_experiments()
    
    # Count experiments
    total = sum(len(exps) for exps in experiments.values())
    print(f"Found {total} experiments:")
    for model_type, exps in experiments.items():
        print(f"  - {model_type}: {len(exps)}")
    
    if total == 0:
        print("\n❌ No experiments found! Run some experiments first.")
        return
    
    # Create output directory
    output_dir = Path('experiments/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    plot_loss_curves(experiments, output_dir)
    plot_accuracy_comparison(experiments, output_dir)
    plot_training_time_comparison(experiments, output_dir)
    plot_quantum_analysis(experiments, output_dir)
    
    print("\nGenerating summary report...")
    generate_summary_report(experiments, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ Analysis complete! Check experiments/analysis/ for results")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
