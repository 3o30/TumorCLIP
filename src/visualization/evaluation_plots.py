"""Model-evaluation visualization functions.

Simplified version for the Model Evaluation & Visualization notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_models_comparison_bar(results_dict, fusion_result=None, save_dir="results/plots"):
    """Plot a bar chart comparing model performance."""
    
    print("\nGenerating model comparison bar chart...")
    
    # Prepare data
    model_names = []
    accuracies = []
    colors_list = []
    
    # Add single-modal results (sorted by accuracy)
    sorted_results = sorted(results_dict.items(), 
                           key=lambda x: x[1]['test_accuracy'], 
                           reverse=True)
    
    for name, result in sorted_results:
        model_name = result.get('model_name', name)
        # Clean display name
        display_name = model_name.replace('_', ' ')
        model_names.append(display_name)
        accuracies.append(result['test_accuracy'] * 100)
        colors_list.append('skyblue')
    
    # Add fusion model
    if fusion_result:
        model_names.append('CLIP Fusion')
        accuracies.append(fusion_result['test_accuracy'] * 100)
        colors_list.append('lightcoral')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, len(model_names) * 0.4)))
    
    # Horizontal bars
    bars = ax.barh(model_names, accuracies, color=colors_list, edgecolor='black', linewidth=0.8)
    
    # Titles and labels
    ax.set_xlabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison on Test Set', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(90, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'models_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    
    plt.show()
    
    return save_path


def plot_confusion_matrix_grid(results_dict, class_names, save_dir="results/plots", max_models=6):
    """Plot a grid of confusion matrices for multiple models."""
    
    print("\nGenerating confusion matrices grid...")
    
    # Prepare data
    cm_data = []
    sorted_results = sorted(results_dict.items(), 
                           key=lambda x: x[1]['test_accuracy'], 
                           reverse=True)
    
    # Only keep the top max_models models
    for name, result in sorted_results[:max_models]:
        if 'predictions' in result and 'labels' in result:
            cm = confusion_matrix(result['labels'], result['predictions'])
            model_name = result.get('model_name', name)
            cm_data.append({
                'name': model_name,
                'cm': cm,
                'accuracy': result['test_accuracy']
            })
    
    if not cm_data:
        print("   WARNING: No confusion matrix data available")
        return None
    
    # Layout
    n_models = len(cm_data)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    # Create figure
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.15)
    
    # Shorten class names
    short_names = []
    for name in class_names:
        if 'Glioma' in name:
            short_names.append('Glioma')
        elif 'Meningioma' in name:
            short_names.append('Menin')
        elif 'NORMAL' in name or 'Normal' in name:
            short_names.append('Normal')
        elif 'Neurocitoma' in name or 'Neurocytoma' in name:
            short_names.append('Neuro')
        elif 'Schwannoma' in name:
            short_names.append('Schwan')
        elif 'Outros' in name or 'Other' in name:
            short_names.append('Others')
        else:
            short_names.append(name[:8])
    
    # Plot each confusion matrix
    for idx, data in enumerate(cm_data):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        cm = data['cm']
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        # Heatmap
        im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
        
        # Title
        ax.set_title(f"{data['name']}\nAccuracy: {data['accuracy']:.3f}", 
                    fontsize=11, fontweight='bold')
        
        # Value annotations
        thresh = 50  # 50% threshold
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                percent = cm_percent[i, j]
                text = f'{count}\n({percent:.1f}%)'
                
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if percent > thresh else "black",
                       fontsize=9, fontweight='bold')
        
        # Axes
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(short_names, fontsize=10)
    
    # Global labels
    fig.text(0.5, 0.02, 'Predicted Class', ha='center', fontsize=13, fontweight='bold')
    fig.text(0.02, 0.5, 'True Class', ha='center', rotation='vertical', fontsize=13, fontweight='bold')
    fig.suptitle('Confusion Matrices - Model Comparison', fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'confusion_matrices_grid_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {save_path}")
    
    plt.show()
    
    return save_path


def plot_fusion_vs_best_comparison(best_single_result, fusion_result, save_dir="results/plots"):
    """Plot fusion model vs best single-modal comparison."""
    
    if fusion_result is None:
        print("   INFO: No fusion results to compare")
        return None
    
    print("\nGenerating fusion comparison chart...")
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: accuracy comparison
    models = ['Best Single-Modal', 'CLIP Fusion']
    accuracies = [best_single_result['test_accuracy'] * 100, 
                  fusion_result['test_accuracy'] * 100]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax1.bar(models, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(90, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Subplot 2: performance breakdown
    diff = fusion_result['test_accuracy'] - best_single_result['test_accuracy']
    diff_pct = diff * 100
    
    categories = ['Baseline\nPerformance', 'Fusion\nImprovement', 'Final\nPerformance']
    values = [best_single_result['test_accuracy'] * 100, 
              diff_pct, 
              fusion_result['test_accuracy'] * 100]
    colors2 = ['skyblue', 'lightgreen' if diff > 0 else 'lightyellow', 'lightcoral']
    
    bars2 = ax2.bar(categories, values, color=colors2, width=0.6, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Breakdown', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=best_single_result['test_accuracy'] * 100, color='red', 
               linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline')
    
    # Add value labels
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        if val < 10:  # improvement is usually small
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:+.2f}%' if val == diff_pct else f'{val:.2f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{val:.2f}%',
                    ha='center', va='center', fontweight='bold', fontsize=12, color='black')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'fusion_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    
    plt.show()
    
    return save_path


def plot_detailed_metrics_table(results_dict, save_dir="results/plots"):
    """Plot a detailed metrics table (as a visualization)."""
    
    print("\nGenerating detailed metrics table...")
    
    # Prepare data
    sorted_results = sorted(results_dict.items(), 
                           key=lambda x: x[1]['test_accuracy'], 
                           reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(sorted_results) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    table_data = []
    table_data.append(['Rank', 'Model', 'Val Acc', 'Test Acc', 'Difference', 'Epoch', 'Config'])
    
    for i, (name, result) in enumerate(sorted_results, 1):
        model_name = result.get('model_name', name)
        test_acc = result['test_accuracy']
        val_acc = result.get('val_accuracy', 0)
        diff = test_acc - val_acc
        epoch = result.get('best_epoch', 'N/A')
        optimizer = result.get('optimizer', 'N/A')
        lr = result.get('lr', 0)
        
        table_data.append([
            str(i),
            model_name,
            f'{val_acc:.4f}',
            f'{test_acc:.4f}',
            f'{diff:+.4f}',
            str(epoch),
            f'{optimizer}\nLR={lr}'
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.08, 0.25, 0.12, 0.12, 0.12, 0.08, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header style
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Highlight the top row
    for i in range(len(table_data[0])):
        cell = table[(1, i)]
        cell.set_facecolor('#FFD700')
        cell.set_text_props(weight='bold')
    
    ax.set_title('Detailed Model Performance Metrics', fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'metrics_table_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {save_path}")
    
    plt.show()
    
    return save_path


def plot_single_confusion_matrix(predictions, labels, class_names, model_name, save_dir="results/plots"):
    """Plot a detailed confusion matrix for a single model."""
    
    print(f"\nGenerating confusion matrix for {model_name}...")
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(cm_percent, annot=False, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, vmin=0, vmax=100)
    
    # Value annotations (show counts and percentages)
    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            percent = cm_percent[i, j]
            text = f'{count}\n({percent:.1f}%)'
            
            ax.text(j + 0.5, i + 0.5, text,
                   ha="center", va="center",
                   color="white" if percent > thresh else "black",
                   fontsize=10, fontweight='bold')
    
    # Titles and labels
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(save_dir, f'confusion_matrix_{safe_name}_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {save_path}")
    
    # Print per-class accuracy
    print(f"\n   Per-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = cm_percent[i, i]
        print(f"      {class_name:<30} {class_acc:6.2f}%")
    
    plt.show()
    
    return save_path


def generate_all_evaluation_plots(results_dict, fusion_result=None, class_names=None, save_dir="results/plots"):
    """Generate all evaluation visualization plots."""
    
    print(f"\n{'='*70}")
    print("Generating All Evaluation Visualizations")
    print(f"{'='*70}")
    
    os.makedirs(save_dir, exist_ok=True)
    generated_plots = {}
    
    try:
        # 1. Model-comparison bar chart
        print(f"\n1/4 Model comparison bar chart...")
        bar_path = plot_models_comparison_bar(results_dict, fusion_result, save_dir)
        generated_plots['comparison_bar'] = bar_path
        
        # 2. Confusion-matrix grid
        if class_names:
            print(f"\n2/4 Confusion matrices grid...")
            cm_grid_path = plot_confusion_matrix_grid(results_dict, class_names, save_dir, max_models=6)
            generated_plots['confusion_grid'] = cm_grid_path
        
        # 3. Detailed metrics table
        print(f"\n3/4 Detailed metrics table...")
        table_path = plot_detailed_metrics_table(results_dict, save_dir)
        generated_plots['metrics_table'] = table_path
        
        # 4. Fusion comparison (if available)
        if fusion_result:
            print(f"\n4/4 Fusion comparison...")
            best_single = max(results_dict.items(), key=lambda x: x[1]['test_accuracy'])[1]
            fusion_path = plot_fusion_vs_best_comparison(best_single, fusion_result, save_dir)
            generated_plots['fusion_comparison'] = fusion_path
        else:
            print(f"\n4/4 Fusion comparison... (Skipped - no fusion results)")
        
        print(f"\n{'='*70}")
        print("All visualizations generated!")
        print(f"{'='*70}")
        print("\nGenerated files:")
        for plot_type, path in generated_plots.items():
            if path:
                file_size = os.path.getsize(path) / 1024
                print(f"   {plot_type:<25} {path} ({file_size:.1f} KB)")
        
        return generated_plots
        
    except Exception as e:
        print(f"   ERROR: Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None


