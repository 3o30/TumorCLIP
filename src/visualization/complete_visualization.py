"""Complete model-evaluation visualization system.

Professional visualization functions migrated from models_comparation.ipynb.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from datetime import datetime


# Default class names
DEFAULT_CANONICAL = ['Glioma', 'Meningioma', 'NORMAL', 'Neurocitoma', 'Outros Tipos de Lesões', 'Schwannoma']

# Professional color palette
PALETTE = {
    'light_models': '#66C2A5',    # lightweight models
    'trad_cnn': '#FC8D62',        # traditional CNN
    'transformer': '#8DA0CB',     # Transformer
    'other': '#E78AC3',           # other
    'multimodal': '#A6D854'       # multimodal
}



import os
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curves_comparison(single_modal_results, clip_results=None, save_dir="results/visualizations"):
    """Plot ROC-curve comparisons for single-modal and CLIP models."""
    print("Generating ROC-curve comparison...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_data = []
    
    # Process single-modal results
    for result in single_modal_results:
        if 'test_results' in result and result['test_results']:
            test_result = result['test_results'][0]  # take the first test result
            if 'probabilities' in test_result and 'true_labels' in test_result:
                model_data.append({
                    'name': result['name'],
                    'y_true': np.array(test_result['true_labels']),
                    'y_prob': np.array(test_result['probabilities']),
                    'type': 'Single-Modal'
                })
    
    # Process CLIP results
    if clip_results:
        for result in clip_results:
            if 'test_probabilities' in result and 'test_labels' in result:
                model_data.append({
                    'name': result['name'],
                    'y_true': np.array(result['test_labels']),
                    'y_prob': np.array(result['test_probabilities']),
                    'type': 'Multi-Modal (CLIP)'
                })
    
    if not model_data:
        print("WARNING: No usable probability data found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    for i, data in enumerate(model_data):
        color = next(colors)
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        # Compute ROC curve
        if y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            y_true_bin = label_binarize(y_true, classes=range(y_prob.shape[1]))
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            roc_auc = auc(fpr, tpr)
        
        linestyle = '-' if data['type'] == 'Single-Modal' else '--'
        ax1.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2.5, 
                 label=f"{data['name']} (AUC={roc_auc:.3f})")
    
    # Add diagonal baseline
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: zoomed view (FPR < 0.2, TPR > 0.8)
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    for i, data in enumerate(model_data):
        color = next(colors)
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        if y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=range(y_prob.shape[1]))
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        
        linestyle = '-' if data['type'] == 'Single-Modal' else '--'
        ax2.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2.5, 
                 label=data['name'])
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax2.set_xlim(0.0, 0.2)
    ax2.set_ylim(0.8, 1.0)
    ax2.set_xlabel('False Positive Rate (Zoomed)', fontsize=14)
    ax2.set_ylabel('True Positive Rate (Zoomed)', fontsize=14)
    ax2.set_title('ROC Curves (Zoomed View)', fontsize=16, fontweight='bold')
    ax2.axvline(0.05, color='gray', linestyle=':', alpha=0.6, label='FPR=5%')
    ax2.legend(loc="lower right", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    
    # Save figure
    save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ROC plot saved: {save_path}")
    return save_path


def plot_simple_performance_comparison(single_modal_results, clip_results=None,
                                       save_dir="results/visualizations",
                                       filename="performance_simple.png"):
    """
    Compact grouped bar chart:
    - Shows only Accuracy / Sensitivity / Specificity / F1 (no AUC).
    - Sorted by Accuracy.
    - No numeric labels on bars.
    - Uses an academic palette (falls back to blues if not defined).
    - High-resolution output with larger fonts.
    """
    import os, numpy as np, matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ---------- Helpers ----------
    def macro_specificity(cm):
        if cm is None: return None
        cm = np.array(cm)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]: return None
        specs = []
        for c in range(cm.shape[0]):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            denom = tn + fp
            if denom > 0:
                specs.append(tn / denom)
        return float(np.mean(specs)) if specs else None

    def collect_metrics_from_result(test_result, name_for_plot):
        report = (test_result.get('test_report') or {})
        cm = test_result.get('confusion_matrix')

        if 'macro avg' in report:
            rec = report['macro avg'].get('recall')
            f1  = report['macro avg'].get('f1-score')
        else:
            cls = [k for k in report.keys() if k not in ('accuracy','macro avg','weighted avg')]
            if cls:
                rec = float(np.mean([report[c]['recall'] for c in cls]))
                f1  = float(np.mean([report[c]['f1-score'] for c in cls]))
            else:
                rec = f1 = None

        acc = (test_result.get('test_accuracy')
               or test_result.get('final_test_acc')
               or test_result.get('accuracy'))

        spec = macro_specificity(cm)

        return {
            'name': name_for_plot,
            'Accuracy': acc,
            'Sensitivity': rec,
            'Specificity': spec,
            'F1': f1
        }

    # ---------- Data aggregation ----------
    rows = []
    for sm in (single_modal_results or []):
        if sm.get('test_results'):
            rows.append(collect_metrics_from_result(sm['test_results'][0], sm['name']))

    for cr in (clip_results or []):
        pseudo = {
            'test_report': cr.get('test_report'),
            'confusion_matrix': cr.get('confusion_matrix'),
            'final_test_acc': cr.get('final_test_acc'),
        }
        rows.append(collect_metrics_from_result(pseudo, cr.get('name','CLIP Fusion Model')))

    if not rows:
        print("WARNING: No usable metric data found")
        return None

    # ---------- Sort by Accuracy ----------
    rows = sorted(rows, key=lambda r: (r['Accuracy'] if r['Accuracy'] is not None else 0), reverse=True)

    metric_order = ['Accuracy','Sensitivity','Specificity','F1']
    metric_order = [m for m in metric_order if any(r[m] is not None for r in rows)]

    # ---------- Colors ----------
    try:
        base = list(palette)  # reuse the palette defined in your notebook
    except NameError:
        base = ['#1b4f72', '#2e86c1', '#5dade2', '#a9cce3']  # fallback blues
    def pick(i): return base[i % len(base)]
    metric_colors = {m: pick(i) for i,m in enumerate(metric_order)}

    # ---------- Plot ----------
    model_names = [r['name'] for r in rows]
    n_models, n_metrics = len(model_names), len(metric_order)
    x = np.arange(n_models)
    width = min(0.8 / max(n_metrics,1), 0.18)

    fig_h = 5.5
    fig_w = max(10, 1.2 * n_models + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Collect valid metric values to set Y-axis range
    all_vals = []
    for i, m in enumerate(metric_order):
        vals = [(rows[j][m] * 100 if rows[j][m] is not None else np.nan) for j in range(n_models)]
        all_vals.extend([v for v in vals if not np.isnan(v)])
        xs = x + i*width - (n_metrics-1)*width/2
        ax.bar(xs, vals, width, label=m, color=metric_colors[m], alpha=0.95)

    # Dynamically set Y-axis range to better highlight differences
    if all_vals:
        y_min = min(all_vals)
        y_max = max(all_vals)
        # For readability: add padding below min and above max (capped at 0..100)
        y_range = y_max - y_min
        y_bottom = max(0, y_min - y_range * 0.15)  # 15% padding below min (not below 0)
        y_top = min(100, y_max + 2)  # +2 points above max (capped at 100)
        ax.set_ylim(y_bottom, y_top)
    else:
        ax.set_ylim(0, 100)
    
    ax.set_ylabel("Score (%)", fontsize=16)  # larger Y-axis label font
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=14)  # larger X-axis tick font
    ax.set_title("Performance Comparison (macro where applicable)", fontsize=18, pad=35)  # larger title and more top padding
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(ncol=min(4, n_metrics), loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=14)  # adjust legend position
    
    # Increase tick-label font sizes
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename)
    # Increase DPI to improve clarity
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"High-resolution performance comparison saved: {out_path}")
    return out_path

def plot_bubble_chart_compact(single_modal_results, clip_results=None, save_dir="results/visualizations"):
    """Plot a compact bubble-chart comparison."""
    print("Generating bubble-chart comparison...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define palette
    palette = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854']
    
    bubble_data = []
    
    # Process single-modal results
    for result in single_modal_results:
        if 'test_results' in result and result['test_results']:
            test_result = result['test_results'][0]
            if 'test_accuracy' in test_result:
                param_count = result.get('param_count', 0) / 1e6
                complexity = result.get('model_complexity', 1.0)
                
                bubble_data.append({
                    'name': result['name'],
                    'accuracy': test_result['test_accuracy'] * 100,
                    'parameters': param_count,
                    'complexity': complexity,
                    'type': 'Single-Modal',
                    'color': palette[0] if 'EfficientNet' in result['name'] or 'MobileNet' in result['name'] 
                            else palette[1] if 'ResNet' in result['name'] or 'DenseNet' in result['name']
                            else palette[2] if 'ViT' in result['name'] or 'Transformer' in result['name']
                            else palette[3]
                })
    
    # Process CLIP results
    if clip_results:
        for result in clip_results:
            if 'final_test_acc' in result:
                param_count = result.get('model_params', 0) / 1e6
                complexity = result.get('model_complexity', 15.0)
                
                bubble_data.append({
                    'name': result['name'],
                    'accuracy': result['final_test_acc'] * 100,
                    'parameters': param_count,
                    'complexity': complexity,
                    'type': 'Multi-Modal (CLIP)',
                    'color': palette[4]
                })
    
    if not bubble_data:
        print("No usable data found")
        return
    
    # Create figure - compact version
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Recompute bubble sizes to enhance visual separation
    parameters = [d['parameters'] for d in bubble_data]
    min_param = min(parameters)
    max_param = max(parameters)
    
    if max_param > min_param:
        # Use sqrt scaling: keep small bubbles visible while amplifying differences
        sqrt_params = [np.sqrt(p) for p in parameters]
        sqrt_min = np.sqrt(min_param)
        sqrt_max = np.sqrt(max_param)
        # Base size 150 with max increment 1200 (range 150-1350)
        sizes = [150 + (sqrt_p - sqrt_min) / (sqrt_max - sqrt_min) * 1200 for sqrt_p in sqrt_params]
    else:
        sizes = [400] * len(bubble_data)
    
    # Draw bubble chart
    for i, data in enumerate(bubble_data):
        marker = 'o' if data['type'] == 'Single-Modal' else 's'
        alpha = 0.7 if data['type'] == 'Single-Modal' else 0.8
        
        ax.scatter(data['parameters'], data['accuracy'], 
                  s=sizes[i], c=data['color'], marker=marker, 
                  alpha=alpha, edgecolors='black', linewidth=1.5)
        
        # Do not show labels (keep chart clean)
        # ax.annotate(data['name'], 
        #            (data['parameters'], data['accuracy']),
        #            xytext=(8, 8), textcoords='offset points',
        #            fontsize=12, fontweight='bold',
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Axes and title
    ax.set_xlabel('Parameters (millions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Analysis', fontsize=15, fontweight='bold', pad=15)
    
    # Simplified legend
    legend_elements = [
        plt.scatter([], [], s=100, c=palette[0], label='Lightweight', alpha=0.7),
        plt.scatter([], [], s=100, c=palette[1], label='Traditional CNN', alpha=0.7),
        plt.scatter([], [], s=100, c=palette[2], label='Transformer', alpha=0.7),
        plt.scatter([], [], s=100, c=palette[3], label='Other Models', alpha=0.7),
        plt.scatter([], [], s=100, c=palette[4], label='Multi-Modal', alpha=0.7)
    ]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.2),
             fontsize=11, frameon=True, title='Model Types', title_fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'bubble_chart_compact.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Bubble chart saved: {save_path}")
    return save_path

print("Compact bubble-chart function defined")

def _extract_class_names_from_report(test_report, canonical=None):
    """Extract class names from a classification report; replace Class_0-style names with canonical when needed."""
    if canonical is None:
        canonical = DEFAULT_CANONICAL
    
    if test_report is None:
        return canonical
    
    # Extract class names from report
    class_names = []
    for key in test_report.keys():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            class_names.append(key)
    
    # If names are like Class_0/Class_1, replace using canonical
    if class_names and all('Class_' in name or name.startswith('Class') for name in class_names):
        return canonical[:len(class_names)]
    
    # Otherwise return original class names
    return class_names if class_names else canonical


def plot_super_compact_confusion_matrices(single_modal_results, clip_results=None, save_dir="results/visualizations", canonical_class_names=DEFAULT_CANONICAL):
    """Super-compact confusion matrices optimized for tight column spacing; can force true labels via canonical_class_names."""
    print("Generating super-compact confusion-matrix comparison...")
    os.makedirs(save_dir, exist_ok=True)

    cm_data = []

    # Process single-modal
    for result in single_modal_results:
        if 'test_results' in result and result['test_results']:
            test_result = result['test_results'][0]
            if 'confusion_matrix' in test_result and 'test_report' in test_result:
                cm = np.array(test_result['confusion_matrix'])
                # Single-modal reports often already use real class names; still normalize for consistency
                class_names = _extract_class_names_from_report(test_result['test_report'], canonical=canonical_class_names)
                accuracy = test_result['test_accuracy']
                cm_data.append({
                    'name': result['name'],
                    'cm': cm,
                    'class_names': class_names,
                    'accuracy': accuracy,
                    'type': 'SM'
                })

    # Process CLIP (often Class_0..Class_5; override with canonical here)
    if clip_results:
        for result in clip_results:
            if 'confusion_matrix' in result and 'test_report' in result:
                cm = np.array(result['confusion_matrix'])
                class_names = _extract_class_names_from_report(result['test_report'], canonical=canonical_class_names)
                accuracy = result.get('final_test_acc', result.get('test_accuracy', 0.0))
                cm_data.append({
                    'name': result['name'],
                    'cm': cm,
                    'class_names': class_names,
                    'accuracy': accuracy,
                    'type': 'MM'
                })

    if not cm_data:
        print("No usable confusion-matrix data found")
        return

    # ==== Layout ====
    n_models = len(cm_data)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    plt.style.use('default')
    fig = plt.figure(figsize=(4.8 * cols, 4.5 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.25, wspace=0.08, left=0.05, right=0.95, top=0.92, bottom=0.12)
    print(f"Layout: {n_models} models, column spacing significantly reduced")

    for i, data in enumerate(cm_data):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])

        cm = data['cm']
        cm_counts = cm  # use raw counts as the heatmap base
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # compute percentages (optional)

        # Heatmap: use integer counts as the background
        im = ax.imshow(cm_counts, interpolation='nearest', cmap='Blues')

        ax.set_title(f"{data['name']}\n({data['type']}) Acc: {data['accuracy']:.3f}", fontsize=11, fontweight='bold', pad=5)

        # Annotations: show integer counts
        thresh = cm_counts.max() / 2.0
        for i_cm in range(cm_counts.shape[0]):
            for j_cm in range(cm_counts.shape[1]):
                count = int(cm_counts[i_cm, j_cm])  # force integer
                text = f'{count}'  # show integer count

                ax.text(j_cm, i_cm, text,
                        ha="center", va="center",
                        color="white" if count > thresh else "black",
                        fontsize=10, fontweight='bold')

        n_classes = len(data['class_names'])

        # Use standardized class-name labels
        short_names = []
        for name in data['class_names']:
            nm = name
            if 'Glioma' in nm:
                short_names.append('Glioma')
            elif 'Meningioma' in nm:
                short_names.append('Menin')
            elif 'NORMAL' in nm or 'Normal' in nm:
                short_names.append('Normal')
            elif 'Neurocytoma' in nm or 'Neurocitoma' in nm:
                short_names.append('Neuro')
            elif 'Schwannoma' in nm:
                short_names.append('Schwan')
            elif 'Outros' in nm or 'Other' in nm:
                short_names.append('Others')
            else:
                short_names.append(nm if len(nm) <= 8 else nm[:8])

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(short_names, rotation=25, ha='right', fontsize=11)
        ax.set_yticklabels(short_names, fontsize=11)
        ax.tick_params(pad=2)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Global labels
    fig.text(0.5, 0.04, 'Predicted Class', ha='center', va='center', fontsize=13, fontweight='bold')
    fig.text(0.03, 0.5, 'True Class', ha='center', va='center', rotation='vertical', fontsize=13, fontweight='bold')
    fig.suptitle('Model Performance Comparison - Confusion Matrices', fontsize=15, fontweight='bold', y=0.96)

    # Save figure
    save_path = os.path.join(save_dir, 'SUPER_COMPACT_confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Super-compact confusion matrices saved: {save_path}")
    print("Key optimization: wspace=0.08 (column spacing significantly reduced)")

    return save_path


def generate_all_visualizations(single_modal_results, clip_results=None, save_dir="results/visualizations"):
    """Generate all four visualization charts (simplified)."""
    print("Generating all visualizations...")
    print("=" * 80)
    
    os.makedirs(save_dir, exist_ok=True)
    visualization_paths = {}
    
    try:
        # 1. ROC curves
        print("\n1/4 Generating ROC curves...")
        roc_path = plot_roc_curves_comparison(single_modal_results, clip_results, save_dir)
        visualization_paths['roc_curves'] = roc_path
        
        # 2. Bubble chart
        print("\n2/4 Generating bubble chart...")
        bubble_path = plot_bubble_chart_compact(single_modal_results, clip_results, save_dir)
        visualization_paths['bubble_chart'] = bubble_path
        
        # 3. Summary metrics chart
        print("\n3/4 Generating summary metrics chart...")
        metrics_path = plot_simple_performance_comparison(single_modal_results, clip_results, save_dir)
        visualization_paths['comprehensive_metrics'] = metrics_path
        
        # 4. Confusion-matrix grid (super-compact)
        print("\n4/4 Generating confusion-matrix grid (super-compact)...")
        cm_path = plot_super_compact_confusion_matrices(single_modal_results, clip_results, save_dir)
        visualization_paths['confusion_matrices'] = cm_path
        
        print("\n" + "=" * 80)
        print("All visualizations generated!")
        print("=" * 80)
        
        # Print output paths
        print("\nGenerated files:")
        for chart_type, path in visualization_paths.items():
            print(f"   {chart_type}: {path}")
        
        return visualization_paths
        
    except Exception as e:
        print(f"ERROR: Error while generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return None

print("Simplified main visualization function defined")
print("Call directly: plot_super_compact_confusion_matrices")