"""Plotting utilities."""

import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Font settings
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12


class PaperPalette:
    """Paper-style color palettes (two schemes) with global application and a heatmap cmap.

    SCHEME_1:  #9CBF8C, #83B1DA, #F99058, #E8DCD2, #C1BCDE, #8ECEC8, #AED2E4, #3580B8
    SCHEME_2:  #073068, #206FB6, #6BADD7, #C5DAEE, #FDDFD0, #FC9171, #EE3B2A, #A60E16
    """

    SCHEME_1 = ['#9CBF8C', '#83B1DA', '#F99058', '#E8DCD2',
                '#C1BCDE', '#8ECEC8', '#AED2E4', '#3580B8']
    SCHEME_2 = ['#073068', '#206FB6', '#6BADD7', '#C5DAEE',
                '#FDDFD0', '#FC9171', '#EE3B2A', '#A60E16']

    _ACTIVE = SCHEME_2  # default to scheme 2 (a more standard blue-red sequence)

    @classmethod
    def set_active(cls, scheme: str = 'scheme2'):
        scheme = (scheme or '').lower()
        if scheme in ('scheme1', 'palette1', 'set1'):
            cls._ACTIVE = cls.SCHEME_1
        else:
            cls._ACTIVE = cls.SCHEME_2
        # Set the global color cycle
        matplotlib.rcParams['axes.prop_cycle'] = cycler('color', cls._ACTIVE)

    @classmethod
    def get_active(cls):
        return list(cls._ACTIVE)

    @classmethod
    def get_heatmap_cmap(cls):
        """Create a continuous heatmap colormap from the active palette.
        - For blues: use a light-to-dark linear gradient
        - For green/orange: fall back to SCHEME_1 light blue (#AED2E4) to dark blue (#3580B8)
        """
        if cls._ACTIVE is cls.SCHEME_2:
            c_lo, c_hi = '#C5DAEE', '#073068'
        else:
            c_lo, c_hi = '#AED2E4', '#3580B8'
        return LinearSegmentedColormap.from_list('paper_heatmap', [c_lo, c_hi], N=256)


# Initialize global palette
PaperPalette.set_active('scheme2')



class LearningCurvePlotter:
    """Learning-curve plotter."""
    
    def __init__(self, save_dir="results/plots", dpi=300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        
        # Create learning-curves subdirectory
        self.lc_dir = os.path.join("results", "plots", "learning_curves")
        os.makedirs(self.lc_dir, exist_ok=True)
        
        palette = PaperPalette.get_active()
        # Use one palette for both optimizer groups with different offsets to avoid identical colors
        self.sgd_colors = palette
        self.adam_colors = palette[2:] + palette[:2]
    
    def plot_model_curves(self, model_results, model_name):
        """Plot learning curves for a single model."""
        series = []
        for k, v in model_results.items():
            if not v or 'train_losses' not in v or len(v['train_losses']) == 0:
                continue
            if v.get('final_test_acc', 0) <= 0:
                continue
            series.append((k, v))
        
        if not series:
            print(f"WARNING: {model_name}: no valid learning curves; skipping")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1.set_title(f'{model_name} - Training Loss', fontweight='bold')
        ax2.set_title(f'{model_name} - Test Loss', fontweight='bold')
        ax3.set_title(f'{model_name} - Training Accuracy', fontweight='bold')
        ax4.set_title(f'{model_name} - Test Accuracy', fontweight='bold')
        
        sgd_count = 0
        adam_count = 0
        
        for k, v in series:
            if '__lr=' in k:
                optimizer_name = k.split('__lr=')[0]
                lr_str = k.split('__lr=')[1]
            else:
                optimizer_name = 'SGD'
                lr_str = str(k)
            
            try:
                lr_float = float(lr_str)
                if lr_float >= 1:
                    lr_display = f"{lr_float:.0f}"
                elif lr_float >= 0.01:
                    lr_display = f"{lr_float:.2f}".rstrip('0').rstrip('.')
                else:
                    lr_display = f"{lr_float:.3f}".rstrip('0').rstrip('.')
            except:
                lr_display = lr_str
            
            if optimizer_name.upper() == 'SGD':
                color = self.sgd_colors[sgd_count % len(self.sgd_colors)]
                linestyle = '-'
                label = f'SGD, LR={lr_display}'
                sgd_count += 1
            else:
                color = self.adam_colors[adam_count % len(self.adam_colors)]
                linestyle = '--'
                label = f'Adam, LR={lr_display}'
                adam_count += 1
            
            epochs = list(range(1, len(v['train_losses']) + 1))
            
            train_losses = [float(x) for x in v['train_losses']]
            test_losses = [float(x) for x in v['test_losses']]
            train_accs = [float(x) for x in v['train_accs']]
            test_accs = [float(x) for x in v['test_accs']]
            
            ax1.plot(epochs, train_losses, color=color, linestyle=linestyle,
                    label=label, linewidth=2.5, alpha=0.9)
            ax2.plot(epochs, test_losses, color=color, linestyle=linestyle,
                    label=label, linewidth=2.5, alpha=0.9)
            ax3.plot(epochs, train_accs, color=color, linestyle=linestyle,
                    label=label, linewidth=2.5, alpha=0.9)
            ax4.plot(epochs, test_accs, color=color, linestyle=linestyle,
                    label=label, linewidth=2.5, alpha=0.9)
        
        for ax in (ax1, ax2, ax3, ax4):
            ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(self.lc_dir, f'{safe_model_name}_learning_curves.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"{model_name} learning curves saved: {save_path}")
    
    def plot_clip_curves(self, clip_result):
        """Plot learning curves for a CLIP model."""
        model_name = clip_result['name']
        
        if 'train_losses' not in clip_result or len(clip_result['train_losses']) == 0:
            print(f"WARNING: {model_name}: no learning-curve data; skipping")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = list(range(1, len(clip_result['train_losses']) + 1))
        
        # Training Loss
        ax1.plot(epochs, clip_result['train_losses'], 'b-', linewidth=2.5, label='Training Loss')
        ax1.set_title(f'{model_name} - Training Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation Loss
        ax2.plot(epochs, clip_result['val_losses'], 'r-', linewidth=2.5, label='Validation Loss')
        ax2.set_title(f'{model_name} - Validation Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training Accuracy
        ax3.plot(epochs, clip_result['train_accs'], 'g-', linewidth=2.5, label='Training Accuracy')
        ax3.set_title(f'{model_name} - Training Accuracy', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Validation Accuracy
        ax4.plot(epochs, clip_result['val_accs'], 'orange', linewidth=2.5, label='Validation Accuracy')
        ax4.set_title(f'{model_name} - Validation Accuracy', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(self.lc_dir, f'{safe_model_name}_learning_curves.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"{model_name} CLIP learning curves saved: {save_path}")


class ComparisonPlotter:
    """Model-comparison plotter."""
    
    def __init__(self, save_dir="results/plots", dpi=300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        
        # Create analysis subdirectory
        self.analysis_dir = os.path.join("results", "plots", "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def plot_model_comparison(self, results):
        """Plot model comparison charts."""
        valid_results = [r for r in results if r.get('best_acc', 0) > 0]
        
        if not valid_results:
            print("WARNING: No valid result data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        names = [r['name'] for r in valid_results]
        accuracies = [r['best_acc'] * 100.0 for r in valid_results]
        params = [r['param_count'] / 1e6 for r in valid_results]
        complexities = [r['model_complexity'] for r in valid_results]
        
        palette = PaperPalette.get_active()
        colors = [palette[i % len(palette)] for i in range(len(valid_results))]

        # Dynamically widen the canvas to reduce x-tick overlap
        try:
            fig.set_size_inches(max(15, 1.0 * max(6, len(names))), 10)
        except Exception:
            pass
        
        # 1. Accuracy comparison
        bars1 = ax1.bar(names, accuracies, color=colors)
        ax1.set_title('Best Accuracy (%)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=35, labelsize=9)
        ax1.set_ylabel('Accuracy (%)')
        
        if len(names) <= 12:
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
        
        # 2. Parameter count comparison
        bars2 = ax2.bar(names, params, color=colors)
        ax2.set_title('Parameters (millions)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=35, labelsize=9)
        ax2.set_ylabel('Parameters (M)')
        
        if len(names) <= 12:
            for i, v in enumerate(params):
                ax2.text(i, v + max(params)*0.02 if params else 0.02, f'{v:.1f}M', ha='center', fontweight='bold', fontsize=9)
        
        # 3. Complexity comparison
        bars3 = ax3.bar(names, complexities, color=colors)
        ax3.set_title('Model Complexity (GFLOPs)', fontweight='bold')
        ax3.tick_params(axis='x', rotation=35, labelsize=9)
        ax3.set_ylabel('Complexity (GFLOPs)')
        
        if len(names) <= 12:
            for i, v in enumerate(complexities):
                if complexities and max(complexities) > 0:
                    ax3.text(i, v + max(complexities)*0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)
        
        # 4. Efficiency scatter plot
        scatter = ax4.scatter(params, accuracies, s=100, c=complexities,
                            cmap=PaperPalette.get_heatmap_cmap(), alpha=0.7, edgecolors='w', linewidth=0.5)
        ax4.set_xlabel('Parameters (millions)')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Efficiency (top-left is best)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Complexity (GFLOPs)', rotation=270, labelpad=20)
        
        # Annotate model names: reduce labels for many models and alternate offsets
        offsets = [(6,6), (-6,6), (6,-6), (-6,-6)]
        if len(names) > 18:
            # Only annotate top 18 by accuracy
            order = np.argsort([-a for a in accuracies])[:18]
            for k, i in enumerate(order):
                dx, dy = offsets[k % len(offsets)]
                ax4.annotate(names[i], (params[i], accuracies[i]), 
                            xytext=(dx, dy), textcoords='offset points', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        else:
            for i, name in enumerate(names):
                dx, dy = offsets[i % len(offsets)]
                ax4.annotate(name, (params[i], accuracies[i]), 
                            xytext=(dx, dy), textcoords='offset points', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        
        plt.tight_layout()
        
        save_path = os.path.join(self.analysis_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved: {save_path}")
    
    def plot_hyperparameter_analysis(self, results):
        """Plot hyperparameter analysis charts (high-resolution), up to 4 subplots per row."""
        valid_results = [r for r in results if r.get('best_acc', 0) > 0 
                        and 'optimizer_lr_results' in r]
        
        if not valid_results:
            print("WARNING: No valid hyperparameter result data")
            return
        
        n_models = len(valid_results)
        # Compute rows/cols (max 4 per row)
        n_cols = min(4, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols  # ceil
        
        # Dynamically adjust figure size
        max_trials = 1
        for vr in valid_results:
            try:
                max_trials = max(max_trials, len(vr.get('optimizer_lr_results', {})))
            except Exception:
                pass
        
        # Figure size
        fig_width = min(20, 4.5 * n_cols)  # 4.5 inches per column (max 20)
        fig_height = 5 * n_rows  # 5 inches per row
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Normalize axes shape for 1D/2D cases
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        # When n_rows > 1 and n_cols > 1, axes is already a 2D array
        
        for i, result in enumerate(valid_results):
            row = i // n_cols
            col = i % n_cols
            # Handle numpy-array indexing
            if n_rows > 1 and n_cols > 1:
                ax = axes[row, col]  # numpy arrays use [row, col]
            else:
                ax = axes[row][col]  # lists use [row][col]
            
            optimizer_lr_results = result.get('optimizer_lr_results', {})
            trial_entries = []
            
            for key, val in optimizer_lr_results.items():
                if '__lr=' in key:
                    optimizer_name = key.split('__lr=')[0]
                    lr_str = key.split('__lr=')[1]
                else:
                    optimizer_name = 'SGD'
                    lr_str = str(key)
                
                try:
                    lr_float = float(lr_str)
                except:
                    lr_float = float('inf')
                
                acc_pct = float(val.get('final_test_acc', 0.0)) * 100.0
                trial_entries.append((optimizer_name, lr_str, lr_float, acc_pct))
            
            trial_entries.sort(key=lambda t: (t[0].upper(), -t[2]))
            
            accuracies = [t[3] for t in trial_entries]
            # Use single-line labels to avoid wrapping/overlap
            labels = [f"{t[0]}-{t[1]}" for t in trial_entries]
            
            # Increase spacing by reducing bar width
            bar_width = 0.6  # narrower bars -> more spacing
            x_positions = range(len(accuracies))
            bars = ax.bar(x_positions, accuracies, width=bar_width, color='steelblue')
            
            if accuracies:
                best_idx = accuracies.index(max(accuracies))
                bars[best_idx].set_color('orange')
            
            # Larger title font
            ax.set_title(f'{result["name"]}\nBest: {max(accuracies):.1f}%' if accuracies else result["name"], 
                        fontsize=18, pad=15, fontweight='bold')
            ax.set_xticks(x_positions)
            # X tick labels: single line, rotated, bold
            ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right', va='top', fontweight='bold')
            # Tick label sizes
            ax.tick_params(axis='x', labelsize=11, rotation=45)
            ax.tick_params(axis='y', labelsize=14)
            # Margins to make room for rotated labels
            ax.margins(x=0.1, y=0.1)
            # Larger Y-axis label
            ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
            
            # Smaller value labels above bars
            for j, v in enumerate(accuracies):
                ax.text(j, v+1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
        
        # Hide unused subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            # Handle numpy-array indexing
            if n_rows > 1 and n_cols > 1:
                axes[row, col].set_visible(False)
            else:
                axes[row][col].set_visible(False)
        
        # Adjust subplot spacing to make room for rotated labels
        plt.subplots_adjust(wspace=0.5, hspace=0.6, bottom=0.25, top=0.88, left=0.12, right=0.95)
        plt.tight_layout(rect=[0, 0.08, 1, 0.92])
        
        save_path = os.path.join(self.analysis_dir, 'hyperparameter_analysis.png')
        # Increase DPI to improve clarity
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"High-resolution hyperparameter analysis saved: {save_path}")
    
    def plot_bubble_chart(self, results, clip_best_result=None):
        """Plot a bubble chart (improved version) using log scaling; optionally include CLIP."""
        valid_results = [r for r in results if r.get('best_acc', 0) > 0]
        # Optionally include CLIP as an extra point
        if clip_best_result and clip_best_result.get('final_test_acc', 0) > 0:
            valid_results = valid_results + [{
                'name': clip_best_result.get('name', 'CLIP (Multimodal)'),
                'param_count': clip_best_result.get('model_params', 80_000_000),
                'best_acc': clip_best_result.get('final_test_acc', 0.0),
                'model_complexity': 15.0
            }]
        
        if not valid_results:
            print("WARNING: No valid result data")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        x = [r['param_count'] / 1e6 for r in valid_results]
        y = [r['best_acc'] * 100.0 for r in valid_results]
        complexities = [r['model_complexity'] for r in valid_results]
        
        # Debug output
        print("Model complexity debug info:")
        for i, r in enumerate(valid_results):
            print(f"  {r['name']}: complexity={r['model_complexity']:.2f} GFLOPs")
        
        # Improved bubble-size computation (log scale)
        min_complexity = min(complexities) if complexities else 1
        max_complexity = max(complexities) if complexities else 1
        
        if max_complexity > min_complexity and min_complexity > 0:
            # Use log scaling to better show differences
            log_complexities = np.log10(complexities)
            log_min = np.log10(min_complexity)
            log_max = np.log10(max_complexity)
            
            if log_max > log_min:
                # Normalize to the 100-800 range
                normalized_sizes = []
                for log_c in log_complexities:
                    norm_size = 100 + (log_c - log_min) / (log_max - log_min) * 700
                    normalized_sizes.append(norm_size)
                sizes = normalized_sizes
            else:
                sizes = [400] * len(complexities)
        else:
            sizes = [400] * len(complexities)
        
        print(f"Bubble size range: {min(sizes):.1f} - {max(sizes):.1f}")
        
        # Choose colors by model type
        colors = []
        for result in valid_results:
            model_name = result['name'].lower()
            if 'efficientnet' in model_name or 'mobilenet' in model_name:
                colors.append('#1f77b4')  # blue - lightweight models
            elif 'resnet' in model_name or 'densenet' in model_name:
                colors.append('#ff7f0e')  # orange - traditional CNN
            elif 'vit' in model_name or 'transformer' in model_name or 'deit' in model_name:
                colors.append('#2ca02c')  # green - Transformer
            elif 'mamba' in model_name:
                colors.append('#d62728')  # red - Mamba
            else:
                colors.append('#9467bd')  # purple - other
        
        # Scatter plot
        scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, 
                           edgecolors='black', linewidth=1.5)
        
        # Do not show text labels (keep chart clean)
        # for i, result in enumerate(valid_results):
        #     ax.annotate(result['name'], (x[i], y[i]), 
        #                xytext=(8, 8), textcoords='offset points', 
        #                fontsize=12, fontweight='bold',
        #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Axes
        ax.set_xlabel('Parameters (millions)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Analysis\n(Bubble Size = Computational Complexity)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Legend (place outside to the right with larger fonts)
        legend_elements = [
            plt.scatter([], [], s=100, c='#1f77b4', label='Lightweight (EfficientNet/MobileNet)', alpha=0.7),
            plt.scatter([], [], s=100, c='#ff7f0e', label='Traditional CNN (ResNet/DenseNet)', alpha=0.7),
            plt.scatter([], [], s=100, c='#2ca02c', label='Transformer (ViT/DeiT)', alpha=0.7),
            plt.scatter([], [], s=100, c='#d62728', label='Mamba', alpha=0.7)
        ]
        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),  # outside, right side
            borderaxespad=0.0,
            fontsize=12,
            frameon=True,
            title='Model Types',
            title_fontsize=12
        )
        
        # Complexity summary box
        complexity_info = f'Complexity Range: {min_complexity:.2f} - {max_complexity:.2f} GFLOPs\n'
        complexity_info += f'Bubble Size: Logarithmic Scale\n'
        complexity_info += f'Models: {len(valid_results)}'
        
        ax.text(0.02, 0.98, complexity_info, 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.analysis_dir, 'bubble_chart.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Improved bubble chart saved: {save_path}")
        
        # Print complexity-difference analysis
        if max_complexity > min_complexity:
            ratio = max_complexity / min_complexity
            print("Complexity difference analysis:")
            print(f"  Max/min ratio: {ratio:.2f}x")
            print(f"  Range: {max_complexity - min_complexity:.2f} GFLOPs")
            if ratio > 3.0:
                print("  Complexity difference is large; bubble sizes should differ clearly")
            elif ratio > 2.0:
                print("  Complexity difference is moderate; bubble-size differences should be observable")
            else:
                print("  Complexity difference is small; bubble-size differences may be subtle")
    
    def plot_clip_comparison(self, clip_results):
        """Plot CLIP model comparison charts."""
        if not clip_results:
            print("WARNING: No CLIP results to compare")
            return
        
        valid_results = [r for r in clip_results if r.get('final_test_acc', 0) > 0]
        
        if not valid_results:
            print("WARNING: No valid CLIP results to compare")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy Comparison
        model_names = [r['name'] for r in valid_results]
        accuracies = [r['final_test_acc'] * 100 for r in valid_results]
        
        bars = ax1.bar(model_names, accuracies, color='orange', alpha=0.8)
        ax1.set_title('CLIP Models - Test Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Training Time Comparison
        training_times = [r['total_time'] / 60 for r in valid_results]  # Convert to minutes
        
        bars = ax2.bar(model_names, training_times, color='lightblue', alpha=0.8)
        ax2.set_title('CLIP Models - Training Time Comparison', fontweight='bold')
        ax2.set_ylabel('Training Time (minutes)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars, training_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time_val:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # 3. Best Validation Accuracy Comparison
        best_val_accs = [r['best_val_acc'] * 100 for r in valid_results]
        
        bars = ax3.bar(model_names, best_val_accs, color='lightgreen', alpha=0.8)
        ax3.set_title('CLIP Models - Best Validation Accuracy', fontweight='bold')
        ax3.set_ylabel('Best Validation Accuracy (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, best_val_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Model Parameter Count Comparison
        param_counts = [r['model_params'] / 1e6 for r in valid_results]  # Convert to millions
        
        bars = ax4.bar(model_names, param_counts, color='lightcoral', alpha=0.8)
        ax4.set_title('CLIP Models - Parameter Count', fontweight='bold')
        ax4.set_ylabel('Parameters (M)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, params in zip(bars, param_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{params:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.analysis_dir, 'clip_models_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"CLIP comparison plot saved: {save_path}")
    
    def plot_training_time_comparison(self, single_modal_results, clip_results):
        """Plot training-time comparison: single-modal vs multi-modal."""
        print("Plotting training-time comparison...")
        
        # Prepare data
        single_modal_data = []
        for result in single_modal_results:
            if result.get('best_training_time', 0) > 0:
                single_modal_data.append({
                    'name': result['name'],
                    'time': result['best_training_time'] / 60,  # convert to minutes
                    'acc': result['best_acc'] * 100,
                    'type': 'Single-Modal'
                })
        
        clip_data = []
        for result in clip_results:
            if result.get('total_time', 0) > 0:
                clip_data.append({
                    'name': result['name'],
                    'time': result['total_time'] / 60,  # convert to minutes
                    'acc': result['final_test_acc'] * 100,
                    'type': 'Multi-Modal (CLIP)'
                })
        
        if not single_modal_data and not clip_data:
            print("WARNING: No training-time data")
            return
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Training-time bar chart
        all_data = single_modal_data + clip_data
        if all_data:
            names = [d['name'] for d in all_data]
            times = [d['time'] for d in all_data]
            palette = PaperPalette.get_active()
            c_single = palette[1 % len(palette)]
            c_multi = palette[2 % len(palette)]
            colors = [c_single if d['type'] == 'Single-Modal' else c_multi for d in all_data]
            
            bars = ax1.bar(names, times, color=colors, alpha=0.8)
            ax1.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Training Time (minutes)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            
            # Add value labels
            for bar, time_val in zip(bars, times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time_val:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # 2. Accuracy vs training-time scatter plot
        if all_data:
            single_times = [d['time'] for d in single_modal_data]
            single_accs = [d['acc'] for d in single_modal_data]
            clip_times = [d['time'] for d in clip_data]
            clip_accs = [d['acc'] for d in clip_data]
            
            palette = PaperPalette.get_active()
            c_single = palette[1 % len(palette)]
            c_multi = palette[2 % len(palette)]
            ax2.scatter(single_times, single_accs, c=c_single, s=100, alpha=0.7, 
                       label='Single-Modal', edgecolors='black')
            ax2.scatter(clip_times, clip_accs, c=c_multi, s=100, alpha=0.7, 
                       label='Multi-Modal (CLIP)', edgecolors='black')
            
            ax2.set_xlabel('Training Time (minutes)', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Accuracy vs Training Time', fontweight='bold', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Average training-time comparison
        if single_modal_data and clip_data:
            avg_single_time = np.mean([d['time'] for d in single_modal_data])
            avg_clip_time = np.mean([d['time'] for d in clip_data])
            
            categories = ['Single-Modal\nAverage', 'Multi-Modal\n(CLIP) Average']
            avg_times = [avg_single_time, avg_clip_time]
            palette = PaperPalette.get_active()
            colors = [palette[1 % len(palette)], palette[2 % len(palette)]]
            
            bars = ax3.bar(categories, avg_times, color=colors, alpha=0.8)
            ax3.set_title('Average Training Time Comparison', fontweight='bold', fontsize=14)
            ax3.set_ylabel('Average Training Time (minutes)', fontsize=12)
            
            # Add value labels
            for bar, time_val in zip(bars, avg_times):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time_val:.1f}m', ha='center', va='bottom', fontweight='bold')
            
            # Compute time-saving percentage
            time_saving = ((avg_single_time - avg_clip_time) / avg_single_time) * 100
            ax3.text(0.5, 0.95, f'CLIP saves {time_saving:.1f}% time', 
                    transform=ax3.transAxes, ha='center', va='top', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontweight='bold', fontsize=12)
        
        # 4. Efficiency comparison (accuracy/time)
        if all_data:
            efficiencies = []
            model_names = []
            for d in all_data:
                efficiency = d['acc'] / d['time']  # accuracy per minute
                efficiencies.append(efficiency)
                model_names.append(d['name'])
            
            palette = PaperPalette.get_active()
            c_single = palette[1 % len(palette)]
            c_multi = palette[2 % len(palette)]
            bars = ax4.bar(model_names, efficiencies,
                          color=[c_multi if 'clip' in name.lower() else c_single for name in model_names],
                          alpha=0.8)
            ax4.set_title('Training Efficiency (Accuracy/Time)', fontweight='bold', fontsize=14)
            ax4.set_ylabel('Efficiency (Accuracy % / minute)', fontsize=12)
            ax4.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Add value labels
            for bar, eff in zip(bars, efficiencies):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.analysis_dir, 'training_time_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Training-time comparison saved: {save_path}")
        
        # Print summary stats
        if single_modal_data and clip_data:
            print("\nTraining-time stats:")
            print(f"  Single-modal average training time: {avg_single_time:.1f} min")
            print(f"  Multi-modal (CLIP) average training time: {avg_clip_time:.1f} min")
            print(f"  CLIP time saving: {time_saving:.1f}%")

    def plot_best_training_time_across_models(self, single_modal_results, best_clip_result=None):
        """Compare overall training time of each single-modal model's best hyperparameter run; optionally include the best CLIP run.

        - Single-modal: result['best_training_time'] (seconds) → minutes
        - CLIP: best_clip_result['total_time'] (seconds) → minutes
        """
        print("Plotting overall training time for best hyperparameter run...")

        entries = []
        for r in single_modal_results:
            best_time = float(r.get('best_training_time', 0.0))
            best_acc = float(r.get('best_acc', 0.0))
            name = r.get('name', 'Model')
            if best_time > 0 and best_acc > 0:
                entries.append((name, best_time / 60.0, best_acc * 100.0))

        if best_clip_result and isinstance(best_clip_result, dict):
            clip_time = float(best_clip_result.get('total_time', 0.0))
            clip_acc = float(best_clip_result.get('final_test_acc', 0.0))
            clip_name = best_clip_result.get('name', 'CLIP (Multimodal)')
            if clip_time > 0 and clip_acc > 0:
                entries.append((clip_name, clip_time / 60.0, clip_acc * 100.0))

        if not entries:
            print("WARNING: No data for time comparison (best-run time missing)")
            return

        # Sort by time (ascending) to highlight faster models
        entries.sort(key=lambda t: t[1])

        names = [e[0] for e in entries]
        times = [e[1] for e in entries]
        accs = [e[2] for e in entries]

        fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(names)), 5))
        palette = PaperPalette.get_active()
        colors = [palette[2 % len(palette)] if 'clip' in n.lower() else palette[1 % len(palette)] for n in names]
        bars = ax.bar(names, times, color=colors, alpha=0.85)
        ax.set_title('Overall Training Time of Best Hyperparameter Group', fontweight='bold')
        ax.set_ylabel('Time (minutes)')
        ax.tick_params(axis='x', rotation=35, labelsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate each bar with time and best accuracy
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{times[i]:.1f}m\nAcc {accs[i]:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.analysis_dir, 'best_training_time_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Best-run training-time comparison saved: {save_path}")

    def plot_performance_comparison(self, results, clip_best_result=None):
        """Plot performance comparison (Accuracy, Sensitivity, Specificity, AUC, F1) with improved readability; optionally include CLIP."""
        print("Plotting performance comparison (5 metrics)...")
        valid_results = []
        for result in results:
            name = result.get('name')
            best_key = None
            best_acc = -1.0
            cm_data = None
            for key, lr_result in result.get('optimizer_lr_results', {}).items():
                acc = float(lr_result.get('final_test_acc', 0.0))
                if acc > best_acc and 'confusion_matrix_data' in lr_result:
                    best_acc = acc
                    best_key = key
                    cm_data = lr_result['confusion_matrix_data']
            if not cm_data or not cm_data.get('y_true'):
                continue
            y_true = np.array(cm_data['y_true'])
            y_pred = np.array(cm_data['y_pred'])
            y_probs = np.array(cm_data.get('y_probs', []), dtype=float) if cm_data.get('y_probs') else None
            class_names = cm_data.get('class_names', [])
            num_classes = len(class_names) if class_names else int(max(y_true)+1)

            # Accuracy
            accuracy = float((y_true == y_pred).mean())

            # Sensitivity (Recall) macro；Specificity macro: per-class TN/(TN+FP)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            with np.errstate(divide='ignore', invalid='ignore'):
                tp = np.diag(cm).astype(float)
                fn = cm.sum(axis=1) - tp
                fp = cm.sum(axis=0) - tp
                tn = cm.sum() - (tp + fp + fn)
                sensitivity_per_class = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
                specificity_per_class = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
            sensitivity = float(np.nanmean(sensitivity_per_class))
            specificity = float(np.nanmean(specificity_per_class))

            # F1 macro
            f1_per_class = []
            for i in range(num_classes):
                prec_denom = tp[i] + fp[i]
                rec_denom = tp[i] + fn[i]
                precision_i = (tp[i] / prec_denom) if prec_denom > 0 else 0.0
                recall_i = (tp[i] / rec_denom) if rec_denom > 0 else 0.0
                denom = precision_i + recall_i
                f1_i = (2 * precision_i * recall_i / denom) if denom > 0 else 0.0
                f1_per_class.append(f1_i)
            f1 = float(np.mean(f1_per_class)) if f1_per_class else 0.0

            # AUC (macro-ovr). Requires probabilities; set to NaN if unavailable.
            auc_macro = float('nan')
            if y_probs is not None and y_probs.size == y_true.shape[0] * num_classes:
                try:
                    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                    auc_macro = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
                except Exception:
                    pass

            valid_results.append({
                'name': name,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'auc_macro': auc_macro,
                'f1': f1,
            })

        # Append CLIP
        if clip_best_result and clip_best_result.get('final_test_acc', 0) > 0:
            try:
                y_true = np.array(clip_best_result.get('test_labels', []))
                y_pred = np.array(clip_best_result.get('test_predictions', []))
                y_probs = np.array(clip_best_result.get('test_probabilities', []), dtype=float) if clip_best_result.get('test_probabilities') else None
                class_names = clip_best_result.get('confusion_matrix_data', {}).get('class_names', [])
                num_classes = len(class_names) if class_names else (int(y_true.max())+1 if y_true.size>0 else 0)
                accuracy = float((y_true == y_pred).mean()) if y_true.size>0 else 0.0
                # Reuse the computation below
                # Specificity/Sensitivity
                if y_true.size>0:
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        tp = np.diag(cm).astype(float)
                        fn = cm.sum(axis=1) - tp
                        fp = cm.sum(axis=0) - tp
                        tn = cm.sum() - (tp + fp + fn)
                        sensitivity_per_class = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
                        specificity_per_class = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
                    sensitivity = float(np.nanmean(sensitivity_per_class))
                    specificity = float(np.nanmean(specificity_per_class))
                else:
                    sensitivity = 0.0
                    specificity = 0.0
                # F1
                f1 = 0.0
                if y_true.size>0:
                    f1_per_class = []
                    for i in range(num_classes):
                        prec_denom = tp[i] + fp[i]
                        rec_denom = tp[i] + fn[i]
                        precision_i = (tp[i] / prec_denom) if prec_denom > 0 else 0.0
                        recall_i = (tp[i] / rec_denom) if rec_denom > 0 else 0.0
                        denom = precision_i + recall_i
                        f1_i = (2 * precision_i * recall_i / denom) if denom > 0 else 0.0
                        f1_per_class.append(f1_i)
                    f1 = float(np.mean(f1_per_class)) if f1_per_class else 0.0
                # AUC
                auc_macro = float('nan')
                if y_probs is not None and y_probs.size == y_true.shape[0] * num_classes and num_classes>1:
                    try:
                        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                        auc_macro = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
                    except Exception:
                        pass
                valid_results.append({
                    'name': clip_best_result.get('name', 'CLIP (Multimodal)'),
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'auc_macro': auc_macro,
                    'f1': f1,
                })
            except Exception:
                pass

        if not valid_results:
            print("WARNING: No data for performance comparison")
            return

        # Sort by Accuracy descending; display with AUC first
        metrics = ['auc_macro', 'accuracy', 'sensitivity', 'specificity', 'f1']
        order = np.argsort([-vr['accuracy'] for vr in valid_results])
        labels = [valid_results[i]['name'] for i in order]
        values_matrix = [[valid_results[i][m] if not (isinstance(valid_results[i][m], float) and np.isnan(valid_results[i][m])) else 0.0 for i in order] for m in metrics]

        x = np.arange(len(labels))
        width = 0.15
        fig_width = max(10, 1.8*len(labels))
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        colors = PaperPalette.get_active()[:5]
        label_map = {
            'auc_macro': 'AUC',
            'accuracy': 'Accuracy',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'f1': 'F1',
        }
        bars = []
        for i, m in enumerate(metrics):
            bars.append(
                ax.bar(
                    x + i*width - (len(metrics)-1)*width/2,
                    values_matrix[i],
                    width,
                    label=label_map.get(m, m),
                    color=colors[i]
                )
            )

        # Dynamic Y-axis: zoom near the top to highlight differences
        all_vals = np.array(values_matrix).flatten()
        finite_vals = all_vals[np.isfinite(all_vals)]
        y_min = 0.0
        if len(finite_vals) > 0:
            # Use the smallest of per-metric maxima as a reference; leave 0.1 margin below (clamped to [0, 1))
            per_max = [max(col) if len(col) else 1.0 for col in values_matrix]
            ref = min(per_max) if per_max else 1.0
            y_min = max(0.0, min(0.95, ref - 0.1))
        ax.set_ylim(y_min, 1.0)

        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison (macro where applicable)', pad=40, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.legend(
            ncol=len(metrics),
            loc='upper center',
            bbox_to_anchor=(0.5, 1.30),
            fontsize=12,
            frameon=True,
            title='Metrics',
            title_fontsize=12
        )
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate Accuracy bars with absolute value and delta to best
        acc_idx = metrics.index('accuracy')
        best_acc = max(values_matrix[acc_idx]) if values_matrix[acc_idx] else 0.0
        for j, bar in enumerate(bars[acc_idx]):
            v = values_matrix[acc_idx][j]
            delta = best_acc - v
            ax.text(
                bar.get_x() + bar.get_width()/2,
                v + 0.003,
                f"{v:.3f}\nΔ{delta:.3f}",
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )

        plt.tight_layout(rect=[0, 0, 1, 0.85])
        save_path = os.path.join(self.analysis_dir, 'performance_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison saved: {save_path}")

    def plot_average_accuracy(self, results):
        """Plot average accuracy across hyperparameter runs for each model."""
        print("Plotting average accuracy across hyperparameters...")
        names = []
        avgs = []
        for result in results:
            name = result.get('name')
            accs = [float(v.get('final_test_acc', 0.0)) for v in result.get('optimizer_lr_results', {}).values()]
            if accs:
                names.append(name)
                avgs.append(float(np.mean(accs)))
        if not names:
            print("WARNING: No data for average accuracy")
            return
        fig, ax = plt.subplots(figsize=(max(8, 1.5*len(names)), 5))
        bars = ax.bar(names, [a*100 for a in avgs], color='teal', alpha=0.8)
        ax.set_title('Average Accuracy Across Hyperparameters', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(avgs):
            ax.text(i, v*100 + 1, f"{v*100:.1f}%", ha='center', fontweight='bold')
        ax.set_ylim(0, 100)
        plt.tight_layout()
        save_path = os.path.join(self.analysis_dir, 'average_accuracy_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Average-accuracy comparison saved: {save_path}")

    def plot_roc_comparison(self, results, clip_best_result=None):
        """Plot ROC comparison (micro-average for multi-class) with improved readability; optionally include CLIP."""
        print("Plotting ROC comparison...")
        series = []
        for result in results:
            name = result.get('name')
            best_key = None
            best_acc = -1.0
            cm_data = None
            for key, lr_result in result.get('optimizer_lr_results', {}).items():
                acc = float(lr_result.get('final_test_acc', 0.0))
                if acc > best_acc and 'confusion_matrix_data' in lr_result:
                    best_acc = acc
                    best_key = key
                    cm_data = lr_result['confusion_matrix_data']
            if not cm_data:
                continue
            y_true = np.array(cm_data.get('y_true', []))
            y_probs = np.array(cm_data.get('y_probs', []), dtype=float) if cm_data.get('y_probs') else None
            class_names = cm_data.get('class_names', [])
            if y_true.size == 0 or y_probs is None or y_probs.size == 0:
                continue
            num_classes = len(class_names) if class_names else y_probs.shape[1]
            series.append((name, y_true, y_probs, num_classes))

        # Append CLIP
        if clip_best_result:
            try:
                y_true = np.array(clip_best_result.get('test_labels', []))
                y_probs = np.array(clip_best_result.get('test_probabilities', []), dtype=float) if clip_best_result.get('test_probabilities') else None
                class_names = clip_best_result.get('confusion_matrix_data', {}).get('class_names', [])
                if y_true.size>0 and y_probs is not None and y_probs.size>0:
                    num_classes = len(class_names) if class_names else y_probs.shape[1]
                    series.append((clip_best_result.get('name', 'CLIP (Multimodal)'), y_true, y_probs, num_classes))
            except Exception:
                pass

        if not series:
            print("WARNING: No probability data; cannot plot ROC")
            return

        def tpr_at_fpr(fpr, tpr, target=0.05):
            try:
                return float(np.interp(target, fpr, tpr))
            except Exception:
                return float('nan')

        # Plot 1: full view
        fig_main, ax_main = plt.subplots(figsize=(8, 6))
        for name, y_true, y_probs, num_classes in series:
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
                auc_val = roc_auc_score(y_true, y_probs[:, 1])
                tpr5 = tpr_at_fpr(fpr, tpr, 0.05)
                ax_main.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f}, TPR@5%={tpr5:.3f})")
            else:
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
                auc_val = roc_auc_score(y_true_bin, y_probs, average='micro', multi_class='ovr')
                tpr5 = tpr_at_fpr(fpr, tpr, 0.05)
                ax_main.plot(fpr, tpr, label=f"{name} micro (AUC={auc_val:.3f}, TPR@5%={tpr5:.3f})")
        ax_main.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_main.set_xlabel('False Positive Rate')
        ax_main.set_ylabel('True Positive Rate')
        ax_main.set_title('ROC Comparison')
        ax_main.legend(loc='lower right')
        ax_main.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path_main = os.path.join(self.analysis_dir, 'roc_comparison.png')
        plt.savefig(save_path_main, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig_main)
        print(f"ROC comparison saved: {save_path_main}")

        # Plot 2: zoomed view (saved separately, not embedded)
        fig_zoom, ax_zoom = plt.subplots(figsize=(8, 6))
        for name, y_true, y_probs, num_classes in series:
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            else:
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
            ax_zoom.plot(fpr, tpr, label=name)
        ax_zoom.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax_zoom.set_xlim(0.0, 0.2)
        ax_zoom.set_ylim(0.8, 1.0)
        ax_zoom.set_xlabel('False Positive Rate (Zoomed)')
        ax_zoom.set_ylabel('True Positive Rate (Zoomed)')
        ax_zoom.set_title('ROC Comparison (Zoomed: FPR≤0.2, TPR≥0.8)')
        ax_zoom.axvline(0.05, color='gray', linestyle=':', alpha=0.6)
        ax_zoom.legend(loc='lower right')
        ax_zoom.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path_zoom = os.path.join(self.analysis_dir, 'roc_comparison_zoom.png')
        plt.savefig(save_path_zoom, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig_zoom)
        print(f"ROC zoomed view saved: {save_path_zoom}")

        
class ConfusionMatrixPlotter:
    """Confusion-matrix plotter."""
    
    def __init__(self, save_dir="results/plots", dpi=300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        
        # Create confusion-matrix subdirectory (save under results/plots/confusion_matrices)
        self.cm_dir = os.path.join("results", "plots", "confusion_matrices")
        os.makedirs(self.cm_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, model_name, 
                            optimizer_name=None, lr=None):
        """Plot a single confusion matrix."""
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix (counts)
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=PaperPalette.get_heatmap_cmap())
        if optimizer_name and lr is not None:
            ax1.set_title(f'{model_name} [{optimizer_name}, lr={lr}] - Confusion Matrix (Counts)')
        else:
            ax1.set_title(f'{model_name} - Confusion Matrix (Counts)')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
        
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1)
        
        # Confusion matrix (percentages)
        im2 = ax2.imshow(cm_percent, interpolation='nearest', cmap=PaperPalette.get_heatmap_cmap())
        if optimizer_name and lr is not None:
            ax2.set_title(f'{model_name} [{optimizer_name}, lr={lr}] - Confusion Matrix (Percentages)')
        else:
            ax2.set_title(f'{model_name} - Confusion Matrix (Percentages)')
        
        # Add text annotations
        thresh = cm_percent.max() / 2.
        for i in range(cm_percent.shape[0]):
            for j in range(cm_percent.shape[1]):
                ax2.text(j, i, f'{cm_percent[i, j]:.1f}%',
                        ha="center", va="center",
                        color="white" if cm_percent[i, j] > thresh else "black",
                        fontweight='bold')
        
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_yticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        # Save figure
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        if optimizer_name and lr:
            filename = f'{safe_model_name}_{optimizer_name}_lr{lr}_confusion_matrix.png'
        else:
            filename = f'{safe_model_name}_confusion_matrix.png'
        
        save_path = os.path.join(self.cm_dir, filename)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"{model_name} best confusion matrix saved: {save_path}")
        
        return cm, cm_percent
    
    def plot_model_confusion_matrices(self, model_result):
        """Plot the confusion matrix for a model's best hyperparameter configuration."""
        model_name = model_result['name']
        optimizer_lr_results = model_result.get('optimizer_lr_results', {})
        
        print(f"Generating best confusion matrix for {model_name}...")
        
        # Find the best hyperparameter configuration
        best_key = None
        best_acc = 0.0
        
        for key, result in optimizer_lr_results.items():
            if result.get('final_test_acc', 0) > best_acc:
                best_acc = result['final_test_acc']
                best_key = key
        
        if not best_key or 'confusion_matrix_data' not in optimizer_lr_results[best_key]:
            print(f"WARNING: {model_name}: no valid confusion-matrix data")
            return None
        
        # Get data for the best result
        best_result = optimizer_lr_results[best_key]
        cm_data = best_result['confusion_matrix_data']
        y_true = np.array(cm_data['y_true'])
        y_pred = np.array(cm_data['y_pred'])
        class_names = cm_data['class_names']
        
        # Parse optimizer and learning rate
        if '__lr=' in best_key:
            optimizer_name = best_key.split('__lr=')[0]
            lr_str = best_key.split('__lr=')[1]
        else:
            optimizer_name = 'SGD'
            lr_str = str(best_key)
        
        # Plot best confusion matrix
        cm, cm_percent = self.plot_confusion_matrix(
            y_true, y_pred, class_names, model_name, optimizer_name, lr_str
        )
        
        return {
            'cm': cm,
            'cm_percent': cm_percent,
            'accuracy': best_acc,
            'optimizer': optimizer_name,
            'learning_rate': lr_str
        }
    
    def plot_best_models_comparison(self, results, clip_best_result=None, clip_class_names=None):
        """Plot confusion-matrix comparison across best results of all models; optionally include CLIP best result."""
        valid_results = [r for r in results if r.get('best_acc', 0) > 0]
        
        if not valid_results:
            print("WARNING: No valid model results for confusion-matrix comparison")
            return
        
        # Determine class names (from the best-accuracy configuration per model)
        class_names = None
        for result in valid_results:
            optimizer_lr_results = result.get('optimizer_lr_results', {})
            if not optimizer_lr_results:
                continue
            # Find best key
            best_key = None
            best_acc = -1.0
            for key, lr_result in optimizer_lr_results.items():
                acc = float(lr_result.get('final_test_acc', 0.0))
                if acc > best_acc:
                    best_acc = acc
                    best_key = key
            if best_key and 'confusion_matrix_data' in optimizer_lr_results[best_key]:
                cm_data = optimizer_lr_results[best_key]['confusion_matrix_data']
                if cm_data and cm_data.get('class_names'):
                    class_names = cm_data['class_names']
                    break
        
        if not class_names:
            print("WARNING: Class names not found; cannot plot confusion-matrix comparison")
            return
        
        # Compute number of subplots (optionally include CLIP)
        n_models = len(valid_results)
        include_clip = False
        clip_cm_pair = None
        clip_labels = None
        if clip_best_result and (
            ('confusion_matrix_data' in clip_best_result and clip_best_result['confusion_matrix_data'].get('y_true'))
            or ('test_labels' in clip_best_result and 'test_predictions' in clip_best_result)
        ):
            try:
                if 'confusion_matrix_data' in clip_best_result and clip_best_result['confusion_matrix_data']:
                    cdata = clip_best_result['confusion_matrix_data']
                    y_true_c = np.array(cdata.get('y_true', []))
                    y_pred_c = np.array(cdata.get('y_pred', []))
                    clip_labels = cdata.get('class_names', clip_class_names)
                else:
                    y_true_c = np.array(clip_best_result.get('test_labels', []))
                    y_pred_c = np.array(clip_best_result.get('test_predictions', []))
                    clip_labels = clip_class_names
                if y_true_c.size > 0 and y_pred_c.size > 0:
                    clip_cm_pair = (y_true_c, y_pred_c)
                    include_clip = True
                    n_models += 1
            except Exception:
                include_clip = False
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(valid_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            model_name = result['name']
            
            # Use confusion-matrix data from the highest-accuracy configuration
            best_cm_data = None
            optimizer_lr_results = result.get('optimizer_lr_results', {})
            if optimizer_lr_results:
                best_key = None
                best_acc = -1.0
                for key, lr_result in optimizer_lr_results.items():
                    acc = float(lr_result.get('final_test_acc', 0.0))
                    if acc > best_acc:
                        best_acc = acc
                        best_key = key
                if best_key and 'confusion_matrix_data' in optimizer_lr_results[best_key]:
                    best_cm_data = optimizer_lr_results[best_key]['confusion_matrix_data']
            
            if best_cm_data:
                y_true = np.array(best_cm_data['y_true'])
                y_pred = np.array(best_cm_data['y_pred'])
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                
                # Plot confusion matrix
                im = ax.imshow(cm_percent, interpolation='nearest', cmap=PaperPalette.get_heatmap_cmap())
                ax.set_title(f'{model_name}\nAccuracy: {float(result.get("best_acc", 0.0)):.3f}')
                
                # Add text annotations
                thresh = cm_percent.max() / 2.
                for i_cm in range(cm_percent.shape[0]):
                    for j_cm in range(cm_percent.shape[1]):
                        ax.text(j_cm, i_cm, f'{cm_percent[i_cm, j_cm]:.1f}%',
                                ha="center", va="center",
                                color="white" if cm_percent[i_cm, j_cm] > thresh else "black",
                                fontsize=8, fontweight='bold')
                
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(class_names, fontsize=8)
            else:
                ax.text(0.5, 0.5, f'{model_name}\nNo confusion matrix data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(model_name)
        
        # Append CLIP best-result confusion matrix
        if include_clip and len(valid_results) < len(axes):
            ax = axes[len(valid_results)]
            model_name = clip_best_result.get('name', 'CLIP (Multimodal)')
            y_true_c, y_pred_c = clip_cm_pair
            cm = confusion_matrix(y_true_c, y_pred_c)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            im = ax.imshow(cm_percent, interpolation='nearest', cmap=PaperPalette.get_heatmap_cmap())
            # Use provided test accuracy if available; otherwise compute it
            if 'final_test_acc' in clip_best_result:
                acc_val = float(clip_best_result['final_test_acc'])
            else:
                acc_val = float((y_true_c == y_pred_c).mean())
            ax.set_title(f'{model_name}\nAccuracy: {acc_val:.3f}')
            thresh = cm_percent.max() / 2.
            for i_cm in range(cm_percent.shape[0]):
                for j_cm in range(cm_percent.shape[1]):
                    ax.text(j_cm, i_cm, f'{cm_percent[i_cm, j_cm]:.1f}%',
                            ha="center", va="center",
                            color="white" if cm_percent[i_cm, j_cm] > thresh else "black",
                            fontsize=8, fontweight='bold')
            # Axis labels
            cls = clip_labels
            if cls is None:
                # Fall back to numeric labels
                num_classes = cm.shape[0]
                cls = [f'class_{k}' for k in range(num_classes)]
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_xticks(range(len(cls)))
            ax.set_yticks(range(len(cls)))
            ax.set_xticklabels(cls, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(cls, fontsize=8)
        
        # Hide unused subplots
        for i in range(len(valid_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.cm_dir, 'best_models_confusion_matrices_comparison.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Best-model confusion-matrix comparison saved: {save_path}")
    
    def plot_clip_confusion_matrix(self, clip_result, class_names):
        """Plot the confusion matrix for a CLIP model."""
        if 'test_predictions' not in clip_result or 'test_labels' not in clip_result:
            print("WARNING: CLIP result is missing predictions/labels")
            return None
        
        y_pred = clip_result['test_predictions']
        y_true = clip_result['test_labels']
        model_name = clip_result['name']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix (counts)
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=PaperPalette.get_heatmap_cmap())
        ax1.set_title(f'{model_name} - Confusion Matrix (Counts)')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
        
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1)
        
        # Confusion matrix (percentages)
        im2 = ax2.imshow(cm_percent, interpolation='nearest', cmap=PaperPalette.get_heatmap_cmap())
        ax2.set_title(f'{model_name} - Confusion Matrix (Percentages)')
        
        # Add text annotations
        thresh = cm_percent.max() / 2.
        for i in range(cm_percent.shape[0]):
            for j in range(cm_percent.shape[1]):
                ax2.text(j, i, f'{cm_percent[i, j]:.1f}%',
                        ha="center", va="center",
                        color="white" if cm_percent[i, j] > thresh else "black",
                        fontweight='bold')
        
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_yticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        # Save figure
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        filename = f'{safe_model_name}_confusion_matrix.png'
        save_path = os.path.join(self.cm_dir, filename)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"{model_name} confusion matrix saved: {save_path}")
        
        return cm, cm_percent
        
        if not valid_results:
            print("WARNING: No valid result data")
            return
        
        # Find the best configuration for each model
        best_cms = []
        model_names = []
        
        for result in valid_results:
            model_name = result['name']
            optimizer_lr_results = result.get('optimizer_lr_results', {})
            
            best_acc = 0
            best_cm_data = None
            
            for key, res in optimizer_lr_results.items():
                if res['final_test_acc'] > best_acc and 'confusion_matrix_data' in res:
                    best_acc = res['final_test_acc']
                    best_cm_data = res['confusion_matrix_data']
            
            if best_cm_data:
                y_true = np.array(best_cm_data['y_true'])
                y_pred = np.array(best_cm_data['y_pred'])
                class_names = best_cm_data['class_names']
                
                cm = confusion_matrix(y_true, y_pred)
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                
                best_cms.append(cm_percent)
                model_names.append(f"{model_name}\n(Acc: {best_acc:.3f})")
        
        if not best_cms:
            print("WARNING: No confusion-matrix data found")
            return
        
        # Create comparison figure
        n_models = len(best_cms)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (cm_percent, model_name) in enumerate(zip(best_cms, model_names)):
            row = i // cols
            col = i % cols
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col] if cols > 1 else axes[i]
            
            im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')
            ax.set_title(model_name, fontsize=10)
            
            # Add text annotations
            thresh = cm_percent.max() / 2.
            for ii in range(cm_percent.shape[0]):
                for jj in range(cm_percent.shape[1]):
                    ax.text(jj, ii, f'{cm_percent[ii, jj]:.1f}',
                           ha="center", va="center",
                           color="white" if cm_percent[ii, jj] > thresh else "black",
                           fontsize=8)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
        
        # Hide unused subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.cm_dir, 'best_models_confusion_matrices.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Best-model confusion-matrix comparison saved: {save_path}")


    