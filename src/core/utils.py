import torch
import numpy as np
import random
import os
import json
from datetime import datetime


def set_seed(seed=42):
    """Set random seeds to ensure experiment reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return the available computation device (CUDA if available, otherwise CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_numpy_types(obj):
    """Recursively convert NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_json(data, filepath):
    """Save data to a JSON file with automatic NumPy type conversion."""
    ensure_dir(os.path.dirname(filepath) if os.path.dirname(filepath) else '.')

    # Convert NumPy types before serialization
    converted_data = convert_numpy_types(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    """Load and return data from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_parameters(model):
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format time duration (in seconds) into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def setup_model_cache():
    """Initialize model cache configuration if available."""
    try:
        from config.model_cache_config import ModelCacheConfig
        ModelCacheConfig.setup_cache_environment()
    except ImportError:
        print("Model cache configuration not found. Using default system path.")


class TrainingLogger:
    """Training logger for structured experiment tracking and result recording."""

    def __init__(self, log_dir='logs', experiment_name=None):
        self.log_dir = log_dir
        ensure_dir(log_dir)

        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f'{experiment_name}.log')
        self.json_log_file = os.path.join(log_dir, f'{experiment_name}_detailed.json')

        # Initialize structured logging dictionary
        self.logs = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'models': {}
        }

        # Create initial log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"Experiment Name: {experiment_name}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 80 + "\n\n")

    def log_config(self, config_dict):
        """Record experiment configuration settings."""
        self.logs['config'] = config_dict

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("Configuration\n")
            for key, value in config_dict.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    def log_message(self, message, level='INFO'):
        """Log a general message with timestamp and severity level."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)

        print(message)

    def log_model_start(self, model_name, config):
        """Log the start of model training with configuration details."""
        if model_name not in self.logs['models']:
            self.logs['models'][model_name] = {
                'experiments': [],
                'best_result': None
            }

        self.log_message(f"\n{'='*80}")
        self.log_message(f"Starting training for model: {model_name}")
        self.log_message(f"Configuration: {config}")
        self.log_message(f"{'='*80}\n")

    def log_epoch(self, model_name, exp_id, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log per-epoch training and validation metrics."""
        message = (
            f"  Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        self.log_message(message)

    def log_model_result(self, model_name, optimizer, lr, result):
        """Log final training results for a model configuration."""
        exp_key = f"{optimizer}_lr{lr}"

        if model_name not in self.logs['models']:
            self.logs['models'][model_name] = {'experiments': [], 'best_result': None}

        self.logs['models'][model_name]['experiments'].append({
            'optimizer': optimizer,
            'lr': lr,
            'best_val_acc': result.get('best_val_acc', 0),
            'best_epoch': result.get('best_epoch', 0),
            'time_to_best': result.get('time_to_best', 0),
            'early_stopped': result.get('early_stopped', False),
            'model_path': result.get('best_model_path', '')
        })

        if self.logs['models'][model_name]['best_result'] is None or \
           result.get('best_val_acc', 0) > self.logs['models'][model_name]['best_result'].get('best_val_acc', 0):
            self.logs['models'][model_name]['best_result'] = {
                'optimizer': optimizer,
                'lr': lr,
                'best_val_acc': result.get('best_val_acc', 0),
                'best_epoch': result.get('best_epoch', 0),
                'model_path': result.get('best_model_path', '')
            }

        self.log_message(f"\n {model_name} - {optimizer} lr={lr} training completed")
        self.log_message(f"   Best validation accuracy: {result.get('best_val_acc', 0):.4f} (Epoch {result.get('best_epoch', 0)+1})")
        self.log_message(f"   Time to best: {result.get('time_to_best', 0)/60:.2f} minutes")
        self.log_message(f"   Model checkpoint path: {result.get('best_model_path', '')}")

        if result.get('early_stopped', False):
            self.log_message(f"   Status: Early stopped")
        else:
            self.log_message(f"   Status: Training completed")

    def log_experiment_summary(self, total_time):
        """Log final experiment summary including best-performing models."""
        self.logs['end_time'] = datetime.now().isoformat()
        self.logs['total_time_seconds'] = total_time

        self.log_message(f"\n\n{'='*80}")
        self.log_message("Experiment Summary")
        self.log_message(f"{'='*80}")
        self.log_message(f"Total runtime: {format_time(total_time)}")

        best_models = []
        for model_name, model_data in self.logs['models'].items():
            if model_data['best_result']:
                best_models.append({
                    'name': model_name,
                    'acc': model_data['best_result']['best_val_acc'],
                    'config': f"{model_data['best_result']['optimizer']}_lr{model_data['best_result']['lr']}",
                    'path': model_data['best_result']['model_path']
                })

        if best_models:
            best_models.sort(key=lambda x: x['acc'], reverse=True)
            self.log_message(f"\n Best results per model:")
            for i, model in enumerate(best_models, 1):
                self.log_message(f"  [{i}] {model['name']}: {model['acc']:.4f} ({model['config']})")
                self.log_message(f"      Model path: {model['path']}")

            self.log_message(f"\n Overall best model: {best_models[0]['name']} - {best_models[0]['acc']:.4f}")

        save_json(self.logs, self.json_log_file)
        self.log_message(f"\n Detailed logs saved to:")
        self.log_message(f"   Text log: {self.log_file}")
        self.log_message(f"   JSON log: {self.json_log_file}")

    def save(self):
        """Manually save structured log data to JSON."""
        save_json(self.logs, self.json_log_file)