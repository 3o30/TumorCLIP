"""Enhanced single-modal trainer with early stopping, best-model saving, and precise timing."""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .single_modal_trainer import SingleModalTrainer
from ..core.utils import get_device, count_parameters, save_json
from ..data.data_loaders import DataLoaderFactory


class EnhancedSingleModalTrainer(SingleModalTrainer):
    """Enhanced single-modal trainer."""
    
    def __init__(self, config):
        super().__init__(config)
        self.best_models_dir = Path("results/best_models")
        self.best_models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Best-model save directory: {self.best_models_dir}")
    
    def train_with_early_stopping(self, model_name, train_loader, val_loader, 
                                 optimizer_name, lr, num_epochs, patience=10):
        """
        Train a model while tracking learning curves, with early stopping and best-model saving.
        
        Args:
            model_name: Model name
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer_name: Optimizer name
            lr: Learning rate
            num_epochs: Maximum number of epochs
            patience: Early-stopping patience
        """
        
        # Record training start time
        training_start_time = time.time()
        
        # Create model
        num_classes = len(train_loader.dataset.dataset.classes)
        class_names = train_loader.dataset.dataset.classes
        model = self.create_model(model_name, num_classes)
        model = model.to(self.device)
        
        # Model parameter statistics
        total_params = count_parameters(model)
        print(f"    Model parameters: {total_params:,}")
        
        # Create optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = self.create_optimizer(model, optimizer_name, lr)
        scheduler = self.create_scheduler(optimizer, num_epochs=num_epochs)
        
        # Track metrics
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # Early-stopping state (validation-based)
        best_val_acc = 0.0
        best_val_epoch = 0
        patience_counter = 0
        best_val_model_state = None
        time_to_best_val = 0.0

        # Best-on-train state
        best_train_acc = 0.0
        best_train_epoch = 0
        best_train_model_state = None
        time_to_best_train = 0.0
        
        # Model save paths
        model_safe_name = str(model_name).replace(' ', '_').replace('/', '_')
        optimizer_safe_name = str(optimizer_name).replace(' ', '_').replace('/', '_')
        best_val_model_path = self.best_models_dir / f"{model_safe_name}_{optimizer_safe_name}_lr{lr}_best_val.pth"
        best_train_model_path = self.best_models_dir / f"{model_safe_name}_{optimizer_safe_name}_lr{lr}_best_train.pth"
        
        # Create a compact progress bar (no verbose details)
        pbar = tqdm(range(num_epochs), 
                   desc=f"{model_name} {optimizer_name} lr={lr}", 
                   ncols=120, leave=False, 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] Best:{postfix}')
        
        try:
            for epoch in pbar:
                epoch_start_time = time.time()
                
                # Train one epoch
                train_loss, train_acc = self.train_epoch_simple(
                    model, train_loader, optimizer, criterion)
                
                # Validation evaluation
                val_loss, val_acc = self.evaluate_simplified(model, val_loader, criterion)
                
                # Update learning rate
                if scheduler:
                    scheduler.step()
                
                # Record learning curves
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                epoch_time = time.time() - epoch_start_time
                
                # Check for best validation model
                val_improved = False
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    patience_counter = 0
                    time_to_best_val = time.time() - training_start_time
                    val_improved = True

                    # Save best-validation model weights
                    best_val_model_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'train_loss': train_loss,
                        'lr': lr,
                        'optimizer_name': optimizer_name,
                        'model_name': model_name,
                        'time_to_best': time_to_best_val,
                        'class_names': class_names,
                        'criterion': 'validation_accuracy'
                    }

                # Check for best training model
                train_improved = False
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    best_train_epoch = epoch
                    time_to_best_train = time.time() - training_start_time
                    train_improved = True

                    # Save best-training model weights
                    best_train_model_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'train_loss': train_loss,
                        'lr': lr,
                        'optimizer_name': optimizer_name,
                        'model_name': model_name,
                        'time_to_best': time_to_best_train,
                        'class_names': class_names,
                        'criterion': 'train_accuracy'
                    }

                # Update progress-bar status
                if val_improved and train_improved:
                    pbar.set_postfix_str(f'Val:{best_val_acc:.3f} Train:{best_train_acc:.3f} (BOTH NEW)')
                elif val_improved:
                    pbar.set_postfix_str(f'Val:{best_val_acc:.3f} Train:{best_train_acc:.3f} (VAL NEW)')
                elif train_improved:
                    pbar.set_postfix_str(f'Val:{best_val_acc:.3f} Train:{best_train_acc:.3f} (TRAIN NEW)')
                else:
                    patience_counter += 1
                    pbar.set_postfix_str(f'Val:{best_val_acc:.3f} Train:{best_train_acc:.3f} ({patience_counter}/{patience})')
                
                # Early-stopping check
                if patience_counter >= patience:
                    pbar.set_postfix_str(f'{best_val_acc:.3f} (EARLY STOP)')
                    break
            
            pbar.close()
            
            # Compute total training time
            total_training_time = time.time() - training_start_time
            
            # Compact completion summary
            early_stop_info = "EARLY STOP" if len(train_losses) < num_epochs else "DONE"
            print(f'    {early_stop_info}: Val acc {best_val_acc:.4f} (epoch {best_val_epoch+1}) | Train acc {best_train_acc:.4f} (epoch {best_train_epoch+1})')

        except Exception as e:
            pbar.close()
            raise e

        # Save both best models
        if best_val_model_state:
            torch.save(best_val_model_state, best_val_model_path)
            print(f"    Saved best validation model: {best_val_model_path}")

        if best_train_model_state:
            torch.save(best_train_model_state, best_train_model_path)
            print(f"    Saved best training model: {best_train_model_path}")

        # Load best validation model for final evaluation (primarily for compatibility)
        if best_val_model_state:
            model.load_state_dict(best_val_model_state['model_state_dict'])
            print(f"    Loaded best validation model weights (epoch {best_val_epoch+1})")
        elif best_train_model_state:
            model.load_state_dict(best_train_model_state['model_state_dict'])
            print(f"    Loaded best training model weights (epoch {best_train_epoch+1})")
        
        # Prepare return payload
        result = {
            'final_val_acc': best_val_acc,  # Use best validation accuracy
            'final_test_acc': best_val_acc,  # Compatibility
            'best_val_acc': best_val_acc,   # Explicit best validation accuracy
            'best_train_acc': best_train_acc,  # Best training accuracy
            'best_val_epoch': best_val_epoch,  # Best validation epoch
            'best_train_epoch': best_train_epoch,  # Best training epoch
            'best_epoch': best_val_epoch,       # Best epoch (compatibility)
            'total_epochs': len(train_losses),  # Actual trained epochs
            'early_stopped': len(train_losses) < num_epochs,  # Whether early-stopped
            'time_to_best_val': time_to_best_val,  # Time to reach best val accuracy
            'time_to_best_train': time_to_best_train,  # Time to reach best train accuracy
            'time_to_best': time_to_best_val,   # Time to reach best accuracy (compatibility)
            'total_training_time': total_training_time,  # Total training time
            'best_val_model_path': str(best_val_model_path),  # Best-val model path
            'best_train_model_path': str(best_train_model_path),  # Best-train model path
            'best_model_path': str(best_val_model_path),  # Best model path (compatibility)
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': val_losses,      # Compatibility
            'test_accs': val_accs,          # Compatibility
            'val_losses': val_losses,
            'val_accs': val_accs,
            'trained_model': model,         # Return best model (best-on-val)
            'patience_used': patience,      # Patience used
            'model_params': total_params    # Parameter count
        }
        
        # Cleanup
        del optimizer, scheduler
        torch.cuda.empty_cache()
        
        return result
    
    def run_enhanced_grid_search(self, resume_from_checkpoint=True, patience=10):
        """
        Run an enhanced grid search with early stopping and best-model saving.
        
        Args:
            resume_from_checkpoint: Whether to resume from checkpoints
            patience: Early-stopping patience
        """
        print("Starting enhanced single-modal grid search")
        print(f"Early-stopping patience: {patience} epochs")
        print("=" * 80)
        
        start_time = time.time()
        
        # Only train DenseNet121 as specified
        model_names = ['DenseNet121']
        
        # Hyperparameters
        optimizers = self.config.OPTIMIZERS
        learning_rates = self.config.LEARNING_RATES
        
        results = []
        total_experiments = len(model_names) * len(optimizers) * len(learning_rates)
        experiment_count = 0
        
        for model_name in model_names:
            print(f"\nChecking model: {model_name}")
            print("-" * 60)

            # Get model-specific batch size
            batch_size = self.config.BATCH_SIZE_CONFIG.get(model_name, 32)

            # Check whether a trained model already exists
            model_safe_name = str(model_name).replace(' ', '_').replace('/', '_')
            best_model_path = self.best_models_dir / f"{model_safe_name}_Adam_lr0.0001_best.pth"

            if best_model_path.exists() and resume_from_checkpoint:
                print(f"Found a trained model: {best_model_path}")
                print("Skipping training; using saved weights and logs")

                try:
                    # Create data loaders first to obtain class_names
                    train_loader, val_loader, test_loader, class_names = DataLoaderFactory.create_single_modal_loaders_with_val(
                        self.config.DATA_TRAIN_PATH,
                        self.config.DATA_TEST_PATH,
                        batch_size=batch_size,
                        num_workers=self.config.NUM_WORKERS
                    )

                    # Load checkpoint for basic metadata
                    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=True)

                    # Create a mock training result
                    mock_result = {
                        'final_val_acc': checkpoint.get('val_acc', 0.0),
                        'final_test_acc': checkpoint.get('val_acc', 0.0),
                        'best_val_acc': checkpoint.get('val_acc', 0.0),
                        'best_train_acc': checkpoint.get('train_acc', 0.0),
                        'best_val_epoch': checkpoint.get('epoch', 0),
                        'best_train_epoch': checkpoint.get('epoch', 0),
                        'best_epoch': checkpoint.get('epoch', 0),
                        'total_epochs': checkpoint.get('epoch', 0) + 1,
                        'early_stopped': False,
                        'time_to_best_val': checkpoint.get('time_to_best', 0.0),
                        'time_to_best_train': checkpoint.get('time_to_best', 0.0),
                        'time_to_best': checkpoint.get('time_to_best', 0.0),
                        'total_training_time': checkpoint.get('time_to_best', 0.0),
                        'best_val_model_path': str(best_model_path),
                        'best_train_model_path': str(best_model_path),
                        'best_model_path': str(best_model_path),
                        'train_losses': [],  # Not available
                        'train_accs': [],   # Not available
                        'test_losses': [],   # Not available
                        'test_accs': [],    # Not available
                        'val_losses': [],   # Not available
                        'val_accs': [],    # Not available
                        'trained_model': None,  # Avoid loading full model to save memory
                        'patience_used': patience,
                        'model_params': 0  # Not available
                    }

                    # Build model-level results container
                    model_results = {
                        'name': model_name,
                        'batch_size': batch_size,
                        'class_names': class_names,
                        'optimizer_lr_results': {'Adam_lr0.0001': mock_result},
                        'best_acc': mock_result['best_val_acc'],
                        'best_config': {
                            'optimizer': 'Adam',
                            'lr': 0.0001,
                            'time_to_best': mock_result['time_to_best_val'],
                            'best_epoch': mock_result['best_epoch'],
                            'early_stopped': mock_result['early_stopped']
                        },
                        'model_complexity': self.estimate_model_complexity(model_name),
                        'param_count': mock_result['model_params'],
                        'best_optimizer': 'Adam',
                        'best_lr': 0.0001,
                        'best_training_time': mock_result['time_to_best']
                    }

                    results.append(model_results)
                    print(f"Loaded model results: {model_name} - best acc: {model_results['best_acc']:.4f}")
                    continue

                except Exception as e:
                    print(f"WARNING: Unable to load saved model results: {e}")
                    print("Will retrain the model")

            print(f"Starting training: {model_name}")
            print("-" * 60)
            
            try:
                # Create data loaders
                train_loader, val_loader, test_loader, class_names = DataLoaderFactory.create_single_modal_loaders_with_val(
                    self.config.DATA_TRAIN_PATH, 
                    self.config.DATA_TEST_PATH, 
                    batch_size=batch_size,
                    num_workers=self.config.NUM_WORKERS
                )
                
                model_results = {
                    'name': model_name,
                    'batch_size': batch_size,
                    'class_names': class_names,
                    'optimizer_lr_results': {},
                    'best_acc': 0,
                    'best_config': None,
                    'model_complexity': self.estimate_model_complexity(model_name),
                    'param_count': 0,
                    # Fields required by the plotting system
                    'best_optimizer': None,
                    'best_lr': None,
                    'best_training_time': 0.0
                }
                
                # Hyperparameter grid search
                for optimizer_name in optimizers:
                    for lr in learning_rates:
                        experiment_count += 1
                        try:
                            # Train model
                            trial_result = self.train_with_early_stopping(
                                model_name, train_loader, val_loader,
                                optimizer_name, lr, self.config.NUM_EPOCHS, patience
                            )
                            
                            # Record results
                            config_key = f"{optimizer_name}_lr{lr}"
                            model_results['optimizer_lr_results'][config_key] = trial_result
                            
                            # Update best-on-validation result
                            if trial_result['best_val_acc'] > model_results['best_acc']:
                                # If there was a previous best model, clean it up first
                                if model_results['best_config']:
                                    old_best_key = f"{model_results['best_config']['optimizer']}_lr{model_results['best_config']['lr']}"
                                    if old_best_key in model_results['optimizer_lr_results']:
                                        if 'trained_model' in model_results['optimizer_lr_results'][old_best_key]:
                                            del model_results['optimizer_lr_results'][old_best_key]['trained_model']
                                
                                model_results['best_acc'] = trial_result['best_val_acc']
                                model_results['best_config'] = {
                                    'optimizer': optimizer_name,
                                    'lr': lr,
                                    'time_to_best': trial_result['time_to_best'],
                                    'best_epoch': trial_result['best_epoch'],
                                    'early_stopped': trial_result['early_stopped']
                                }
                                model_results['param_count'] = trial_result['model_params']
                                # Fields required by the plotting system
                                model_results['best_optimizer'] = optimizer_name
                                model_results['best_lr'] = lr
                                model_results['best_training_time'] = trial_result['time_to_best']
                                # Keep the current best model; do not delete it
                            else:
                                # Not the best model; clean it up
                                if 'trained_model' in trial_result:
                                    del trial_result['trained_model']

                            # Track best-on-train result
                            if 'best_train_acc' not in model_results:
                                model_results['best_train_acc'] = 0.0
                                model_results['best_train_config'] = None

                            if trial_result['best_train_acc'] > model_results['best_train_acc']:
                                model_results['best_train_acc'] = trial_result['best_train_acc']
                                model_results['best_train_config'] = {
                                    'optimizer': optimizer_name,
                                    'lr': lr,
                                    'time_to_best': trial_result['time_to_best_train'],
                                    'best_epoch': trial_result['best_train_epoch']
                                }
                            
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"    ERROR: {optimizer_name} lr={lr} failed: {e}")
                            continue
                
                if model_results['best_acc'] > 0:
                    # Save best-on-validation model weights
                    if model_results['best_config']:
                        best_optimizer = model_results['best_config']['optimizer']
                        best_lr = model_results['best_config']['lr']
                        
                        # Build best-on-validation model path
                        model_safe_name = str(model_name).replace(' ', '_').replace('/', '_')
                        best_optimizer_safe = str(best_optimizer).replace(' ', '_').replace('/', '_')
                        best_model_filename = f"{model_safe_name}_{best_optimizer_safe}_lr{best_lr}_best.pth"
                        best_model_path = self.best_models_dir / best_model_filename
                        
                        # Get weights from the best hyperparameter run
                        best_key = f"{best_optimizer}_lr{best_lr}"
                        if best_key in model_results['optimizer_lr_results']:
                            best_result = model_results['optimizer_lr_results'][best_key]
                            if 'trained_model' in best_result:
                                # Save best-on-validation model weights
                                best_model_state = {
                                    'epoch': best_result['best_epoch'],
                                    'model_state_dict': best_result['trained_model'].state_dict(),
                                    'val_acc': best_result['best_val_acc'],
                                    'val_loss': best_result['val_losses'][best_result['best_epoch']],
                                    'train_acc': best_result['train_accs'][best_result['best_epoch']],
                                    'train_loss': best_result['train_losses'][best_result['best_epoch']],
                                    'lr': best_lr,
                                    'optimizer_name': best_optimizer,
                                    'model_name': model_name,
                                    'time_to_best': best_result['time_to_best'],
                                    'class_names': class_names,
                                    'criterion': 'validation_accuracy'
                                }
                                
                                torch.save(best_model_state, best_model_path)
                                print(f"    Saved best validation weights: {best_model_filename}")
                                
                                # Cleanup after saving
                                del best_result['trained_model']
                                torch.cuda.empty_cache()

                    # Save best-on-train model weights
                    if model_results.get('best_train_config'):
                        train_optimizer = model_results['best_train_config']['optimizer']
                        train_lr = model_results['best_train_config']['lr']

                        # Build best-on-train model path
                        train_optimizer_safe = str(train_optimizer).replace(' ', '_').replace('/', '_')
                        train_best_model_filename = f"{model_safe_name}_{train_optimizer_safe}_lr{train_lr}_best_train.pth"
                        train_best_model_path = self.best_models_dir / train_best_model_filename

                        # Get best-on-train weights from the corresponding run
                        train_key = f"{train_optimizer}_lr{train_lr}"
                        if train_key in model_results['optimizer_lr_results']:
                            train_result = model_results['optimizer_lr_results'][train_key]

                            # Best-on-train weights should already be saved by train_with_early_stopping.
                            # Here we only ensure the file exists; if not, create it.
                            if not train_best_model_path.exists():
                                try:
                                    # Recreate model and load best-on-train weights
                                    train_model = self.create_model(model_name, len(class_names))
                                    train_model = train_model.to(self.device)

                                    # Restore best-on-train state from the run result
                                    train_model.load_state_dict(train_result['trained_model'].state_dict())

                                    # Build best-on-train checkpoint payload
                                    train_best_model_state = {
                                        'epoch': model_results['best_train_config']['best_epoch'],
                                        'model_state_dict': train_model.state_dict(),
                                        'val_acc': train_result['val_accs'][model_results['best_train_config']['best_epoch']],
                                        'val_loss': train_result['val_losses'][model_results['best_train_config']['best_epoch']],
                                        'train_acc': model_results['best_train_acc'],
                                        'train_loss': train_result['train_losses'][model_results['best_train_config']['best_epoch']],
                                        'lr': train_lr,
                                        'optimizer_name': train_optimizer,
                                        'model_name': model_name,
                                        'time_to_best': model_results['best_train_config']['time_to_best'],
                                        'class_names': class_names,
                                        'criterion': 'train_accuracy'
                                    }

                                    torch.save(train_best_model_state, train_best_model_path)
                                    print(f"    Saved best training weights: {train_best_model_filename} (Train Acc: {model_results['best_train_acc']:.4f})")

                                except Exception as e:
                                    print(f"    WARNING: Failed to save best training weights: {e}")
                            else:
                                print(f"    Best training weights already exist: {train_best_model_filename} (Train Acc: {model_results['best_train_acc']:.4f})")
                    
                    results.append(model_results)
                    best_config = model_results['best_config']
                    print(f"\n{model_name} best: {model_results['best_acc']:.4f} "
                          f"({best_config['optimizer']} lr={best_config['lr']}) "
                          f"time {best_config['time_to_best']/60:.1f} min")
                else:
                    print(f"\nERROR: {model_name}: all experiments failed")
                
                # Cleanup data loaders
                del train_loader, val_loader, test_loader
                torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"ERROR: {model_name} training failed: {e}")
                continue
        
        # Save results
        if results:
            save_path = "results/training_logs/enhanced_single_modal_results.json"
            save_json(results, save_path)
            print(f"\nSaved enhanced results to: {save_path}")
            
            # Print summary
            print("\n" + "=" * 80)
            print("Enhanced experiment summary")
            print("=" * 80)
            
            successful_results = [r for r in results if r['best_acc'] > 0]
            if successful_results:
                print(f"\nCompleted experiments ({len(successful_results)} models):")
                
                for result in successful_results:
                    best_config = result['best_config']
                    early_stop_info = "EARLY STOP" if best_config['early_stopped'] else "DONE"
                    print(f"  {result['name']}: {result['best_acc']:.4f} "
                          f"({best_config['optimizer']} lr={best_config['lr']}, {early_stop_info})")
                
                best_result = max(successful_results, key=lambda x: x['best_acc'])
                print(f"\nBest: {best_result['name']} - {best_result['best_acc']:.4f}")
                
                total_time_to_best = sum(r['best_config']['time_to_best'] for r in successful_results)
                total_time = time.time() - start_time
                print(f"Total time: {total_time/60:.1f} min | Effective training: {total_time_to_best/60:.1f} min")
        
        return results
    
    def estimate_model_complexity(self, model_name):
        """Estimate model complexity (GFLOPs)."""
        complexity_map = {
            'EfficientNet_b0': 0.39,
            'ResNet50': 4.1,
            'MobileNetV3_large': 0.22,
            'ViT': 17.6,
            'DenseNet121': 2.9,
            'DeiT': 17.6,
            'Swin Transformer': 15.4,
            'MambaOut': 1.2
        }
        return complexity_map.get(model_name, 5.0)  # Default
