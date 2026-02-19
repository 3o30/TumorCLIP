"""Base single-modal trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from ..core.utils import get_device
from ..models.single_modal.model_factory import ModelFactory


class SingleModalTrainer:
    """Base single-modal trainer."""

    def __init__(self, config):
        """Initialize the trainer."""
        self.config = config
        self.device = get_device()

        print("Trainer initialization complete")
        print(f"   Device: {self.device}")

    def create_model(self, model_name, num_classes):
        """Create a model."""
        return ModelFactory.create_model(model_name, num_classes)

    def create_optimizer(self, model, optimizer_name, lr, weight_decay=1e-4):
        """Create an optimizer."""
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        print(f"Optimizer: {optimizer_name}, learning rate: {lr}")
        return optimizer

    def create_scheduler(self, optimizer, scheduler_type='cosineannealinglr', num_epochs=50, **kwargs):
        """Create a learning-rate scheduler."""
        if scheduler_type.lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type.lower() == 'steplr':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'exponentiallr':
            gamma = kwargs.get('gamma', 0.95)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type.lower() == 'reducelronplateau':
            patience = kwargs.get('patience', 5)
            factor = kwargs.get('factor', 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        else:
            # Default: cosine annealing
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        print(f"LR scheduler: {scheduler_type}")
        return scheduler

    def train_epoch_simple(self, model, data_loader, optimizer, criterion):
        """Train for a single epoch."""
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def evaluate_simplified(self, model, data_loader, criterion):
        """Evaluate the model."""
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc