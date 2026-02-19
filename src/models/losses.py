"""Custom loss functions.

This module provides custom loss functions for brain tumor classification:
- FocalLoss: addresses class imbalance
- LabelSmoothingCrossEntropy: helps prevent overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss - addresses class imbalance.
    
    Focal Loss down-weights easy examples so the model focuses on harder ones.
    Formula: FL(pt) = -α(1-pt)^γ * log(pt)
    
    Args:
        alpha (float): Balance factor, default 1.0
        gamma (float): Focusing parameter, default 2.0. Larger gamma down-weights easy samples more.
        reduction (str): Reduction mode: 'mean', 'sum', or 'none'
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Model logits, shape [batch_size, num_classes]
            targets (Tensor): Ground-truth labels, shape [batch_size]
            
        Returns:
            Tensor: Loss value
        """
        # Cross-entropy loss per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # pt (estimated probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label-smoothing cross-entropy loss - helps prevent overfitting.
    
    Label smoothing softens hard targets to reduce overconfidence and improve generalization.
    It reduces the target probability for the true class from 1.0 to (1-ε) and distributes ε over the others.
    
    Args:
        smoothing (float): Smoothing parameter ε, default 0.05. Range [0, 1)
                          0 means no smoothing (same as standard cross-entropy).
    
    Reference:
        Szegedy et al. "Rethinking the Inception Architecture" (CVPR 2016)
    """
    
    def __init__(self, smoothing=0.05):
        super().__init__()
        assert 0 <= smoothing < 1, "smoothing参数应在[0, 1)范围内"
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Model logits, shape [batch_size, num_classes]
            targets (Tensor): Ground-truth labels, shape [batch_size]
            
        Returns:
            Tensor: Loss value
        """
        # Log-softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Negative log-likelihood for the true class
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        # Mean loss across classes (smoothing term)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combine both terms
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()


# ==================== Helper functions ====================

def get_loss_function(loss_type='ce', **kwargs):
    """Get a loss function.
    
    Args:
        loss_type (str): Loss type:
            - 'ce': standard cross-entropy
            - 'focal': Focal Loss
            - 'label_smooth': label-smoothing cross-entropy
        **kwargs: Extra arguments for the loss constructor
        
    Returns:
        nn.Module: Loss instance
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'label_smooth':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


# ==================== Usage example ====================

if __name__ == "__main__":
    # Test code
    import torch
    
    # Mock data
    batch_size = 4
    num_classes = 6
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print("=" * 60)
    print("Loss function test")
    print("=" * 60)
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss_focal = focal_loss(logits, targets)
    print(f"Focal Loss (γ=2.0): {loss_focal.item():.4f}")
    
    # Test label smoothing
    label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_smooth = label_smooth_loss(logits, targets)
    print(f"Label Smoothing Loss (ε=0.1): {loss_smooth.item():.4f}")
    
    # Test standard cross-entropy
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, targets)
    print(f"Cross Entropy Loss: {loss_ce.item():.4f}")
    
    print("=" * 60)
    print("All loss-function tests passed")

