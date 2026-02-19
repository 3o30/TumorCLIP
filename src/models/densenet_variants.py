"""DenseNet variants.

This module provides DenseNet variants for different tasks:
- DenseNetClassifier: for classification
- DenseNetEncoder: for feature extraction (CLIP fusion)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .losses import FocalLoss, LabelSmoothingCrossEntropy


class DenseNetClassifier(nn.Module):
    """DenseNet for classification.
    
    A DenseNet121-based classifier that supports:
    - loading pretrained weights
    - layer-wise learning rates (different LRs for backbone vs head)
    - custom losses (Focal Loss, label smoothing)
    
    Args:
        num_classes (int): Number of classes, default 6
        backbone_lr (float): Backbone learning rate, default 5e-5
        head_lr (float): Head learning rate, default 1e-3
        focal_gamma (float): Gamma for Focal Loss, default 2.0
        label_smoothing (float): Smoothing for label smoothing, default 0.05
    """
    
    def __init__(self, num_classes=6, backbone_lr=5e-5, head_lr=1e-3, 
                 focal_gamma=2.0, label_smoothing=0.05):
        super().__init__()
        
        # Use the exact same architecture as the original training
        self.backbone = models.densenet121(pretrained=False, num_classes=num_classes)
        
        # Loss functions
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # Learning-rate configuration
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        
        print("DenseNet classifier created")
        print(f"   Classes: {num_classes}")
        print(f"   Backbone learning rate: {backbone_lr}")
        print(f"   Head learning rate: {head_lr}")
    
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path (str): Checkpoint file path
            
        Returns:
            bool: Whether loading succeeded
        """
        print(f"Loading pretrained weights: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print("   WARNING: Weights file not found; using random initialization")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights into the backbone
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            
            print("   Weights loaded")
            if missing_keys:
                print(f"   WARNING: Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   WARNING: Unexpected keys: {len(unexpected_keys)}")
            
            return True
        except Exception as e:
            print(f"   ERROR: Failed to load weights: {str(e)}")
            return False
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (Tensor): Input images, shape [batch_size, 3, 224, 224]
            
        Returns:
            Tensor: Classification logits, shape [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def get_optimizer_params(self):
        """Get parameter groups for layer-wise learning rates.
        
        Returns:
            list: Parameter group dicts with params, lr and name
        """
        backbone_params = []
        head_params = []
        
        # Split backbone vs head parameters
        for name, param in self.named_parameters():
            if 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': self.backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': self.head_lr, 'name': 'head'}
        ]
    
    def compute_loss(self, logits, targets, loss_type='focal'):
        """Compute loss.
        
        Args:
            logits (Tensor): Model outputs
            targets (Tensor): Ground-truth labels
            loss_type (str): Loss type: 'focal', 'label_smooth', or 'ce'
            
        Returns:
            Tensor: Loss value
        """
        if loss_type == 'focal':
            return self.focal_loss(logits, targets)
        elif loss_type == 'label_smooth':
            return self.label_smooth_loss(logits, targets)
        else:
            return F.cross_entropy(logits, targets)


class DenseNetEncoder(nn.Module):
    """DenseNet encoder for feature extraction (for CLIP fusion).
    
    Adds a projection layer on top of DenseNet to:
    - extract image features for CLIP alignment
    - preserve classification capability
    - support multi-task learning
    
    Args:
        embed_dim (int): CLIP embedding dimension, default 512
        dropout (float): Dropout rate, default 0.1
        num_classes (int): Number of classes, default 6
    """
    
    def __init__(self, embed_dim=512, dropout=0.1, num_classes=6):
        super().__init__()
        
        # Use the exact same architecture as the original training
        self.backbone = models.densenet121(pretrained=False, num_classes=num_classes)
        backbone_dim = self.backbone.classifier.in_features
        
        # Keep the original classifier for CE loss
        self.original_classifier = self.backbone.classifier
        
        # Feature projection head for CLIP space
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # New classification head (for combined prediction)
        self.new_classification_head = nn.Linear(embed_dim, num_classes)
        
        self.embed_dim = embed_dim
        self.backbone_dim = backbone_dim
        
        print("DenseNet encoder created")
        print(f"   Backbone dim: {backbone_dim} → CLIP dim: {embed_dim}")
    
    def forward_features(self, x):
        """Extract features for CLIP alignment.
        
        Args:
            x (Tensor): Input images
            
        Returns:
            Tensor: CLIP features, shape [batch_size, embed_dim]
        """
        features = self.backbone.features(x)  # [B, 1024, 7, 7]
        features = F.adaptive_avg_pool2d(features, (1, 1))  # [B, 1024, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 1024]
        
        # Project to CLIP space
        projected_features = self.feature_projection(features)  # [B, 512]
        return projected_features
    
    def forward(self, x, return_features=False, use_original_classifier=False):
        """Forward pass.
        
        Args:
            x (Tensor): Input images
            return_features (bool): Whether to return both features and logits
            use_original_classifier (bool): Whether to use the original classifier
            
        Returns:
            If use_original_classifier=True: original classification logits
            If return_features=True: (clip_features, new_logits)
            Otherwise: clip_features
        """
        if use_original_classifier:
            # Use the original full model (including the original classifier)
            return self.backbone(x)
        else:
            # Extract features for CLIP
            clip_features = self.forward_features(x)
            
            if return_features:
                # Return CLIP features and new head logits
                new_logits = self.new_classification_head(clip_features)
                return clip_features, new_logits
            else:
                return clip_features
    
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path (str): Checkpoint file path
            
        Returns:
            bool: Whether loading succeeded
        """
        print(f"Loading pretrained weights: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print("   WARNING: Weights file not found")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load directly into backbone (including classifier)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            
            print("   Weights loaded")
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)}")
            
            return len(missing_keys) == 0 and len(unexpected_keys) == 0
        except Exception as e:
            print(f"   ERROR: Failed to load weights: {str(e)}")
            return False


# ==================== Helper functions ====================

def create_densenet_classifier(num_classes=6, pretrained_path=None, **kwargs):
    """Convenience function to create a DenseNet classifier.
    
    Args:
        num_classes (int): Number of classes
        pretrained_path (str): Optional pretrained weights path
        **kwargs: Extra arguments passed to DenseNetClassifier
        
    Returns:
        DenseNetClassifier: Model instance
    """
    model = DenseNetClassifier(num_classes=num_classes, **kwargs)
    if pretrained_path:
        model.load_pretrained_weights(pretrained_path)
    return model


def create_densenet_encoder(embed_dim=512, num_classes=6, pretrained_path=None, **kwargs):
    """Convenience function to create a DenseNet encoder.
    
    Args:
        embed_dim (int): CLIP embedding dimension
        num_classes (int): Number of classes
        pretrained_path (str): Optional pretrained weights path
        **kwargs: Extra arguments passed to DenseNetEncoder
        
    Returns:
        DenseNetEncoder: Model instance
    """
    model = DenseNetEncoder(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
    if pretrained_path:
        model.load_pretrained_weights(pretrained_path)
    return model


# ==================== Usage example ====================

if __name__ == "__main__":
    print("=" * 60)
    print("DenseNet variants test")
    print("=" * 60)
    
    # Test classifier
    print("\nTesting DenseNetClassifier:")
    classifier = DenseNetClassifier(num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    output = classifier(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test encoder
    print("\nTesting DenseNetEncoder:")
    encoder = DenseNetEncoder(embed_dim=512, num_classes=6)
    features = encoder(x)
    print(f"CLIP feature shape: {features.shape}")
    
    features, logits = encoder(x, return_features=True)
    print(f"CLIP feature shape: {features.shape}, classification logits shape: {logits.shape}")
    
    print("\n=" * 60)
    print("All model tests passed")

