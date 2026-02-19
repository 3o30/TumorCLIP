from torchvision import models
import timm
import torch.nn as nn

# Add cache configuration
try:
    from config.model_cache_config import ModelCacheConfig
    ModelCacheConfig.setup_cache_environment()
except ImportError:
    print("WARNING: Model cache configuration not found; using system default path")



class ModelRegistry:
    """Model registry"""
    
    _models = {}
    
    @classmethod
    def register(cls, name, loader_func, model_type='standard', batch_size=32):
        """Register a model."""
        cls._models[name] = {
            'loader': loader_func,
            'type': model_type,
            'batch_size': batch_size
        }
    
    @classmethod
    def get_model_info(cls, name):
        """Get model metadata by name."""
        return cls._models.get(name)
    
    @classmethod
    def get_all_models(cls):
        """Get all registered models."""
        return cls._models
    
    @classmethod
    def get_model_names(cls):
        """Get all registered model names."""
        return list(cls._models.keys())

# Register all models
ModelRegistry.register(
    'EfficientNet_b0', 
    lambda: models.efficientnet_b0(weights='DEFAULT'),
    'standard', 32
)

ModelRegistry.register(
    'ResNet50',
    lambda: models.resnet50(weights='DEFAULT'), 
    'standard', 32
)

ModelRegistry.register(
    'MobileNetV3_large',
    lambda: models.mobilenet_v3_large(weights='DEFAULT'),
    'standard', 32
)

ModelRegistry.register(
    'ViT',
    lambda: timm.create_model('vit_base_patch16_224', pretrained=True),
    'standard', 32
)

ModelRegistry.register(
    'DenseNet121', 
    lambda: models.densenet121(weights='DEFAULT'),
    'standard', 32
)

ModelRegistry.register(
    'DeiT',
    lambda: timm.create_model('deit_base_patch16_224', pretrained=True),
    'standard', 32
)

ModelRegistry.register(
    'Swin Transformer',
    lambda: timm.create_model('swin_base_patch4_window7_224', pretrained=True),
    'swin', 32
)

ModelRegistry.register(
    'MambaOut',
    lambda: timm.create_model('mambaout_tiny.in1k', pretrained=True),
    'standard', 32
)

