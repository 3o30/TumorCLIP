"""Single-modal model factory"""

import torch.nn as nn
from .model_registry import ModelRegistry

class ModelFactory:
    """Model factory."""
    
    @staticmethod
    def create_model(model_name, num_classes):
        """Create a model by name."""
        model_info = ModelRegistry.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"未知模型: {model_name}")
        
        print(f"Creating model: {model_name}")
        
        # Create base model
        model = model_info['loader']()
        
        if model is None:
            raise RuntimeError(f"模型 {model_name} 创建失败")
        
        # Modify classification head
        model = ModelFactory._modify_classifier(model, num_classes, model_name, model_info['type'])
        
        print(f"Model {model_name} created successfully")
        return model
    
    @staticmethod
    def _modify_classifier(model, num_classes, model_name, model_type):
        """Modify a model's classification head."""
        try:
            if model_type == 'swin':
                return ModelFactory._modify_swin_classifier(model, num_classes)
            
            elif 'mambaout' in model_name.lower():
                if hasattr(model, 'head'):
                    if hasattr(model.head, 'fc'):
                        in_features = model.head.fc.in_features
                        model.head.fc = nn.Linear(in_features, num_classes)
                    elif hasattr(model.head, 'in_features'):
                        in_features = model.head.in_features
                        model.head = nn.Linear(in_features, num_classes)
                elif hasattr(model, 'classifier'):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            
            elif 'deit' in model_name.lower() or 'vit' in model_name.lower():
                if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
                elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    for i in range(len(model.classifier)-1, -1, -1):
                        if isinstance(model.classifier[i], nn.Linear):
                            in_features = model.classifier[i].in_features
                            model.classifier[i] = nn.Linear(in_features, num_classes)
                            break
                elif isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            
            print(f"Updated {model_name} classifier to {num_classes} classes")
        except Exception as e:
            print(f"ERROR: Failed to update {model_name} classifier: {e}")
        
        return model
    
    @staticmethod
    def _modify_swin_classifier(model, num_classes):
        """Special handling for Swin Transformer's classification head."""
        print("Detected Swin Transformer; applying special handling...")
        
        if hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                in_features = model.head.fc.in_features
                new_fc = nn.Linear(in_features, num_classes)
                model.head.fc = new_fc
                print(f"   Updated head.fc: {in_features} -> {num_classes}")
            elif hasattr(model.head, 'in_features'):
                in_features = model.head.in_features
                new_head = nn.Linear(in_features, num_classes)
                model.head = new_head
                print(f"   Updated head: {in_features} -> {num_classes}")
        
        return model
    
    @staticmethod
    def get_batch_size(model_name):
        """Get the recommended batch size for a model."""
        model_info = ModelRegistry.get_model_info(model_name)
        return model_info['batch_size'] if model_info else 32