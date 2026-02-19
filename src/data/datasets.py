"""数据集类定义"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from collections import defaultdict

class BrainTumorDataset(Dataset):
    """脑瘤数据集类"""
    
    def __init__(self, root_dir, transform=None, mode='single_modal'):
        """
        Args:
            root_dir: 数据根目录
            transform: 数据变换
            mode: 'single_modal' 或 'multimodal'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # 使用torchvision的ImageFolder来处理数据
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
        # 多模态时的文本模板
        if mode == 'multimodal':
            self.text_templates = self._create_text_templates()
    
    def _create_text_templates(self):
        """为多模态模型创建文本描述模板"""
        templates = {
            'Glioma': [
                "a medical scan showing glioma brain tumor",
                "brain MRI with glioma tumor",
                "glioma brain cancer in medical imaging"
            ],
            'Meningioma': [
                "a medical scan showing meningioma brain tumor", 
                "brain MRI with meningioma tumor",
                "meningioma brain tumor in medical imaging"
            ],
            'Neurocitoma': [
                "a medical scan showing neurocitoma brain tumor",
                "brain MRI with neurocitoma tumor", 
                "neurocitoma brain tumor in medical imaging"
            ],
            'Outros Tipos de Lesões': [
                "a medical scan showing other brain lesions",
                "brain MRI with atypical lesions",
                "other types of brain abnormalities"
            ],
            'Schwannoma': [
                "a medical scan showing schwannoma brain tumor",
                "brain MRI with schwannoma tumor",
                "schwannoma brain tumor in medical imaging"
            ],
            'NORMAL': [
                "a normal brain scan without tumors",
                "healthy brain MRI with no abnormalities", 
                "normal brain medical imaging"
            ]
        }
        return templates
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.mode == 'single_modal':
            return self.dataset[idx]
        
        elif self.mode == 'multimodal':
            image, label = self.dataset[idx]
            class_name = self.classes[label]
            
            # 随机选择文本模板
            import random
            text = random.choice(self.text_templates.get(class_name, ["medical brain scan"]))
            
            return {
                'image': image,
                'text': text,
                'label': label,
                'class_name': class_name
            }
    
    def get_class_distribution(self):
        """获取类别分布"""
        class_counts = defaultdict(int)
        for _, label in self.dataset:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return dict(class_counts)