"""数据加载器工厂"""

import torch
from torch.utils.data import DataLoader, random_split
from .datasets import BrainTumorDataset
from .transforms import TransformFactory

class DataLoaderFactory:
    """数据加载器工厂类"""
    
    @staticmethod
    def create_single_modal_loaders(train_path, test_path, batch_size, num_workers=4):
        """创建单模态数据加载器"""
        train_transform = TransformFactory.get_single_modal_transforms('train')
        test_transform = TransformFactory.get_single_modal_transforms('test')
        
        train_dataset = BrainTumorDataset(train_path, train_transform, 'single_modal')
        test_dataset = BrainTumorDataset(test_path, test_transform, 'single_modal')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader, train_dataset.classes

    @staticmethod
    def create_single_modal_loaders_with_val(train_path, test_path, batch_size, num_workers=4, val_ratio=0.2):
        """创建单模态数据加载器，包含验证集拆分"""
        train_transform = TransformFactory.get_single_modal_transforms('train')
        test_transform = TransformFactory.get_single_modal_transforms('test')

        full_train_dataset = BrainTumorDataset(train_path, train_transform, 'single_modal')
        test_dataset = BrainTumorDataset(test_path, test_transform, 'single_modal')

        train_size = int((1 - val_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader, full_train_dataset.classes
    
    @staticmethod
    def create_multimodal_loaders(train_path, test_path, processor, batch_size, 
                                val_ratio=0.2, num_workers=0):
        """创建多模态数据加载器"""
        full_train_dataset = BrainTumorDataset(train_path, None, 'multimodal')
        test_dataset = BrainTumorDataset(test_path, None, 'multimodal')
        
        # 分割训练和验证集
        train_size = int((1 - val_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        def collate_fn(batch):
            images = [item['image'] for item in batch]
            texts = [item['text'] for item in batch]
            labels = torch.tensor([item['label'] for item in batch])
            
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs['labels'] = labels
            return inputs
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader, full_train_dataset.classes