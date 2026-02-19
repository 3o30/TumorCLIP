"""数据变换定义"""

from torchvision import transforms

class TransformFactory:
    """数据变换工厂类"""
    
    @staticmethod
    def get_single_modal_transforms(mode='train'):
        """获取单模态数据变换"""
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:  # test/val
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def get_multimodal_transforms():
        """获取多模态数据变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])