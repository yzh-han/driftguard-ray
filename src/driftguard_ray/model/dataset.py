from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

from PIL import Image
import io

class ListDataset(Dataset):
    def __init__(self, samples: List[Tuple[bytes, int]], transform=None):
        """
        Args:
            samples: List of (image_bytes, label) tuples
            transform: Optional transform to apply to images
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_bytes, label = self.samples[idx]
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_inference_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(img_size // 7 * 8),  # 先 resize 到稍大一点， 短边 8/7 倍 保存比例
            T.CenterCrop(img_size),  # 再中心裁剪到 s * s
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )


def get_train_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(img_size),  # Random crop and resize
            T.RandomHorizontalFlip(),  # Random horizontal flip with 50% probability
            T.RandomRotation(5),  # Random rotation with a maximum of 15 degrees
            # T.ColorJitter(
            #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            # ),  # Random color jitter
            T.ToTensor(),  # Convert to C×H×W (float32) Tensor, normalize to [0,1]
            T.RandomErasing(p=0.1),  # Random erasing with 10% probability
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )


def get_inverse_transform() -> T.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Denormalize: new_mean = [-mu/std], new_std = [1/std]
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )

    return T.Compose(
        [
            inv_normalize,  # (C,H,W), float, approximately back to [0,1]
            T.ToPILImage(),  # Convert to PIL Image
        ]
    )