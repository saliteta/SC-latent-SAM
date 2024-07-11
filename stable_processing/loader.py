import os
from PIL import Image
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple


import numpy as np

'''
    This is the image loader for sam images. 
    after loading the sam images, we will process it using encoder to extract features in patch
'''

class TransformImage:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int =1024, device = 'cpu') -> None:
        self.target_length = target_length
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
        
        self.image_size = target_length

    def apply_image(self, image: np.ndarray, device = 'cuda') -> torch.Tensor:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        input_image=  np.array(resize(to_pil_image(image), target_size))
        
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        input_image = self.preprocess(input_image_torch)
        return input_image
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
        
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


    
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): A TransformImage instance or similar for processing images.
            device (string): Device to perform computations on.
        """
        self.directory = directory
        self.transform = transform  # Expecting an instance of TransformImage
        self.images = [os.path.join(directory, img) for img in sorted(os.listdir(directory)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform.apply_image(image, 'cpu')
        
        basename = os.path.basename(img_path)
        return image, basename


def load_dataset(directory, batch_size, num_workers):
    # Initialize the image transformation class
    transform = TransformImage()
    
    # Create dataset
    dataset = ImageDataset(directory, transform=transform)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return dataloader, len(dataset)

    
