import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
import os

class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Normalize mask to 0.0 and 1.0
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256)):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image and mask
        img = Image.new('RGB', self.img_size, color='black')
        mask = Image.new('L', self.img_size, color=0)
        
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Randomly draw a "brain" (ellipse)
        cx, cy = self.img_size[0] // 2, self.img_size[1] // 2
        rx, ry = np.random.randint(50, 100), np.random.randint(60, 110)
        draw_img.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill='gray', outline='white')
        
        # Randomly draw a "tumor" (smaller ellipse inside the brain)
        if np.random.rand() > 0.2: # 80% chance of tumor
            tx = np.random.randint(cx-rx//2, cx+rx//2)
            ty = np.random.randint(cy-ry//2, cy+ry//2)
            tr = np.random.randint(5, 20)
            draw_img.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill='white')
            draw_mask.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill=1) # Mask is 1 for tumor
        
        image = np.array(img)
        mask = np.array(mask, dtype=np.float32)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
