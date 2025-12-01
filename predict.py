import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

def predict(model, image_path, device=DEVICE):
    model.eval()
    image = np.array(Image.open(image_path).convert("RGB"))
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()
    
    return preds.cpu().squeeze().numpy()

def visualize(image_path, mask):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.show()

def main():
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    checkpoint = torch.load("my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    
    # Example usage:
    # pred_mask = predict(model, "Dataset/val_images/some_image.jpg")
    # visualize("Dataset/val_images/some_image.jpg", pred_mask)
    print("Model loaded. Use predict() function to infer on new images.")

if __name__ == "__main__":
    main()
