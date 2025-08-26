import os
import random
import numpy as np
from PIL import Image
import yaml
import glob
import shutil

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image-mask pairs.
    Ensures that both images and masks are resized to the same dimensions.
    """
    def __init__(self, image_dir, mask_dir, image_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask

class DinoV3Segmenter(nn.Module):
    """
    A segmentation model using a pre-trained DINOv3 backbone and a configurable convolutional head.
    """
    def __init__(self, model_name='dinov3_vitb16', weights_path='model/dinov3_vitb16.pth', num_classes=1, head_channels=None):
        super().__init__()
        self.dinov3 = torch.hub.load('facebookresearch/dinov3', model_name, pretrained=False)
        try:
            state_dict = torch.load(weights_path)
            self.dinov3.load_state_dict(state_dict)
            print(f"Successfully loaded backbone weights from {weights_path}")
        except Exception as e:
            print(f"Could not load backbone weights from {weights_path}. Error: {e}")

        for param in self.dinov3.parameters():
            param.requires_grad = False

        dino_feature_dim = self.dinov3.embed_dim
        
        if head_channels is None:
            head_channels = [256, 128, 64, 32]
            
        layers = []
        in_channels = dino_feature_dim
        for out_channels in head_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ])
            in_channels = out_channels
        
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        features_list = self.dinov3.get_intermediate_layers(x, n=1, reshape=True)
        features = features_list[0]
        segmentation_mask = self.head(features)
        return segmentation_mask

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}")

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        
        val_loss = 0.0
        model.eval()
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({'loss': loss.item()})

        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  -> New best model saved to {output_path} with Val Loss: {best_val_loss:.4f}")

    print("Finished Training")

def main():
    config_dir = 'configs'
    config_files = glob.glob(os.path.join(config_dir, '**/*.yaml'), recursive=True)

    if not config_files:
        print(f"Error: No .yaml configuration files found in '{config_dir}/'.")
        return

    for i, config_path in enumerate(sorted(config_files)):
        print("\n" + "="*80)
        print(f"STARTING TRAINING {i+1}/{len(config_files)}: {config_path}")
        print("="*80 + "\n")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model_cfg = config['model']
        train_cfg = config['training']
        data_cfg = config['dataset']
        output_cfg = config['output']

        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dataset = SegmentationDataset(image_dir=data_cfg['train_images'], mask_dir=data_cfg['train_masks'], image_transform=image_transform)
        val_dataset = SegmentationDataset(image_dir=data_cfg['val_images'], mask_dir=data_cfg['val_masks'], image_transform=image_transform)

        train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=4)

        model = DinoV3Segmenter(
            model_name=model_cfg['name'],
            weights_path=model_cfg['weights'],
            num_classes=model_cfg['num_classes'],
            head_channels=model_cfg.get('head_channels') # Pass the head config
        )
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.head.parameters(), lr=train_cfg['optimizer']['lr0'])

        output_dir = os.path.join(output_cfg['save_dir'], output_cfg['run_name'])
        model_save_path = os.path.join(output_dir, 'best_model.pth')
        
        train_model(model, train_loader, val_loader, criterion, optimizer, 
                    num_epochs=train_cfg['epochs'], 
                    output_path=model_save_path)
        
        shutil.copy(config_path, os.path.join(output_dir, 'config.yaml'))
        print(f"Training run for {config_path} complete. Model and config saved to: {output_dir}")

if __name__ == '__main__':
    main()