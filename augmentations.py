import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import random

class AttentionConsistentAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Temporal Space
        self.temporal_aug = K.VideoSequential(
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            data_format="BCTHW",
            same_on_frame=True
        )

        # 2. Geometric Space
        self.shared_geo_aug = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5)
        )

        # NEW: 3. Social Media Wash (Compression & Blur)
        # Adjusted to (50, 95) to prevent total destruction of facial microstructure
        self.social_wash = nn.Sequential(
            K.RandomJPEG(jpeg_quality=(50, 95), p=0.4), 
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5), p=0.2)
        )

        # 4. Spectral Space (RGB Specific)
        self.rgb_spectral_aug = nn.Sequential(
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.2)
        )

    def forward(self, x):
        if not self.training:
            return x, x.clone()

        B, T, C, H, W = x.shape
        
        # --- 1. Temporal Space ---
        x_vid = x.permute(0, 2, 1, 3, 4)
        x_vid = self.temporal_aug(x_vid)
        x_flat = x_vid.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # --- 2. Shared Geometric Space ---
        x_base = self.shared_geo_aug(x_flat)
        
        # NEW: --- 3. Resize-Recompression (Continuous WhatsApp/TikTok Simulation) ---
        if random.random() < 0.3:
            # Shifted to continuous uniform distribution to prevent discrete artifact memorization
            scale = random.uniform(0.5, 0.85) 
            small_H, small_W = int(H * scale), int(W * scale)
            
            x_base = F.interpolate(x_base, size=(small_H, small_W), mode='bilinear', align_corners=False)
            x_base = F.interpolate(x_base, size=(H, W), mode='bilinear', align_corners=False)
            
        # Apply standard Kornia compression wash
        x_base = self.social_wash(x_base)
        
        # --- 4. Bifurcation into Isolated Spectral Spaces ---
        
        # RGB Spectral Corruption
        aug_rgb = self.rgb_spectral_aug(x_base)
        noise_mask = torch.rand(B * T, 1, 1, 1, device=x.device) < 0.3
        noise = torch.randn_like(aug_rgb, dtype=torch.float32).to(aug_rgb.dtype) * 0.05 * noise_mask.float()
        aug_rgb = torch.clamp(aug_rgb + noise, 0.0, 1.0).view(B, T, C, H, W)
        
        # DCT Spectral Corruption
        aug_dct = x_base.clone()
        x_freq = torch.fft.rfft2(aug_dct.float(), norm="ortho")
        x_freq = torch.nan_to_num(x_freq, nan=0.0, posinf=0.0, neginf=0.0)
        
        freq_mask = (torch.rand_like(x_freq.real) > 0.05).float()
        x_freq = x_freq * freq_mask
        x_freq = torch.nan_to_num(x_freq, nan=0.0, posinf=0.0, neginf=0.0)
        
        aug_dct = torch.fft.irfft2(x_freq, s=(H, W), norm="ortho").to(aug_dct.dtype)
        aug_dct = (aug_dct * 255.0).round() / 255.0
        aug_dct = torch.clamp(aug_dct, 0.0, 1.0).view(B, T, C, H, W)
        
        return aug_rgb, aug_dct