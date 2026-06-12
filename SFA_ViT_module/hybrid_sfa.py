import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- STANDARD FFN / MLP ---------------- #
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------- ASYMMETRIC SFA BLOCK (SPATIAL) ---------------- #
class AsymmetricSFABlock(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # RGB Self-Attention
        self.norm_rgb = nn.LayerNorm(dim)
        self.qkv_rgb = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop)

        # DCT Projections (Keys/Values ONLY)
        self.norm_dct = nn.LayerNorm(dim)
        self.kv_dct = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Cross-Attention (RGB queries DCT)
        self.norm_cross = nn.LayerNorm(dim)
        self.q_cross = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Fusion components
        self.norm_fusion = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        # MLP components
        self.mlp = MLP(in_features=dim, drop=drop)
        self.norm_mlp = nn.LayerNorm(dim)
        
        # Apply ViT-specific weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_rgb, x_dct):
        B, N, C = x_rgb.shape

        # 1. RGB Self-Attention
        normed_rgb = self.norm_rgb(x_rgb)
        qkv_r = self.qkv_rgb(normed_rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_r, k_r, v_r = qkv_r[0], qkv_r[1], qkv_r[2]
        
        attn_r = (q_r @ k_r.transpose(-2, -1)) * self.scale
        # AMP Safe Softmax
        attn_r = attn_r.float().softmax(dim=-1).type_as(attn_r)
        attn_r = self.attn_drop(attn_r)
        
        # Residual added directly to attention output
        rgb_sa = x_rgb + (attn_r @ v_r).transpose(1, 2).reshape(B, N, C)

        # 2. Asymmetric Cross-Attention (RGB_SA queries DCT)
        normed_dct = self.norm_dct(x_dct)
        kv_d = self.kv_dct(normed_dct).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_d, v_d = kv_d[0], kv_d[1]
        
        q_c = self.q_cross(self.norm_cross(rgb_sa)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn_c = (q_c @ k_d.transpose(-2, -1)) * self.scale
        # AMP Safe Softmax
        attn_c = attn_c.float().softmax(dim=-1).type_as(attn_c)

        # ---> NEW: Save the raw, pure mathematical Softmax probability matrix BEFORE dropout!
        self.last_spatial_attn = attn_c.detach()

        attn_c = self.attn_drop(attn_c)
        cross_out = (attn_c @ v_d).transpose(1, 2).reshape(B, N, C)

        # 3. Normalized Fusion & MLP
        x = x_rgb + self.proj_drop(self.proj(self.norm_fusion(cross_out)))
        x = x + self.mlp(self.norm_mlp(x))

        # 4. Soft Decorrelation Loss
        f_rgb_norm = F.normalize(x_rgb.reshape(B, -1), p=2, dim=1)
        f_dct_norm = F.normalize(x_dct.reshape(B, -1), p=2, dim=1)
        soft_orth_loss = torch.abs(torch.sum(f_rgb_norm * f_dct_norm, dim=1)).mean()

        return x, x_dct, soft_orth_loss