import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from SFA_ViT_module.hybrid_sfa import AsymmetricSFABlock, MLP

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        l = (a - mean) / std
        u = (b - mean) / std
        tensor.normal_().fmod_(2).clamp_(l, u).mul_(std).add_(mean)
        return tensor

class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, drop=drop)
        self.freq_proj = nn.Linear(dim, dim)
        
        # NEW: Learnable frequency scale, initialized very low to prevent variance explosion
        self.freq_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x_freq = dct.dct(x.float(), norm='ortho')
        freq_bias = self.freq_proj(x_freq.to(x.dtype)) 
        
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # NEW: Scaled residual addition
        x = x + attn_out + (self.freq_scale * freq_bias)
        x = x + self.mlp(self.norm2(x))
        return x

class EnhancedMINTIME(nn.Module):
    def __init__(self, num_classes=2, img_size=224, patch_size=16, seq_length=5, dim=384, depth=4, num_heads=6, drop=0.1):
        super().__init__()
        self.seq_length = seq_length
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed_rgb = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_dct = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        self.norm_pre_rgb = nn.LayerNorm(dim)
        self.norm_pre_dct = nn.LayerNorm(dim)
        
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Removed static temporal_cls_token in favor of data-dependent pooling
        
        self.pos_embed_rgb = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.pos_embed_dct = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.time_embed = nn.Parameter(torch.zeros(1, seq_length, dim)) 
        
        self.pos_drop = nn.Dropout(p=drop)
        
        self.spatial_blocks = nn.ModuleList([AsymmetricSFABlock(dim, num_heads=num_heads, drop=drop) for _ in range(depth)])
        self.norm_post_sfa = nn.LayerNorm(dim)
        
        self.temporal_blocks = nn.ModuleList([TemporalAttentionBlock(dim, num_heads=num_heads, drop=drop) for _ in range(depth)])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, num_classes))
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed_rgb, std=.02)
        trunc_normal_(self.pos_embed_dct, std=.02)
        trunc_normal_(self.spatial_cls_token, std=.02)
        trunc_normal_(self.time_embed, std=.02)

    def extract_features(self, x_rgb, x_dct=None):
        """
        Explicitly extracts the 384-dimensional latent embedding
        immediately preceding the classification MLP head.
        """
        if x_dct is None: x_dct = x_rgb

        B, T, C, H, W = x_rgb.shape
        x_rgb = x_rgb.reshape(B * T, C, H, W).contiguous()
        x_dct = x_dct.reshape(B * T, C, H, W).contiguous()
        
        with torch.cuda.amp.autocast(enabled=False):
            true_dct = dct.dct_2d(x_dct.float(), norm='ortho').to(x_dct.dtype)
            
        x_feat_rgb = self.patch_embed_rgb(x_rgb)
        x_feat_dct = self.patch_embed_dct(true_dct)
        
        x_rgb_flat = self.norm_pre_rgb(x_feat_rgb.flatten(2).transpose(1, 2))
        x_dct_flat = self.norm_pre_dct(x_feat_dct.flatten(2).transpose(1, 2))
        
        cls_tokens = self.spatial_cls_token.expand(B * T, -1, -1)
        x_rgb_final = self.pos_drop(torch.cat((cls_tokens, x_rgb_flat), dim=1) + self.pos_embed_rgb)
        x_dct_final = self.pos_drop(torch.cat((cls_tokens, x_dct_flat), dim=1) + self.pos_embed_dct)
        
        for blk in self.spatial_blocks:
            x_rgb_final, x_dct_final, _ = blk(x_rgb_final, x_dct_final)
            
        x_rgb_final = self.norm_post_sfa(x_rgb_final)
        x_dct_final = self.norm_post_sfa(x_dct_final)
            
        frame_cls_rgb = x_rgb_final[:, 0, :].reshape(B, T, -1).contiguous()
        frame_cls_dct = x_dct_final[:, 0, :].reshape(B, T, -1).contiguous()
        
        frame_cls = 0.5 * (frame_cls_rgb + frame_cls_dct) + self.time_embed
        temp_cls = frame_cls.mean(dim=1, keepdim=True)
        temp_input = torch.cat((temp_cls, frame_cls), dim=1) 
        
        for blk in self.temporal_blocks:
            temp_input = blk(temp_input)
            
        # Return the exact 1D embedding vector [Batch, Dim]
        embedding = self.norm(temp_input[:, 0, :])
        return embedding
    
    def forward(self, x_rgb, x_dct=None):
        if x_dct is None: x_dct = x_rgb

        B, T, C, H, W = x_rgb.shape
        
        x_rgb = x_rgb.reshape(B * T, C, H, W).contiguous()
        x_dct = x_dct.reshape(B * T, C, H, W).contiguous()
        
        with torch.cuda.amp.autocast(enabled=False):
            true_dct = dct.dct_2d(x_dct.float(), norm='ortho')
        true_dct = true_dct.to(x_dct.dtype)
        
        x_feat_rgb = self.patch_embed_rgb(x_rgb)
        x_feat_dct = self.patch_embed_dct(true_dct)
        
        x_rgb_flat = self.norm_pre_rgb(x_feat_rgb.flatten(2).transpose(1, 2))
        x_dct_flat = self.norm_pre_dct(x_feat_dct.flatten(2).transpose(1, 2))
        
        cls_tokens = self.spatial_cls_token.expand(B * T, -1, -1)
        
        x_rgb_final = torch.cat((cls_tokens, x_rgb_flat), dim=1) + self.pos_embed_rgb
        x_dct_final = torch.cat((cls_tokens, x_dct_flat), dim=1) + self.pos_embed_dct
        
        x_rgb_final = self.pos_drop(x_rgb_final)
        x_dct_final = self.pos_drop(x_dct_final)
        
        total_orth_loss = 0
        num_blks = len(self.spatial_blocks)
        for blk in self.spatial_blocks:
            x_rgb_final, x_dct_final, orth_loss = blk(x_rgb_final, x_dct_final)
            # FIX: Accumulate safely to prevent AMP FP16 overflow
            total_orth_loss = total_orth_loss + (orth_loss / num_blks) 
            
        x_rgb_final = self.norm_post_sfa(x_rgb_final)
        x_dct_final = self.norm_post_sfa(x_dct_final)
            
        # FIX: Symmetrical Domain Fusion for Temporal Block
        frame_cls_rgb = x_rgb_final[:, 0, :].reshape(B, T, -1).contiguous()
        frame_cls_dct = x_dct_final[:, 0, :].reshape(B, T, -1).contiguous()
        
        frame_cls = 0.5 * (frame_cls_rgb + frame_cls_dct)
        frame_cls = frame_cls + self.time_embed
        
        # FIX: Data-Dependent Temporal CLS Token
        temp_cls = frame_cls.mean(dim=1, keepdim=True)
        temp_input = torch.cat((temp_cls, frame_cls), dim=1) 
        
        for blk in self.temporal_blocks:
            temp_input = blk(temp_input)
            
        final_out = self.norm(temp_input[:, 0, :])
        logits = self.head(final_out)
        
        return logits, total_orth_loss