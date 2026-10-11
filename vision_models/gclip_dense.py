"""
GCLIP Dense Feature Extractor

Implements the GCLIP (AAAI 2025) training-free method for extracting dense
CLIP-aligned features from a ViT-B/16 backbone. Key modifications to the
standard CLIP ViT forward pass:

1. Attention Map Fusion (AMF): Fuses QK attention from global-token emerging
   blocks with QQ self-attention from the last block.
2. Channel Suppression (CS): Re-normalizes abnormally large FFN weight channels
   in blocks 7-11 (for ViT-B/16).
3. ClearCLIP-style: Skips residual connection and FFN in the last block.

Reference: "Rethinking the Global Knowledge of CLIP in Training-Free
Open-Vocabulary Semantic Segmentation" (Wang et al., AAAI 2025)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import open_clip

from vision_models.base_model import BaseModel
from typing import List


class GCLIPModel(torch.nn.Module, BaseModel):
    def __init__(self,
                 clip_input_size: int = 640,
                 gclip_depth: int = 4,
                 cs_start_block: int = 7,
                 ):
        """
        Args:
            clip_input_size: Input image size for CLIP ViT (will be resized).
                Must be divisible by 16. 640 -> 40x40 grid.
            gclip_depth: Number of last blocks to apply GEM/GCLIP pathway.
            cs_start_block: First block index to apply channel suppression.
        """
        super(GCLIPModel, self).__init__()

        self.clip_input_size = clip_input_size
        self.gclip_depth = gclip_depth
        self.cs_start_block = cs_start_block

        # Load CLIP ViT-B/16 with OpenAI pretrained weights
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', device='cuda'
        )
        clip_model = clip_model.float().eval()

        self.visual = clip_model.visual
        self.text = clip_model  # keep full model for encode_text
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

        # ViT-B/16: width=768, output_dim=512, 12 blocks, 12 heads
        self.width = self.visual.transformer.width  # 768
        self.output_dim = self.visual.output_dim  # 512
        self.feature_dim = self.output_dim  # 512 — used by OneMap
        self.num_blocks = self.visual.transformer.layers  # 12
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads  # 12
        self.head_dim = self.width // self.num_heads  # 64

        # Grid size for positional embedding interpolation
        self.orig_grid_size = self.visual.grid_size  # (14, 14) for 224x224
        self.target_grid_size = (
            clip_input_size // 16,
            clip_input_size // 16,
        )  # e.g. (40, 40) for 640

        # CLIP normalization (OpenAI CLIP uses 0-1 range internally,
        # but we receive 0-255 RGB from the pipeline)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(3, 1, 1) * 255.0
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(3, 1, 1) * 255.0

        # Precompute channel suppression masks for FFN c_proj weights
        self._cs_masks = self._precompute_channel_suppression()

        # Global-token emerging block index (default 6 for ViT-B/16, per GCLIP paper)
        # QK attention from blocks g and g+1 will be fused with QQ from last block
        self._global_block_idx = 6

    def _precompute_channel_suppression(self):
        """Precompute channel suppression corrections for blocks cs_start..N-1."""
        cs_masks = {}
        for i in range(self.cs_start_block, self.num_blocks):
            block = self.visual.transformer.resblocks[i]
            w = block.mlp.c_proj.weight  # [d_model, mlp_width]
            # Compute L2 norm per output channel
            norms = w.norm(dim=1)  # [d_model]
            d_hat = norms.argmax().item()
            # Mean norm excluding the outlier channel
            mask = torch.ones(norms.shape[0], dtype=torch.bool, device=w.device)
            mask[d_hat] = False
            n_bar = norms[mask].mean()
            # Store correction: scale factor for the outlier channel
            cs_masks[i] = (d_hat, n_bar / (norms[d_hat] + 1e-8))
        return cs_masks

    def _interpolate_pos_embed(self, pos_embed):
        """Interpolate positional embeddings for different input sizes."""
        if self.target_grid_size == self.orig_grid_size:
            return pos_embed

        cls_pos = pos_embed[:1, :]  # [1, width]
        patch_pos = pos_embed[1:, :]  # [orig_H*orig_W, width]

        orig_h, orig_w = self.orig_grid_size
        target_h, target_w = self.target_grid_size

        patch_pos = patch_pos.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(), size=(target_h, target_w),
            mode='bicubic', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, self.width)

        return torch.cat([cls_pos, patch_pos], dim=0)

    def _extract_qk_attention(self, block, x):
        """Extract QK attention map from a block without running the full forward."""
        y = block.ln_1(x)
        qkv = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        L, N, _ = y.shape
        q, k, _ = qkv.chunk(3, dim=-1)
        scale = self.head_dim ** -0.5
        q = q.contiguous().view(L, N * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, self.head_dim).transpose(0, 1)
        attn = torch.bmm(q * scale, k.transpose(-1, -2)).softmax(dim=-1)
        # Average over heads
        attn = attn.view(N, self.num_heads, L, L).mean(dim=1)  # [N, L, L]
        return attn

    def _extract_qq_attention(self, block, x):
        """Extract QQ self-self attention map from a block."""
        y = block.ln_1(x)
        qkv = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        L, N, _ = y.shape
        q, _, _ = qkv.chunk(3, dim=-1)
        scale = self.head_dim ** -0.5
        q = q.contiguous().view(L, N * self.num_heads, self.head_dim).transpose(0, 1)
        attn = torch.bmm(q * scale, q.transpose(-1, -2)).softmax(dim=-1)
        # Average over heads
        attn = attn.view(N, self.num_heads, L, L).mean(dim=1)
        return attn

    def _apply_channel_suppression(self, block, block_idx):
        """Apply channel suppression to FFN c_proj weight, return original for restore."""
        if block_idx not in self._cs_masks:
            return None
        d_hat, scale_factor = self._cs_masks[block_idx]
        w = block.mlp.c_proj.weight
        original = w.data[d_hat].clone()
        w.data[d_hat] = w.data[d_hat] * scale_factor
        return (d_hat, original)

    def _restore_channel(self, block, backup):
        """Restore FFN c_proj weight after channel suppression."""
        if backup is None:
            return
        d_hat, original = backup
        block.mlp.c_proj.weight.data[d_hat] = original

    @torch.no_grad()
    def image_forward_gclip(self, images: torch.Tensor):
        """
        GCLIP modified forward pass.

        Args:
            images: [B, 3, H, W] in 0-255 range, RGB

        Returns:
            dense features [B, feature_dim, grid_h, grid_w], L2-normalized
        """
        # Normalize
        images = (images - self.clip_mean) / self.clip_std
        images = F.interpolate(images, size=(self.clip_input_size, self.clip_input_size),
                               mode='bilinear', align_corners=False)

        # Patch embedding
        x = self.visual.conv1(images)  # [B, width, grid_h, grid_w]
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, L, width]

        # Add CLS token
        x = torch.cat([
            self.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # [B, L+1, width]

        # Positional embedding (interpolated if needed)
        pos_embed = self._interpolate_pos_embed(self.visual.positional_embedding)
        x = x + pos_embed.to(x.dtype)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND (for nn.MultiheadAttention)

        N = images.shape[0]  # batch size
        gclip_start = self.num_blocks - self.gclip_depth
        g = self._global_block_idx

        # Collect QK attention maps from global blocks
        qk_attns = []

        # Run blocks 0 .. N-2 normally, collecting QK attns at global blocks
        for i in range(self.num_blocks - 1):
            # Apply channel suppression if applicable
            block = self.visual.transformer.resblocks[i]
            backup = self._apply_channel_suppression(block, i)

            # Collect QK attention from global-token blocks
            if i == g or i == g + 1:
                qk_attn = self._extract_qk_attention(block, x)
                qk_attns.append(qk_attn)

            # Run the block normally
            x = block(x)

            self._restore_channel(block, backup)

        # Last block: GCLIP modified forward
        last_block = self.visual.transformer.resblocks[-1]
        backup = self._apply_channel_suppression(last_block, self.num_blocks - 1)

        # Extract QQ attention from last block
        qq_attn = self._extract_qq_attention(last_block, x)

        # Fuse attention maps: AMF
        # A_f = (sum of QK attns from global blocks + QQ from last block) / count
        fused_attn = qq_attn
        for qk in qk_attns:
            fused_attn = fused_attn + qk
        fused_attn = fused_attn / (len(qk_attns) + 1)

        # Get Value from last block
        y = last_block.ln_1(x)
        qkv = F.linear(y, last_block.attn.in_proj_weight, last_block.attn.in_proj_bias)
        L, B, _ = y.shape
        _, _, v = qkv.chunk(3, dim=-1)  # [L, B, width]

        # Apply fused attention to value: [N, L, L] @ [N, L, D] -> [N, L, D]
        v_for_attn = v.permute(1, 0, 2)  # [B, L, width]
        out = torch.bmm(fused_attn, v_for_attn)  # [B, L, width]

        # Apply output projection (without residual or FFN — ClearCLIP style)
        out = out.permute(1, 0, 2)  # [L, B, width]
        out = F.linear(out, last_block.attn.out_proj.weight, last_block.attn.out_proj.bias)
        out = out.permute(1, 0, 2)  # [B, L, width]

        self._restore_channel(last_block, backup)

        # Extract patch tokens (remove CLS)
        patch_tokens = out[:, 1:, :]  # [B, grid_h*grid_w, width]

        # Apply ln_post and projection
        patch_tokens = self.visual.ln_post(patch_tokens)
        patch_tokens = patch_tokens @ self.visual.proj  # [B, grid_h*grid_w, output_dim]

        # Reshape to spatial
        patch_tokens = patch_tokens.permute(0, 2, 1)  # [B, output_dim, grid_h*grid_w]
        patch_tokens = patch_tokens.reshape(N, self.output_dim, grid_h, grid_w)

        return F.normalize(patch_tokens, dim=1)

    def eval(self):
        super().eval()
        self.visual.eval()
        return self

    def get_image_features(self, images: np.ndarray) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] in RGB, 0-255 range, numpy array

        Returns:
            [B, feature_dim, grid_h, grid_w] L2-normalized features
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        with torch.no_grad():
            images_t = torch.as_tensor(images.astype("float32")).to("cuda")
            return self.image_forward_gclip(images_t)

    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to("cuda")
            text_features = self.text.encode_text(tokens)
            return F.normalize(text_features, dim=1)

    def compute_similarity(self,
                           image_feats: torch.Tensor,
                           text_feats: torch.Tensor) -> torch.Tensor:
        if len(image_feats.shape) == 3:
            return torch.einsum('bcx, bc -> bx', image_feats, text_feats)
        else:
            return torch.einsum('bchw, bc -> bhw', image_feats, text_feats)

