import torch
import torch.nn as nn
from torchvision.models import (
    vit_h_14,
    ViT_H_14_Weights,
    vit_l_16,
    ViT_L_16_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


class ViTBackbone(nn.Module):
    def __init__(self, size: str = "huge", pretrained=True, freeze=False):
        super(ViTBackbone, self).__init__()

        if size == "huge":
            self.model = (
                vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
                if pretrained
                else vit_h_14()
            )
        elif size == "large":
            self.model = (
                vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
                if pretrained
                else vit_l_16()
            )
        elif size == "base":
            self.model = (
                vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                if pretrained
                else vit_b_16()
            )

        # Optionally freeze the backbone
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Step 1: Patch embedding
        x = self.model._process_input(
            x
        )  # [1, 256, 1280] (huge) / [1, 196, 1024] (large)

        # Step 2: Add CLS token
        cls_token = self.model.class_token.expand(
            x.size(0), -1, -1
        )  # [1, 1, 1280] / [1, 1, 1024]
        x = torch.cat((cls_token, x), dim=1)  # [1, 257, 1280] / [1, 197, 1024]

        # Step 3: Add positional embedding
        x = x + self.model.encoder.pos_embedding  # [1, 257, 1280] / [1, 197, 1024]

        # Step 4: Pass through transformer layers
        for blk in self.model.encoder.layers:
            x = blk(x)  # [1, 257, 1280] / [1, 197, 1024]
        x = self.model.encoder.ln(x)  # Final layer norm

        # Step 5: Get patch tokens only, reshape to [B, C, H, W]
        patch_tokens = x[:, 1:]  # Remove CLS token → [1, 196, 1024]
        B, N, C = patch_tokens.shape
        H = W = int(N**0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            B, C, H, W
        )  # [1, 1280, 16, 16] (Huge) / [1, 1024, 14, 14] (Large)
        return patch_tokens  # [1, 1280, 16, 16] (Huge) / [1, 1024, 14, 14] (Large)
