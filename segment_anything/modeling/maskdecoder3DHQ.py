import torch
from torch import Tensor, nn
from typing import Tuple, Type

# Assuming LayerNorm3d and other necessary 3D operation classes are defined elsewhere

class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,  # Added for SAMHQ
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer3D(
            depth=2,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )

        self.num_multimask_outputs = num_multimask_outputs

        # Original SAM tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Added for SAMHQ: HQ-SAM parameters
        self.hf_token = nn.Embedding(1, transformer_dim)  # HQ-Output-Token
        self.hf_mlp = MLPBlock3D(transformer_dim, transformer_dim, transformer_dim // 8, 3)  # New MLP for HQ
        self.num_mask_tokens += 1  # Adjust for the added HQ token

        # Convolutional fusion layers for obtaining HQ-Feature, adapted for 3D
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose3d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim),
            activation(),
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )

        # Embedding encoder and mask feature layers adapted for 3D
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv3d(transformer_dim // 8, transformer_dim // 4, 3, padding=1),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.Conv3d(transformer_dim // 4, transformer_dim // 8, 3, padding=1)
        )

        # Original SAM upscaling and prediction heads
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLPBlock3D(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLPBlock3D(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)
