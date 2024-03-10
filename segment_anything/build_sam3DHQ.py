# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.init as init


from functools import partial

from .modeling import ImageEncoderViT3D, MaskDecoder3DHQ, PromptEncoder3D, Sam3D

def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def custom_initialization(model):
    """Applies custom initialization to uninitialized layers of the model."""
    # This function can be extended to apply different initializations based on layer types.
    model.apply(init_weights)

def initialize_missing_components(model, state_dict):
    """Initialize missing components in the model not covered by the state dict."""
    model_state_dict = model.state_dict()

    # Identify missing and unexpected keys
    missing_keys = [key for key in model_state_dict.keys() if key not in state_dict]
    unexpected_keys = [key for key in state_dict.keys() if key not in model_state_dict]

    # Report missing and unexpected keys (optional)
    print(f"Missing keys in state dict: {missing_keys}")
    print(f"Unexpected keys in state dict: {unexpected_keys}")

    # Apply custom initialization to the whole model, it won't re-initialize loaded weights
    custom_initialization(model)
    
def build_sam3D_hq_vit_h(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam3D_hq = build_sam3D_hq_vit_h


def build_sam3D_hq_vit_l(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam3D_hq_vit_b(checkpoint=None):
    return _build_sam3D(
        # encoder_embed_dim=768,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam3D_hq_vit_b_ori(checkpoint=None):
    return _build_sam3D_hq_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry3D_hq = {
    "default": build_sam3D_hq_vit_h,
    "vit_h": build_sam3D_hq_vit_h,
    "vit_l": build_sam3D_hq_vit_l,
    "vit_b": build_sam3D_hq_vit_b,
    "vit_b_ori": build_sam3D_hq_vit_b_ori,
}



def _build_sam3D_hq(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    hq_token_only=False
):
    prompt_embed_dim = 384
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    interm_embeddings=None
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
       mask_decoder=MaskDecoder3DHQ(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=1024
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_sam3D_hq_ori(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 384
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3DHQ(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.mask_decoder.hf_token.apply(init_weights)
    sam.mask_decoder.hf_mlp.apply(init_weights)
    sam.mask_decoder.compress_vit_feat.apply(init_weights)
    sam.mask_decoder.embedding_encoder.apply(init_weights)
    sam.mask_decoder.embedding_maskfeature.apply(init_weights)
    sam.eval()
    # Load checkpoint if provided
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # Using strict=False allows ignoring unmatched keys, facilitating the addition of new components
        sam.load_state_dict(state_dict, strict=False)
        initialize_missing_components(model, checkpoint['model_state_dict'])



    # Freeze selected parameters
    for name, param in sam.named_parameters():
        if not any(n in name for n in ['hf_token', 'hf_mlp', 'compress_vit_feat', 'embedding_encoder', 'embedding_maskfeature']):
            param.requires_grad = False
    return sam
