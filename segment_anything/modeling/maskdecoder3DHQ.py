# import torch
# from torch import Tensor, nn
# from typing import Tuple, Type

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import torch
# from torch import nn
# from torch.nn import functional as F

# from typing import List, Tuple, Type
# # from .transformer import TwoWayTransformer
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import torch
# from torch import Tensor, nn

# import math
# from typing import Tuple, Type


# class MLPBlock3D(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
#         super(MLPBlock3D, self).__init__()
#         self.num_layers = num_layers
#         layers = [nn.Linear(input_dim, hidden_dim)]
#         layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
#         layers += [nn.Linear(hidden_dim, output_dim)]
        
#         self.layers = nn.ModuleList(layers)
#         self.sigmoid_output = sigmoid_output

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < self.num_layers - 1:
#                 x = F.relu(x)
#             elif self.sigmoid_output:
#                 x = torch.sigmoid(x)
#         return x
# class TwoWayTransformer3D(nn.Module):
#     def __init__(
#         self,
#         depth: int,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#     ) -> None:
#         """
#         A transformer decoder that attends to an input image using
#         queries whose positional embedding is supplied.

#         Args:
#           depth (int): number of layers in the transformer
#           embedding_dim (int): the channel dimension for the input embeddings
#           num_heads (int): the number of heads for multihead attention. Must
#             divide embedding_dim
#           mlp_dim (int): the channel dimension internal to the MLP block
#           activation (nn.Module): the activation to use in the MLP block
#         """
#         super().__init__()
#         self.depth = depth
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.mlp_dim = mlp_dim
#         self.layers = nn.ModuleList()

#         for i in range(depth):
#             self.layers.append(
#                 TwoWayAttentionBlock3D(
#                     embedding_dim=embedding_dim,
#                     num_heads=num_heads,
#                     mlp_dim=mlp_dim,
#                     activation=activation,
#                     attention_downsample_rate=attention_downsample_rate,
#                     skip_first_layer_pe=(i == 0),
#                 )
#             )

#         self.final_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm_final_attn = nn.LayerNorm(embedding_dim)

#     def forward(
#         self,
#         image_embedding: Tensor,
#         image_pe: Tensor,
#         point_embedding: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#           image_embedding (torch.Tensor): image to attend to. Should be shape
#             B x embedding_dim x h x w for any h and w.
#           image_pe (torch.Tensor): the positional encoding to add to the image. Must
#             have the same shape as image_embedding.
#           point_embedding (torch.Tensor): the embedding to add to the query points.
#             Must have shape B x N_points x embedding_dim for any N_points.

#         Returns:
#           torch.Tensor: the processed point_embedding
#           torch.Tensor: the processed image_embedding
#         """
#         # BxCxHxW -> BxHWxC == B x N_image_tokens x C
#         bs, c, x, y, z = image_embedding.shape
#         image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
#         image_pe = image_pe.flatten(2).permute(0, 2, 1)

#         # Prepare queries
#         queries = point_embedding
#         keys = image_embedding

#         # Apply transformer blocks and final layernorm
#         for layer in self.layers:
#             queries, keys = layer(
#                 queries=queries,
#                 keys=keys,
#                 query_pe=point_embedding,
#                 key_pe=image_pe,
#             )

#         # Apply the final attention layer from the points to the image
#         q = queries + point_embedding
#         k = keys + image_pe
#         attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm_final_attn(queries)

#         return queries, keys


# class TwoWayAttentionBlock3D(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int = 2048,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#         skip_first_layer_pe: bool = False,
#     ) -> None:
#         """
#         A transformer block with four layers: (1) self-attention of sparse
#         inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
#         block on sparse inputs, and (4) cross attention of dense inputs to sparse
#         inputs.

#         Arguments:
#           embedding_dim (int): the channel dimension of the embeddings
#           num_heads (int): the number of heads in the attention layers
#           mlp_dim (int): the hidden dimension of the mlp block
#           activation (nn.Module): the activation of the mlp block
#           skip_first_layer_pe (bool): skip the PE on the first layer
#         """
#         super().__init__()
#         self.self_attn = Attention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)

#         self.cross_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.mlp = MLPBlock3D(input_dim=embedding_dim, hidden_dim=mlp_dim, output_dim=embedding_dim, num_layers=3, sigmoid_output=False)
#         self.norm3 = nn.LayerNorm(embedding_dim)
#         self.norm4 = nn.LayerNorm(embedding_dim)
#         self.cross_attn_image_to_token = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )

#         self.skip_first_layer_pe = skip_first_layer_pe

#     def forward(
#         self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
#     ) -> Tuple[Tensor, Tensor]:
#         # Self attention block
#         if self.skip_first_layer_pe:
#             queries = self.self_attn(q=queries, k=queries, v=queries)
#         else:
#             q = queries + query_pe
#             attn_out = self.self_attn(q=q, k=q, v=queries)
#             queries = queries + attn_out
#         queries = self.norm1(queries)

#         # Cross attention block, tokens attending to image embedding
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm2(queries)

#         # MLP block
#         mlp_out = self.mlp(queries)
#         queries = queries + mlp_out
#         queries = self.norm3(queries)

#         # Cross attention block, image embedding attending to tokens
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
#         keys = keys + attn_out
#         keys = self.norm4(keys)

#         return queries, keys


# class Attention(nn.Module):
#     """
#     An attention layer that allows for downscaling the size of the embedding
#     after projection to queries, keys, and values.
#     """

#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         downsample_rate: int = 1,
#     ) -> None:
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.internal_dim = embedding_dim // downsample_rate
#         self.num_heads = num_heads
#         assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

#         self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

#     def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
#         b, n, c = x.shape
#         x = x.reshape(b, n, num_heads, c // num_heads)
#         return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

#     def _recombine_heads(self, x: Tensor) -> Tensor:
#         b, n_heads, n_tokens, c_per_head = x.shape
#         x = x.transpose(1, 2)
#         return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

#     def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
#         # Input projections
#         q = self.q_proj(q)
#         k = self.k_proj(k)
#         v = self.v_proj(v)

#         # Separate into heads
#         q = self._separate_heads(q, self.num_heads)
#         k = self._separate_heads(k, self.num_heads)
#         v = self._separate_heads(v, self.num_heads)

#         # Attention
#         _, _, _, c_per_head = q.shape
#         attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
#         attn = attn / math.sqrt(c_per_head)
#         attn = torch.softmax(attn, dim=-1)

#         # Get output
#         out = attn @ v
#         out = self._recombine_heads(out)
#         out = self.out_proj(out)

#         return out



# class LayerNorm3d(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
#         return x

# # Assuming LayerNorm3d and other necessary 3D operation classes are defined elsewhere

# class MaskDecoder3DHQ(nn.Module):
#     def __init__(
#         self,
#         *,
#         transformer_dim: int,
#         num_multimask_outputs: int = 3,
#         activation: Type[nn.Module] = nn.GELU,
#         iou_head_depth: int = 3,
#         iou_head_hidden_dim: int = 256,
#         vit_dim: int = 768,  # Added for SAMHQ
#     ) -> None:
#         super().__init__()
#         self.transformer_dim = transformer_dim
#         self.transformer = TwoWayTransformer3D(
#             depth=2,
#             embedding_dim=self.transformer_dim,
#             mlp_dim=2048,
#             num_heads=8,
#         )

#         self.num_multimask_outputs = num_multimask_outputs

#         # Original SAM tokens
#         self.iou_token = nn.Embedding(1, transformer_dim)
#         self.num_mask_tokens = num_multimask_outputs + 1
#         self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

#         # Added for SAMHQ: HQ-SAM parameters
#         self.hf_token = nn.Embedding(1, transformer_dim)  # HQ-Output-Token
#         self.hf_mlp = MLPBlock3D(transformer_dim, transformer_dim, transformer_dim // 8, 3)  # New MLP for HQ
#         self.num_mask_tokens += 1  # Adjust for the added HQ token

#         # Convolutional fusion layers for obtaining HQ-Feature, adapted for 3D
#         self.compress_vit_feat = nn.Sequential(
#             nn.ConvTranspose3d(vit_dim, transformer_dim, kernel_size=2, stride=2),
#             LayerNorm3d(transformer_dim),
#             activation(),
#             nn.ConvTranspose3d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
#         )



#         # Embedding encoder and mask feature layers adapted for 3D
#         self.embedding_encoder = nn.Sequential(
#             nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
#             LayerNorm3d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
#         )

#         self.embedding_maskfeature = nn.Sequential(
#             nn.Conv3d(transformer_dim // 8, transformer_dim // 4, 3, padding=1),
#             LayerNorm3d(transformer_dim // 4),
#             activation(),
#             nn.Conv3d(transformer_dim // 4, transformer_dim // 8, 3, padding=1)
#         )

#         # Original SAM upscaling and prediction heads
#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
#             LayerNorm3d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
#             activation(),
#         )
#         self.output_hypernetworks_mlps = nn.ModuleList(
#             [MLPBlock3D(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
#         )
#         self.iou_prediction_head = MLPBlock3D(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)
#     def forward(
#             self,
#             image_embeddings: torch.Tensor,
#             image_pe: torch.Tensor,
#             sparse_prompt_embeddings: torch.Tensor,
#             dense_prompt_embeddings: torch.Tensor,
#             multimask_output: bool,
#             hq_token_only: bool = False,  # Indicates if only HQ masks should be returned
#             interm_embeddings: torch.Tensor = None,  # Intermediate features for HQ processing
#         ) -> Tuple[torch.Tensor, torch.Tensor]:
#             """
#             Predict masks given image and prompt embeddings, adapted for 3DSAMHQ.
#             """
#             hq_features = None
#             # if interm_embeddings is None:
#             #    raise ValueError("interm_embeddings are missing")
#             if interm_embeddings is  None:

#                 print("There's no interm embeddings!")

#             # Adjust the dimensionality as per your model's requirement
#             if interm_embeddings is not None:
#                 # Process interm_embeddings to generate hq_features
                
#                 vit_features = interm_embeddings[0].permute(0,4, 1, 2,3)  
#                 vit_features = self.compress_vit_feat(vit_features)
#                 hq_features = self.embedding_encoder(image_embeddings) + vit_features

#             # Ensure hq_features were generated or handled correctly
#             if hq_features is None:
#                 raise ValueError("hq_features are required but were not generated.")

#             masks, iou_pred = self.predict_masks(
#                 image_embeddings=image_embeddings,
#                 image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_prompt_embeddings,
#                 dense_prompt_embeddings=dense_prompt_embeddings,
#                 hq_features=hq_features,  # Pass HQ features for mask prediction
#             )
        
#             # Adjustments for selecting mask outputs based on HQ features and multimask_output
#             if multimask_output:
#                 mask_slice = slice(1, self.num_mask_tokens-1)  # Adjust slice to exclude HQ token for multi-mask output
#             else:
#                 mask_slice = slice(0, 1)  # Default to the first mask token for single mask output
        
#             masks = masks[:, mask_slice, :, :, :]
        
#             # Special handling for HQ token output
#             if hq_features is not None and hq_token_only:
#                 # Assuming the last token is the HQ token
#                 masks = masks[:, -1:, :, :, :]  # Select only the HQ mask
        
#             iou_pred = iou_pred[:, mask_slice]
        
#             return masks, iou_pred

#     def predict_masks(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#         hq_features: torch.Tensor,  # Add this parameter
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks including the use of high-quality (HQ) features."""
#         output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
#         output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
#         tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

#         src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) + dense_prompt_embeddings
#         pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
#         b, c, x, y, z = src.shape

#         hs, src = self.transformer(src, pos_src, tokens)
#         iou_token_out, mask_tokens_out = hs[:, 0, :], hs[:, 1:(1 + self.num_mask_tokens), :]

#         hq_features_processed = self.embedding_encoder(image_embeddings) + hq_features.repeat(b,1,1,1,1)

#         src = src.transpose(1, 2).view(b, c, x, y, z)
#         upscaled_embedding = self.output_upscaling(src)
#         upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding) + hq_features_processed

#         # Generate masks from tokens
#         masks = self.generate_masks(mask_tokens_out, upscaled_embedding, upscaled_embedding_hq,src,b,c,x,y,z)

#         # Generate mask quality predictions
#         iou_pred = self.iou_prediction_head(iou_token_out)

#         return masks, iou_pred


#     def generate_masks(self, mask_tokens_out, upscaled_embedding, upscaled_embedding_hq,src,b,c,x,y,z):
#         """Generate standard and HQ masks, then combine them."""
#         hyper_in_list = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens - 1)]
#         hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, -1, :]))  # Process HQ token separately
#         hyper_in = torch.stack(hyper_in_list, dim=1)

        
#         upscaled_embedding_reshaped = upscaled_embedding.view(upscaled_embedding.shape[0], upscaled_embedding.shape[1], -1)
#         upscaled_embedding_hq_reshaped = upscaled_embedding_hq.view(upscaled_embedding_hq.shape[0], upscaled_embedding_hq.shape[1], -1)

        
#         masks_standard = torch.matmul(hyper_in[:, :-1], upscaled_embedding_reshaped)
#         masks_hq = torch.matmul(hyper_in[:, -1:], upscaled_embedding_hq_reshaped)

#         # Reshape back to original spatial dimensions for masks
#         masks_standard = masks_standard.view(upscaled_embedding.shape[0], -1, *upscaled_embedding.shape[2:])
#         masks_hq = masks_hq.view(upscaled_embedding_hq.shape[0], -1, *upscaled_embedding_hq.shape[2:])

#         # Combine standard and HQ masks for final output
#         masks_combined = torch.cat([masks_standard, masks_hq], dim=1)

#         return masks_combined




# # Lightly adapted from
# # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         self.sigmoid_output = sigmoid_output

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x
