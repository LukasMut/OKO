#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import flax.linen as nn
import jax.numpy as jnp

from .triplet import TripletHead


Array = jnp.ndarray


def img_to_patch(x, patch_size, flatten_channels=True) -> Array:
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, H', W', p_H, p_W, C]
    x = x.reshape(B, -1, *x.shape[3:])  # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1)  # [B, H'*W', p_H*p_W*C]
    return x


class AttentionBlock(nn.Module):
    embed_dim: int  # Dimensionality of input and attention feature vectors
    hidden_dim: int  # Dimensionality of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in the Multi-Head Attention block
    dropout_prob: float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim),
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    @nn.compact
    def __call__(self, x, train=True) -> Array:
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic=not train)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        return x


class ViT(nn.Module):
    embed_dim: int  # Dimensionality of input and attention feature vectors
    hidden_dim: int  # Dimensionality of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in the Multi-Head Attention block
    num_channels: int  # Number of channels of the input (3 for RGB)
    num_classes: int  # Number of classes to predict
    k: int # number of odd classes in a set
    num_layers: int  # Number of layers to use in the Transformer
    patch_size: int  # Number of pixels that the patches have per dimension
    num_patches: int  # Maximum number of patches an image can have
    dropout_prob: float = 0.0  # Amount of dropout to apply in the feed-forward network
    capture_intermediates: bool = False  # Capture intermediate activations

    def setup(self):
        # Layers/Networks
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [
            AttentionBlock(
                self.embed_dim, self.hidden_dim, self.num_heads, self.dropout_prob
            )
            for _ in range(self.num_layers)
        ]
        self.dropout = nn.Dropout(self.dropout_prob)

        # Parameters/Embeddings
        self.cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=1.0), (1, 1, self.embed_dim)
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, 1 + self.num_patches, self.embed_dim),
        )

        self.head = TripletHead(
                backbone="vit",
                num_classes=self.num_classes,
                k=self.k,
        )

    @nn.compact
    def __call__(
        self,
        x: Array,
        train=True,
    ) -> Array:
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)

        if train:
            # use positional encodings for training
            cls = x[:, 1:, :].mean(axis=1)
        else:
            # use classification token for inference
            cls = x[:, 0]

        if self.capture_intermediates:
            self.sow("intermediates", "latent_reps")
        out = self.head(cls, train)
        return out
