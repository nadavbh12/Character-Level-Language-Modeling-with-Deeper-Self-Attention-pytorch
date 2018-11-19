import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.annotated_attention import *


class NextCharTransformer(nn.Module):
    """
    A standard next-character prediction model. Base for this and many
    other models.
    """
    def __init__(self, encoder, embed, generator):
        super(NextCharTransformer, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.generator = generator

    def forward(self, src, target, src_mask, target_mask):
        """Take in and process masked src and target sequences."""
        src_emb = self.embed(src)
        target_emb = self.embed(target)
        return self.encoder(src_emb, src_mask, src_mask, target_emb, target_mask)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def make_model(src_vocab, tgt_vocab, n_layers=64,
               hidden_size=512, inner_linear=2048,
               n_heads=8, dropout=0.1, tied=True):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, hidden_size)
    ff = PositionwiseFeedForward(hidden_size, inner_linear, dropout)
    position = PositionalEncoding(hidden_size, dropout)
    model = NextCharTransformer(
        Encoder(EncoderLayer(hidden_size, c(attn), c(ff), dropout), n_layers),
        nn.Sequential(Embeddings(hidden_size, src_vocab), c(position)),
        Generator(hidden_size, tgt_vocab)
    )

    # use weight sharing
    if tied:
        model.Generator.proj.weight = model.src_embed.lut.weight

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
