import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.annotated_attention import *


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout,
                 intermediate_layer_losses=True, generator=None, max_sequence_len=512):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.add_positional_encoding = AddPositionalEncoding(size, max_sequence_len)

        self.size = size
        self.intermediate_layer_losses = intermediate_layer_losses
        if intermediate_layer_losses and self.training:
            self.classifier = copy.deepcopy(generator)

    def forward(self, x, mask):
        x = self.add_positional_encoding(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        if self.intermediate_layer_losses and self.training:
            return x, self.classifier(x)
        else:
            return x, None


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n_layers, intermediate_layer_losses=True):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        self.intermediate_layer_losses = intermediate_layer_losses

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        intermediate_predictions = []
        for layer in self.layers:
            x, prediction = layer(x, mask)
            intermediate_predictions.append(prediction)
        return self.norm(x), intermediate_predictions


class MultiLayerCrossEntropy(nn.Module):
    def __init__(self, vocab_size, *args, **kwargs):
        super(MultiLayerCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, inputs, target):
        total_loss = torch.zeros(1, dtype=inputs[0].dtype, device=inputs[0].device)
        for input in inputs:
            if input is not None:
                loss = self.cross_entropy(input.view(-1, self.vocab_size).contiguous(), target)
                total_loss += loss
        return total_loss, loss


class NextCharTransformer(nn.Module):
    """
    A standard next-character prediction model. Base for this and many
    other models.
    """
    def __init__(self, vocab_size, n_layers=64,
                 hidden_size=512, inner_linear=2048,
                 n_heads=8, dropout=0.55, tied=True, max_sequence_len=512,
                 intermediate_layer_losses=True):
        super(NextCharTransformer, self).__init__()

        attn = MultiHeadedAttention(n_heads, hidden_size)
        ff = PositionwiseFeedForward(hidden_size, inner_linear, dropout)
        position = PositionalEncoding(hidden_size, dropout)

        self.generator = Generator(hidden_size, vocab_size)
        self.encoder = Encoder(EncoderLayer(hidden_size, copy.deepcopy(attn), copy.deepcopy(ff),
                                            dropout, intermediate_layer_losses, self.generator,
                                            max_sequence_len),
                               n_layers)
        self.embed = nn.Sequential(Embeddings(hidden_size, vocab_size), copy.deepcopy(position))

        self.criterion = MultiLayerCrossEntropy(vocab_size)
        # self.criterion = nn.CrossEntropyLoss()

        # use weight sharing
        if tied:
            self.generator.proj.weight = self.src_embed.lut.weight

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.vocab_size = vocab_size
        self.intermediate_layer_losses = intermediate_layer_losses
        self.n_layers = n_layers

    def forward(self, src, mask):
        """Take in and process masked src and target sequences."""
        src_emb = self.embed(src)
        emb, intermediate_predictions = self.encoder(src_emb, mask)
        if self.intermediate_layer_losses and self.training:
            return intermediate_predictions
        else:
            prediction = self.generator(emb)
            return [prediction]

    def update(self, training_percent):
        """Stop using losses from intermediate layer as function of time in training.
           See section 2.1 - Intermediate Layer Losses
        """
        for i, layer in enumerate(self.encoder.layers[:-1]):
            if training_percent > (i // (2 * self.n_layers)):
                layer.intermediate_layer_losses = False


def next_char_transformer(src_vocab, n_layers=64, hidden_size=512,
                          inner_linear=2048, n_heads=8, dropout=0.55,
                          tied=True, max_sequence_len=512, intermediate_losses=True):
    return NextCharTransformer(src_vocab,
                               n_layers, hidden_size,
                               inner_linear, n_heads,
                               dropout, tied, max_sequence_len, intermediate_losses)
