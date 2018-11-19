import torch.nn as nn
from modules.transformer_blocks import *


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab_size, hidden_size=512, emb_size=None, n_layers=64, n_heads=8,
                 dropout=0.5, inner_linear=2048, inner_groups=1, stateful=False,
                 tie_weights=False, layer_norm=True, weight_norm=False):
        super(TransformerModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.transformer = nn.ModuleList([TransformerLayer(hidden_size=hidden_size,
                                                           num_heads=n_heads,
                                                           inner_linear=inner_linear,
                                                           inner_groups=inner_groups,
                                                           layer_norm=layer_norm,
                                                           weight_norm=weight_norm,
                                                           dropout=dropout,
                                                           stateful=stateful) for _ in range(n_layers)])
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != emb_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.stateful = stateful

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, state, get_attention=False):
        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x))
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.hidden_size)
