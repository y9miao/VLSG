# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mainly copy-paste from https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PatchAggregator(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        # transformer encoders
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # Initialize the [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        # x: (batch_size, num_patches, d_model)
        cls_tokens = self.cls_token.expand(-1, x.size(1), -1)
        src = torch.cat((cls_tokens, src), dim=0)
        output = self.transformer_encoder(src)
        cls_token_output = output[0, :, :]