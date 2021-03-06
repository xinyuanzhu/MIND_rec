import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(300, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(300, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(300, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        return q, attn


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):

    def __init__(
            self, n_head, d_k, d_v,
            d_model, n_layers=1, dropout=0.2):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.encoder_layer = EncoderLayer(
            d_model, n_head, d_k, d_v, dropout=dropout)
        self.add_att_1 = nn.Linear(d_model, 200)
        self.add_att_2 = nn.Linear(200, 1)

    def forward(self, emb_seq):

        enc_output = self.dropout(emb_seq.float())
        enc_output, enc_slf_attn = self.encoder_layer(
            enc_output)
        rep = F.tanh(self.add_att_1(enc_output))
        add_att_w = F.softmax(self.add_att_2(rep))
        out = torch.sum(enc_output*add_att_w, dim=1)
        return out
