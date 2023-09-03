import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tools import Linear_fw, Conv2d_fw

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = Linear_fw(d_model, h * d_k)
        self.fc_k = Linear_fw(d_model, h * d_k)
        self.fc_v = Linear_fw(d_model, h * d_v)
        self.fc_o = Linear_fw(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(~attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class seqEncoder(nn.Module):
    def __init__(self, input_dim, embd_size=128, head=2, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
        super().__init__()

        self.conv = Conv2d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
        self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=head, dropout=dropout)

    def forward(self, x, mask=None):
        ''' x: modality sequences. [batch_size, embd_size]'''
        x = torch.unsqueeze(x, dim=1)
        b, l, d = x.size()
        x = x.view(b, 1, l, d)
        hidden = F.relu(self.conv(x).squeeze(3)).transpose(1,2)
        attn_hidden = self.self_attn(hidden, hidden, hidden, attention_mask=mask)
        attn_hidden = torch.squeeze(attn_hidden, dim=1)
        return attn_hidden
