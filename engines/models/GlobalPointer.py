# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel


class EffiGlobalPointer(nn.Module):
    def __init__(self, num_labels, device, rope=True):
        super(EffiGlobalPointer, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        self.device = device
        self.inner_dim = 64
        self.hidden_size = self.encoder.config.hidden_size
        self.RoPE = rope

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, num_labels * 2)

    def sinusoidal_position_embedding(self, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # embeddings = torch.reshape(embeddings, (-1, seq_len, output_dim)).to(self.device)
        embeddings = torch.reshape(embeddings, (-1, seq_len, output_dim))
        return embeddings

    @staticmethod
    def sequence_masking(x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state  # [2, 43, 768]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense_1(last_hidden_state)  # [2, 43, 128]
        # 取出q和k
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # [2, 43, 64] 从0,1开始间隔为2
        if self.RoPE:
            pos = self.sinusoidal_position_embedding(seq_len, self.inner_dim)
            # 是将奇数列信息抽取出来也就是cosm拿出来并复制
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            # 是将偶数列信息抽取出来也就是sinm拿出来并复制
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            # 奇数列加上负号 得到第二个q的矩阵
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = torch.reshape(qw2, qw.shape)
            # 最后融入位置信息
            qw = qw * cos_pos + qw2 * sin_pos
            # 奇数列加上负号 得到第二个q的矩阵
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = torch.reshape(kw2, kw.shape)
            # 最后融入位置信息
            kw = kw * cos_pos + kw2 * sin_pos
        # 最后计算初logits结果
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        dense_out = self.dense_2(last_hidden_state)
        dense_out = torch.einsum('bnh->bhn', dense_out) / 2
        # logits[:, None] 增加一个维度
        logits = logits[:, None] + dense_out[:, ::2, None] + dense_out[:, 1::2, :, None]
        logits = self.add_mask_tril(logits, mask=attention_mask)
        probs = torch.sigmoid(logits)
        return logits, probs
