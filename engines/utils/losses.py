# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : losses.py
# @Software: PyCharm
import torch


class MultilabelCategoricalCrossEntropy(torch.nn.Module):
    """
    苏神的多标签分类损失函数
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self):
        super(MultilabelCategoricalCrossEntropy, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = (neg_loss + pos_loss).mean()
        return loss
