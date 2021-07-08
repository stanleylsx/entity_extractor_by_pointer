# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from transformers import BertModel


class Model(nn.Module, ABC):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.fc = nn.Linear(hidden_size, 2 * num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sentences, attention_mask):
        bert_hidden_states = self.bert_model(sentences, attention_mask=attention_mask)[0]
        layer_hidden = self.layer_norm(bert_hidden_states)
        fc_results = self.fc(layer_hidden)
        output = self.sigmoid(fc_results)
        batch_size = output.size(0)
        transfer_output = output.view(batch_size, -1, self.num_labels, 2)
        return transfer_output
