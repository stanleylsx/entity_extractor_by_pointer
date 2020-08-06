from transformers import *
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(512, eps=1e-12)
        self.fc_s = nn.Linear(512, 4)
        self.fc_e = nn.Linear(512, 4)

    def forward(self, sentences):
        batch_hidden_states, batch_attentions = self.bert(sentences)[-2:]
        batch_hidden = self.layer_norm(batch_hidden_states)
        start = self.fc_s(batch_hidden)
        end = self.fc_e(batch_hidden)
        return [start, end]
