from abc import ABC
from torch import nn


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.fc_s = nn.Linear(768, 4)
        self.fc_e = nn.Linear(768, 4)

    def forward(self, batch_hidden_states):
        batch_hidden = self.layer_norm(batch_hidden_states)
        start = self.fc_s(batch_hidden)
        end = self.fc_e(batch_hidden)
        return start, end
