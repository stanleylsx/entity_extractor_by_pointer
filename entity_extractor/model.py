from abc import ABC
from torch import nn


class Model(nn.Module, ABC):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 2 * num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bert_hidden_states):
        layer_hidden = self.layer_norm(bert_hidden_states)
        fc1_results = self.fc_1(layer_hidden)
        fc2_results = self.fc_2(fc1_results)
        output = self.sigmoid(fc2_results)
        batch_size = output.size(0)
        transfer_output = output.view(batch_size, -1, self.num_labels, 2)
        return transfer_output
