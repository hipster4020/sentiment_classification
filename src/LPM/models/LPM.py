import torch
import torch.nn as nn


class loss_prediction_module(nn.Module):
    def __init__(self, seq_len, outputs_size, num_layers):
        super(loss_prediction_module, self).__init__()
        self.num_layers = num_layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(seq_len, outputs_size) for _ in range(num_layers)]
        )
        self.final = nn.Linear((outputs_size * num_layers), 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        result = []
        for i in range(self.num_layers):
            layer = self.linear_layers[i]

            r = self.avgpool(x[i])
            r = torch.flatten(r, 1)
            r = layer(r)
            r = self.relu(r)
            result.append(r)

        final = torch.cat(result, dim=1)
        final = self.final(final)
        return final
