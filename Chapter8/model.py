import torch
import torch.nn as nn


class QNET(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)
