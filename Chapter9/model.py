import torch
from torch import nn


class PolicyNet(nn.Module):

    def __init__(self, input_dim=2, output_dim=5):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)
