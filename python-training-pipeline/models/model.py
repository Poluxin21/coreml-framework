import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)