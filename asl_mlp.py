import torch
import torch.nn as nn

class ASL_MLP(nn.Module):
    def __init__(self, in_dim=63, num_classes=24):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)