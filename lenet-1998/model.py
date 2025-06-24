
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 28x28 -> 24x24
            nn.Tanh(),
            nn.AvgPool2d(2),     # -> 12x12
            nn.Conv2d(6, 16, 5), # -> 8x8
            nn.Tanh(),
            nn.AvgPool2d(2),     # -> 4x4
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)
