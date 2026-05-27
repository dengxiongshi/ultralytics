import torch
import torch.nn as nn


class ADD(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(ADD, self).__init__()
        # 128 256 512
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class RIFusion(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, r=16, dimension=1):
        super().__init__()
        self.c1 = c1 * 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.c1, self.c1 // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c1 // r, self.c1, bias=False),
            nn.Sigmoid()
            # nn.Sigmoid(inplace=True)
        )

    def forward(self, x):
        # return x
        b, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, self.c1)
        y = self.fc(y).view(b, self.c1, 1, 1)

        x1 = x * y
        return x + torch.cat((x1[:, self.c1 // 2:, ...], x1[:, :self.c1 // 2, ...]), dim=1)
