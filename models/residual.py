import torch.nn as nn


# Regular Connection
class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super(_ResBlock, self).__init__()

        mid_c = out_c // 4
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.layer(x)

class TransferBlock(nn.Module):
    def __init__(self, in_c, num_res):
        layers = []
        for i in range(0, num_res):
            layers.append(_ResBlock(in_c, in_c))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Down-sampling & Refinement Block(DR Block)
class DownRefinementBlock(nn.Module):
    def __init__(self, in_c, out_c, num_res, stride=1):
        super().__init__()

        self.res = _ResBlock(in_c, out_c)
        self.regular_connection = [_ResBlock(out_c, out_c) for _ in range(1, num_res)]
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        )
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        out = self.res(x)
        shortcut = self.shortcut(x)
        out = self.regular_connection(out + shortcut)
        return self.pool(out)


# Up-sampling & Refinement Block(UR Block)
class UpRefinementBlock(nn.Module):
    def __init__(self, in_c, out_c, num_res, stride=1, k=1, dilation=1):
        super().__init__()
        
        self.k = k
        self.res = _ResBlock(in_c, out_c, dilation=dilation)
        self.regular_connection = [_ResBlock(out_c, out_c, dilation=dilation) for _ in range(1, num_res)]
      
        self.upsample = nn.Upsample(scale_factor=2)

    def channel_reduction(self, x):
        n, c, *dim = x.shape
        return x.view(n, c // self.k, self.k, *dim).sum(2)
        
    def forward(self, x):
        out = self.res(x)
        shortcut = self.channel_reduction(x)        
        out = self.regular_connection(out + shortcut)
        return self.upsample(out + shortcut)
