import torch
import torch.nn as nn

from residual import DownRefinementBlock, TransferBlock, UpRefinementBlock


class FishTail(nn.Module):
    def __init__(self, in_c, out_c, num_res):
        super().__init__()
        self.layer = DownRefinementBlock(in_c, out_c, num_res)

    def forward(self, x):
        return self.layer(x)

  
class FishBody(nn.Module):
    def __init__(self, in_c, out_c, num_res,
                 trans_in_c, num_trans,
                 k=1, dilation=1):
        super().__init__()
        self.layer = UpRefinementBlock(in_c, out_c, num_res, k=k, dilation=dilation)
        self.transfer = TransferBlock(trans_in_c, num_trans)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)


class FishHead(nn.Module):
    def __init__(self, in_c, out_c, num_res,
                 trans_in_c, num_trans):
        super().__init__()
        self.layer = DownRefinementBlock(in_c, out_c, num_res)
        self.transfer = TransferBlock(trans_in_c, num_trans)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)

