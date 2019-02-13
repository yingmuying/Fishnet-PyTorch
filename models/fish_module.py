import torch
import torch.nn as nn

from residual import DownRefinementBlock, TransferBlock, UpRefinementBlock


class FishTail(nn.Module):
    """
    Construct FishTail module.
    Each instances corresponds to each stages.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks

    Forwarding Path:
        input image - (DRBlock) - output
    """
    def __init__(self, in_c, out_c, num_blk):
        super().__init__()
        self.layer = DownRefinementBlock(in_c, out_c, num_blk)

    def forward(self, x):
        return self.layer(x)

  
class FishBody(nn.Module):
    r"""Construct FishBody module.
    Each instances corresponds to each stages.
    
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        dilation : Dilation rate of Conv in UpRefinementBlock
        
    Forwarding Path:
        input image - (URBlock)   ⎤
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk,
                 trans_in_c, num_trans,
                 dilation=1):
        super().__init__()
        self.layer = UpRefinementBlock(in_c, out_c, num_blk, dilation=dilation)
        self.transfer = TransferBlock(trans_in_c, num_trans)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)

class FishHead(nn.Module):
    r"""Construct FishHead module.
    Each instances corresponds to each stages.
    
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        
    Forwarding Path:
        input image - (DRBlock)   ⎤
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk,
                 trans_in_c, num_trans):
        super().__init__()
        self.layer = DownRefinementBlock(in_c, out_c, num_blk)
        self.transfer = TransferBlock(trans_in_c, num_trans)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)

