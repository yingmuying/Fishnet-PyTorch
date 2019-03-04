import torch.nn as nn


def _bn_relu_conv(in_c, out_c, **conv_kwargs):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        nn.ReLU(True),
        nn.Conv2d(in_c, out_c, **conv_kwargs),
    )


class ResBlock(nn.Module):
    """
    Construct Basic Bottle-necked Residual Block module.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        shortcut : Specific function for skip-connection
            Examples)
                'bn_relu_conv' for DownRefinementBlock 
                'bn_relu_conv with channel reduction' for UpRefinementBlock
                'identity mapping' for regular connection
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer

    Forwarding Path:
                ⎡        (shortcut)         ⎤
        input image - (BN-ReLU-Conv) * 3 - (add) -output
    """
    def __init__(self, in_c, out_c, shortcut=lambda x: x,
                 stride=1, dilation=1):
        super(ResBlock, self).__init__()

        mid_c = out_c // 4
        self.layers = nn.Sequential(
            _bn_relu_conv(in_c, mid_c,  kernel_size=1, bias=False),
            _bn_relu_conv(mid_c, mid_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            _bn_relu_conv(mid_c, out_c, kernel_size=1, bias=False),
        )

        self.shortcut = shortcut

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class TransferBlock(nn.Module):
    """
    Construct Transfer Block module.
    
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks

    Forwarding Path:
        input image - (ResBlock) * num_blk - output
    """
    def __init__(self, ch, num_blk):
        super().__init__()

        self.layers = nn.Sequential(
            *[ResBlock(ch, ch) for _ in range(0, num_blk)]
        )

    def forward(self, x):
        return self.layers(x)


class DownStage(nn.Module):
    """
    Construct a stage for each resolution.
    A DownStage is consisted of one DownRefinementBlock and several residual regular connection blocks.

    (Note: In fact, DownRefinementBlock is not used in FishHead according to original implementation.
           However, it seems needed to be used according to original paper.
           In this version, we followed original implementation, not original paper.)

    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer

    Forwarding Path:
        input image - (ResBlock with Shortcut) - (ResBlock) * num_blk - (MaxPool) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1):
        super().__init__()

        shortcut = _bn_relu_conv(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        self.layer = nn.Sequential(
            ResBlock(in_c, out_c, shortcut=shortcut),
            *[ResBlock(out_c, out_c) for _ in range(1, num_blk)],
            nn.MaxPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.layer(x)


class UpStage(nn.Module):
    """
    Construct a stage for each resolution.
    A DownStage is consisted of one DownRefinementBlock and several residual regular connection blocks.
    Not like DownStage, this module reduces the number of channels of concatenated feature maps in the shortcut path.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer

    Forwarding Path:
        input image - (ResBlock with Channel Reduction) - (ResBlock) * num_blk - (UpSample) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1, dilation=1):
        super().__init__()
        self.k = in_c // out_c

        self.layer = nn.Sequential(
            ResBlock(in_c, out_c, shortcut=self.channel_reduction, dilation=dilation),
            *[ResBlock(out_c, out_c, dilation=dilation) for _ in range(1, num_blk)],
            nn.Upsample(scale_factor=2)            
        )

    def channel_reduction(self, x):
        n, c, *dim = x.shape
        return x.view(n, c // self.k, self.k, *dim).sum(2)
        
    def forward(self, x):
        return self.layer(x)