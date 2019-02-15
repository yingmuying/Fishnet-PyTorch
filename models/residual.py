import torch.nn as nn

def _bn_relu_conv(in_c, out_c, **conv_kwargs):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        nn.ReLU(True),
        nn.Conv2d(in_c, out_c, conv_kwargs),
    )

class _ConvBlock(nn.Module):
    """
    Construct Basic Bottleneck Convolution Block module.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer

    Forwarding Path:
        input image - (BN-ReLU-Conv) * 3 - output
    """
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super(_ConvBlock, self).__init__()

        mid_c = out_c // 4
        self.layers = nn.Sequential(
            _bn_relu_conv(in_c, mid_c, kernel_size=1, bias=False),
            _bn_relu_conv(mid_c, mid_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            _bn_relu_conv(mid_c, out_c, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.layers(x)

class TransferBlock(nn.Module):
    """
    Construct Transfer Block module.
    
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks

    Forwarding Path:
        input image - (ConvBlock) * num_blk - output
    """
    def __init__(self, ch, num_blk):
        super().__init__()

        self.layers = nn.Sequential(
            *[_ConvBlock(ch, ch) for _ in range(0, num_blk)]
        )

    def forward(self, x):
        return self.layers(x)


class DownRefinementBlock(nn.Module):
    """
    Construct Down-RefinementBlock module. (DRBlock from the original paper)
    Consisted of one Residual Block and Conv Blocks.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer

    Forwarding Path:
                    ⎡      (BN-ReLU-Conv)     ⎤
        input image - (ConvBlock) * num_blk -(sum)- feature - (MaxPool) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1):
        super().__init__()

        self.res = _ConvBlock(in_c, out_c)
        self.regular_connection = nn.Sequential(
            *[_ConvBlock(out_c, out_c) for _ in range(1, num_blk)]
        )
        self.shortcut = _bn_relu_conv(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        out = self.res(x)
        shortcut = self.shortcut(x)
        out = self.regular_connection(out + shortcut)
        return self.pool(out)


class UpRefinementBlock(nn.Module):
    """
    Construct Up-RefinementBlock module. (URBlock from the original paper)
    Consisted of Residual Block and Conv Blocks.
    Not like DRBlock, this module reduces the number of channels of concatenated feature maps in the shortcut path.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer

    Forwarding Path:
                    ⎡      (BN-ReLU-Conv)     ⎤
        input image - (ConvBlock) * num_blk -(sum)- feature - (UpSample) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1, dilation=1):
        super().__init__()
        self.k = in_c // out_c
        self.res = _ConvBlock(in_c, out_c, dilation=dilation)
        self.regular_connection = nn.Sequential(
            *[_ConvBlock(out_c, out_c, dilation=dilation) for _ in range(1, num_blk)]
        )
      
        self.upsample = nn.Upsample(scale_factor=2)

    def channel_reduction(self, x):
        n, c, *dim = x.shape
        return x.view(n, c // self.k, self.k, *dim).sum(2)
        
    def forward(self, x):
        out = self.res(x)
        shortcut = self.channel_reduction(x)
        out = self.regular_connection(out + shortcut)
        return self.upsample(out)