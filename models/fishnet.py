import math

import torch.nn as nn

from fish_module import FishHead, FishBody, FishTail, Bridge

# https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py#L197
def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))

class Fishnet(nn.Module):
    """
    Construct entire networks
    
    Args:
        start_c : Number of channels of input image
                  Note that it is NOT the number of channels in initial input image,
                            and it IS the number of output channel of stem
        num_cls : Number of classes
        stride : Stride of middle conv layer
        tail_num_blk : list of the numbers of Conv blocks in each FishTail stages
        body_num_blk : list of the numbers of Conv blocks in each FishBody stages
        head_num_blk : list of the numbers of Conv blocks in each FishHead stages
            (Note : `*_num_blk` includes 1 Residual blocks in the start of each stages)
        body_num_trans : list of the numbers of Conv blocks in transfer paths in each FishTail stages
        head_num_trans : list of the numbers of Conv blocks in transfer paths in each FishHead stages
        tail_channels : list of the number of in, out channel of each stages        
        body_channels : list of the number of in, out channel of each stages
        head_channels : list of the number of in, out channel of each stages

    """
    def __init__(self, start_c=64, num_cls=1000,
                 tail_num_blk=[], bridge_num_blk=2,
                 body_num_blk=[], body_num_trans=[],
                 head_num_blk=[], head_num_trans=[],
                 tail_channels=[], body_channels=[], head_channels=[]):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(3, start_c//2, stride=2),
            _conv_bn_relu(start_c//2, start_c//2),
            _conv_bn_relu(start_c//2, start_c),
            nn.MaxPool2d(3, padding=1, stride=2)
        )

        print("FishNet Initialzation Start")
        
        self.tail_layer = nn.ModuleList()
        for i, num_blk in enumerate(tail_num_blk):            
            layer = FishTail(tail_channels[i], tail_channels[i+1], num_blk)
            self.tail_layer.append(layer)

        self.bridge = Bridge(tail_channels[-1], bridge_num_blk)

        # First body module is not change feature map channel
        self.body_layer = nn.ModuleList()
        for i, (num_blk, num_trans) in enumerate(zip(body_num_blk, body_num_trans)):
            layer = FishBody(body_channels[i][0], body_channels[i][1], num_blk, 
                             tail_channels[-i-2], num_trans, dilation=2**i)
            self.body_layer.append(layer)

        self.head_layer = nn.ModuleList()
        for i, (num_blk, num_trans) in enumerate(zip(head_num_blk, head_num_trans)):
            layer = FishHead(head_channels[i][0], head_channels[i][1], num_blk,
                             body_channels[-i-1][0], num_trans)
            self.head_layer.append(layer)

        last_c = head_channels[-1][1]
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(last_c),
            nn.ReLU(True),
            nn.Conv2d(last_c, last_c//2, 1, bias=False),
            nn.BatchNorm2d(last_c//2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(last_c // 2, num_cls, 1, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        stem = self.stem(x)
        tail_features = [stem]
        for t in self.tail_layer:
            last_feature = tail_features[-1]
            tail_features += [ t(last_feature) ]

        bridge = self.bridge(tail_features[-1])

        body_features = [bridge]
        for b, tail in zip(self.body_layer, reversed(tail_features[:-1])):
            last_feature = body_features[-1]
            body_features += [ b(last_feature, tail) ]

        head_features = [body_features[-1]]
        for h, body in zip(self.head_layer, reversed(body_features[:-1])):
            last_feature = head_features[-1]            
            head_features += [ h(last_feature, body) ]

        out = self.classifier(head_features[-1])
        return out
