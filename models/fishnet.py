import torch.nn as nn

from fish_module import FishHead, FishBody, FishTail

def conv_bn_relu(in_c, out_c, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class Fishnet(nn.Module):
    def __init__(self, start_c=64, num_cls=1000,
                 tail_num_res=[],
                 body_num_res=[], body_num_trans=[],
                 head_num_res=[], head_num_trans=[]):
        super().__init__()

        self.stem = nn.Sequential(
            conv_bn_relu(3, start_c//2, 3, stride=2),
            conv_bn_relu(start_c//2, start_c//2, 3),
            conv_bn_relu(start_c//2, start_c, 3),
            nn.MaxPool2d(3, padding=1, stride=2)
        )
        print("Fishnet Init Start")
        
        self.tail_layer,  tail_c = nn.ModuleList(), [start_c]
        for i, num_blk in enumerate(tail_num_res):            
            in_c = tail_c[-1]
            layer = FishTail(in_c, in_c*2, num_blk)
            self.tail_layer.append(layer)
            tail_c.append(in_c*2)

        self.bridge = nn.Conv2d(tail_c[-1], tail_c[-1], 3, 1, 1)
        
        # Tail C :  [64, 128, 256, 512]
        self.body_layer, body_c = [], [tail_c[-1], tail_c[-1] + tail_c[-2]]
        # First body module is not change feature map channel
        self.body_layer = nn.ModuleList([FishBody(body_c[-2], body_c[-2], body_num_res[0],
                                    tail_c[-2], body_num_trans[0])])

        for i, (num_blk, num_trans) in enumerate(zip(body_num_res[1:], body_num_trans[1:]), start=1):
            in_c = body_c[-1]
            trans_c = tail_c[-i-2]
            layer = FishBody(in_c, in_c//2, num_blk, trans_c, num_trans, dilation=2**i)
            self.body_layer.append(layer)
            body_c.append(in_c//2 + trans_c)


        self.head_layer = nn.ModuleList()
        head_in_c = body_c[-1]
        for i, (num_blk, num_trans) in enumerate(zip(head_num_res, head_num_trans)):
            trans_c = body_c[-i-2]
            layer = FishHead(head_in_c, head_in_c, num_blk, trans_c, num_trans)
            self.head_layer.append(layer)
            head_in_c += trans_c

        last_c = head_in_c
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(last_c),
            nn.ReLU(True),
            nn.Conv2d(last_c, last_c//2, 1, bias=False),
            nn.BatchNorm2d(last_c//2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(last_c // 2, num_cls, 1, bias=True)
        )

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


def fish150():
    net_cfg = {
        # Input channel before FishTail
        "start_c" : 64,
        "num_cls" : 1000,

        "tail_num_res" : [2, 4, 8],

        "body_num_res" : [2, 2, 2],
        "body_num_trans" : [2, 2, 2],

        "head_num_res" : [2, 2, 2],
        "head_num_trans" : [2, 2, 2],
    }
    return Fishnet(**net_cfg)


if __name__ == "__main__":
    net = fish150()
    from torchsummary import summary
    summary(net, (3, 224, 224), device="cpu")