import torch.nn as nn

from fish_module import FishHead, FishBody, FishTail

def conv_bn_relu(in_c, out_c, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class Fishnet(nn.Module):
    def __init__(self, in_c=64, num_cls=1000,
                 tail_c=[], tail_res=[],
                 body_c=[], body_res=[], body_trans=[],
                 head_c=[], head_res=[], head_trans=[]):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_relu(3, in_c//2, 3, stride=2),
            conv_bn_relu(in_c//2, in_c//2, 3),
            conv_bn_relu(in_c//2, in_c, 3),
            nn.MaxPool2d(3, padding=1, stride=2)
        )
        print("Fishnet Init")
        
        self.tails = []
        for (in_c, out_c), num_res in zip(tail_c, tail_res):
            print("tail : ", in_c, out_c, num_res)
            layer = FishTail(in_c, out_c, num_res)
            self.tails.append(layer)

        self.bridge = nn.Conv2d(tail_c[-1][1], tail_c[-1][1], 3, 1, 1)

        self.bodys = []
        for i, ((in_c, out_c), num_res, num_trans) in enumerate(zip(body_c, body_res, body_trans)):
            trans_c = tail_c[-i-1][0]
            print("body : ", in_c, out_c, num_res, trans_c, num_trans)
            layer = FishBody(in_c, out_c, num_res, trans_c, num_trans, dilation=2**i)
            self.bodys.append(layer)

        self.heads = []
        for i, ((in_c, out_c), num_res, num_trans) in enumerate(zip(head_c, head_res, head_trans)):
            trans_c = body_c[-i-1][0]
            print("head : ", in_c, out_c, num_res, trans_c, num_trans)
            layer = FishHead(in_c, out_c, num_res, trans_c, num_trans)
            self.heads.append(layer)
        
        # Last head output channel + First body input channel
        last_c = head_c[-1][1] + body_c[0][0]
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
        for t in self.tails:
            last_feature = tail_features[-1]
            tail_features += [ t(last_feature) ]
        print("Tail last feature : ", tail_features[-1].shape)
        bridge = self.bridge(tail_features[-1])
        body_features = [bridge]
        for b, tail in zip(self.bodys, reversed(tail_features[:-1])):
            last_feature = body_features[-1]
            print("Body feature :", [b.shape for b in body_features])
            body_features += [ b(last_feature, tail) ]

        head_features = [body_features[-1]]
        for h, body in zip(self.heads, reversed(body_features[:-1])):
            last_feature = head_features[-1]
            print("Head feature :", [b.shape for b in head_features])
            head_features += [ h(last_feature, body) ]

        out = self.classifier(head_features[-1])
        return out


def fish150():
    net_cfg = {
        "in_c":64,
        "num_cls":1000,

        "tail_c":[(64,  128), (128, 256), (256, 512)],
        "tail_res" : [2, 4, 8],

        "body_c":[(512, 512), (768, 384), (512, 256)],
        "body_res" : [2, 2, 2],
        "body_trans":[2, 2, 2],

        "head_c":[(320, 320), (832, 832), (1600, 1600)],
        "head_res" : [2, 2, 2],
        "head_trans" : [2, 2, 2],
    }
    return Fishnet(**net_cfg)


if __name__ == "__main__":
    net = fish150()
    from torchsummary import summary
    summary(net, (3, 224, 224), device="cpu")