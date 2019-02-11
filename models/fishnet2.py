import torch.nn as nn

from fish_module import FishHead, FishBody, FishTail


def fish150(self):
    net_cfg = {
        "tail_in_c":[64,  128,  256,  512],
        "body_in_c":[512, 768,  512,  320],
        "head_in_c":[320, 832, 1600, 2112],
    }

class Fishnet(nn.Module):
    def __init__(self, in_c=64, grow_rate=2, num_cls=1000,
                 tails_res=(), bodys_res=(), heads_res=(),
                 bodys_trans=(), heads_trans=()):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_c//2, 3, stride=2),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(True),
            nn.Conv2d(in_c//2, in_c//2, 3),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(True),
            nn.Conv2d(in_c//2, in_c, 3),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(True),
            nn.MaxPool2d(3, padding=1, stride=2)
        )
        

        #         0    1    2    3
        #        -4   -3   -2   -1
        tails = [64, 128, 256, 512]
        num_res = [4, 4, 4, 4]
        self.tail_layers = []
        for in_c, res in zip(tails, num_res):
            self.tail_layers.append(FishTail(in_c, in_c * 2, res))

        
        bodys = [tails[-1], tails[-2] + tails[-1]]
        self.bodys_layer = [FishBody(bodys[0], bodys[0], 1, tails[-2], 1),
                            FishBody(bodys[1], bodys[1]//2, 1, tails[-3], 1)]

        for i in range(2, len(tails)):
            body_in_c = bodys[-1] // 2 + tails[-i-1]
            body_out_c = bodys[i] // 2
            transfer_c = tails[-i-2]
            self.bodys_layer.append(FishBody(body_in_c, body_out_c, 0, transfer_c, 0))
            bodys.append(body_out_c + transfer_c)

       
        heads = [bodys[-1], bodys[-1] + bodys[-2]]
        self.heads_layer = [FishHead(heads[0], heads[0], 0, bodys[-2], 0),
                            FishHead(heads[1], heads[1], 0, bodys[-3], 0)]
        for i in range(1, len(tails) - 1):
            head_c = heads[-1] + bodys[i]
            transfer_c = bodys[-i-2]
            self.heads_layer.append(FishHead(head_c, head_c, 0, transfer_c, 0))
            heads.append(heads[-1] + bodys[i])



        self.classifier = nn.Sequential(
            nn.BatchNorm2d(heads[-1]),
            nn.ReLU(True),
            nn.Conv2d(heads[-1], heads[-1]//2, 1, bias=False),
            nn.BatchNorm2d(heads[-1]//2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(heads[-1] // 2, num_cls, 1, bias=True)
        )
        # --------------------------------------------- #
            
        self.tail1 = FishTail(in_c,   in_c*2, 2)
        self.tail2 = FishTail(in_c*2, in_c*4, 4)
        self.tail3 = FishTail(in_c*4, in_c*8, 8)

        # TODO : Bridge SE Block 
        self.bridge = lambda *x: None
        
        self.body1 = FishBody(in_c*8, in_c*8, 2,
                              in_c*4, 2) # Output channel of Tail2
        self.body2 = FishBody(in_c*8, in_c*8, 2,
                              in_c*2, 2) # Output channel of Tail1
        self.body3 = FishBody(in_c*8, in_c*8, 2,
                              in_c, 2) # Output channel of Stem
        
        # TODO : Check feature map channel
        self.head1 = FishHead(in_c*8, in_c*4, 2,
                              in_c*8, 2)
        self.head2 = FishHead(in_c*8, in_c*4, 2,
                              in_c*8, 2)
        self.head3 = FishHead(in_c*8, in_c*4, 2,
                              in_c*8, 2)

    def forward(self, x):
        stem = self.stem(x)
        tail1 = self.tail1(stem)
        tail2 = self.tail2(tail1)
        tail3 = self.tail3(tail2)
        bridge = self.bridge(tail3)

        body1 = self.body1(bridge, tail2)
        body2 = self.body2(body1, tail1)
        body3 = self.body3(body2, stem)

        head1 = self.head1(body3, body2)
        head2 = self.head2(head1, body2)
        head3 = self.head3(head2, body1)

        out = self.classifier(head3)
        return out

    def forward2(self, x):
        stem = self.stem(x)
        features = [stem]
        for layer in self.tails_layer:
            last_feature = features[-1]
            features += [ layer(last_feature) ]

        bridge = self.bridge(features[-1])
        features = [bridge]
        for layer, tail in zip(self.bodys, reversed(features[:-1])):
            last_feature = features[-1]
            features += [ layer(last_feature, tail) ]

        head_features = [features[-1]]
        for layer, body in zip(self.bodys, reversed(features[:-1])):
            last_feature = head_features[-1]
            head_features += [ layer(last_feature, body) ]

        out = self.classifier(head_features[-1])
        return out  

if __name__ == "__main__":
    a = Fishnet()