import torch.nn as nn

from models.fish_module import FishHead, FishBody, FishTail


class Fishnet(nn.Module):
    def __init__(self, in_c=64, grow_rate=2,
                 tails_res=(), bodys_res=(), heads_res=(),
                 bodys_trans=(), heads_trans=()):

        self.stem = nn.Sequential(
            nn.Conv2d(3, in_c//2, 3, stride=2),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(True),

            nn.Conv2d(in_c//2, in_c//2),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(True),

            nn.Conv2d(in_c//2, in_c),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(True),
            nn.MaxPool2d(3, padding=1, stride=2)
        )
        
        self.tails, self.tails_ch = [], []
        for i in range(self.num_stage):
            # tails_ch = [in_c, in_c * 2, in_c * 4 .... ]
            tail_ch = in_c * (grow_rate ** i)
            self.tails.append(FishTail(tail_ch, tail_ch * grow_rate, tails_res[i]))
            self.tails_ch.append(tail_ch)
        
        bridge_ch = tail_ch * grow_rate
        self.bridge = None

        self.bodys_ch = [bridge_ch, bridge_ch + self.tails_ch[-1]]
        self.bodys = [FishBody(bridge_ch, bridge_ch, bodys_res[0], 
                               self.tails_ch[-1], bodys_trans[0])]
        for i in range(1, self.num_stage):
            transfer_ch = self.tails_ch[-i]
            body_ch = self.bodys_ch[-1] # 768 -> 512 -> 320
            self.bodys.append(FishBody(body_ch, body_ch // grow_rate, tails_res[i],
                                       transfer_ch, bodys_trans[i]))
            self.bodys_ch.append(body_ch // grow_rate + transfer_ch)
        
        
        head_ch = body_ch
        self.heads = []
        for i in range(self.num_stage):
            transfer_ch = self.bodys_ch[-i]
            self.heads.append(FishHead(head_ch, head_ch, heads_res[i],
                                       transfer_ch, heads_trans[i]))
            head_ch += transfer_ch

        # --------------------------------------------- #
            
        self.tail1 = FishTail(in_c,   in_c*2, 2)
        self.tail2 = FishTail(in_c*2, in_c*4, 4)
        self.tail3 = FishTail(in_c*4, in_c*8, 8)

        # TODO : Bridge SE Block
        self.bridge = None
        
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


# ---------------- #

    def forward2(self, x):
        stem = self.stem(x)
        tail_features = [stem]
        for t in self.tails:
            last_feature = tail_features[-1]
            tail_features += [ t(last_feature) ]

        bridge = self.bridge(tail_features[-1])
        body_features = [bridge]
        for b, tail in zip(self.bodys, reversed(tail_features[:-1])):
            last_feature = body_features[-1]
            body_features += [ b(last_feature, tail) ]

        head_features = [body_features[-1]]
        for h, body in zip(self.bodys, reversed(body_features[:-1])):
            last_feature = head_features[-1]
            head_features += [ h(last_feature, body) ]

        out = self.classifier(head_features[-1])
        return out