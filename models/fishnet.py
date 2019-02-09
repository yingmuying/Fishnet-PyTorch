import torch.nn as nn

from models.fish_module import FishHead, FishBody, FishTail

class Fishnet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
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

    def forward2(self, x):
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