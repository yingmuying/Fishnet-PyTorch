from fishnet import Fishnet

def _calc_channel(start_c, num_blk):
    """
    Calculate the number of in and out channels of each stages in FishNet.

    Example:
        fish150 : start channel=64, num_blk=3,
        tail channels : Grow double in each stages,
                        [64, 128, 256 ...] = [start channel ** (2**num_blk) ....] 
        body channels : In first stage, in_channel and out_channel is the same,
                        but the other layers, the number of output channels is half of the number of input channel
                        Add the number of transfer channels to the number of output channels
                        The numbers of transfer channels are reverse of the tail channel[:-2]
                        [(512, 512), + 256
                         (768, 384), + 128
                         (512, 256)] + 64
        head channels : The number of input channels and output channels is the same.
                        Add the number of transfer channels to the number of output channels
                        The numbers of transfer channels are reverse of the tail channel[:-2]
                        [(320, 320),   + 512
                         (832, 832),   + 768
                         (1600, 1600)] + 512

    """

    tail_channels = [start_c]
    for i in range(num_blk):
        tail_channels.append(tail_channels[-1] * 2)
    print("Tail Channels : ", tail_channels)

    in_c, transfer_c = tail_channels[-1], tail_channels[-2]
    body_channels = [(in_c, in_c), (in_c + transfer_c, (in_c + transfer_c)//2)]
    # First body module is not change feature map channel
    for i in range(1, num_blk-1):
        transfer_c = tail_channels[-i-2]
        in_c = body_channels[-1][1] + transfer_c
        body_channels.append((in_c, in_c//2))
    print("Body Channels : ", body_channels)

    in_c = body_channels[-1][1] + tail_channels[0]
    head_channels = [(in_c, in_c)]
    for i in range(num_blk):
        transfer_c = body_channels[-i-1][0]
        in_c = head_channels[-1][1] + transfer_c
        head_channels.append((in_c, in_c))
    print("Head Channels : ", head_channels)
    return {"tail_channels":tail_channels, "body_channels":body_channels, "head_channels":head_channels}
    
def fish99(num_cls=1000):
    start_c = 64

    tail_num_blk = [2, 2, 6]
    bridge_num_blk = 2

    body_num_blk = [1, 1, 1]
    body_num_trans = [1, 1, 1]

    head_num_blk = [1, 2, 2]
    head_num_trans = [1, 1, 4]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_cls, 
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)

def fish150(num_cls=1000):
    start_c = 64

    tail_num_blk = [2, 4, 8]
    bridge_num_blk = 4

    body_num_blk = [2, 2, 2]
    body_num_trans = [2, 2, 2]

    head_num_blk = [2, 2, 4]
    head_num_trans = [2, 2, 4]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_cls, 
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)


if __name__ == "__main__":
    net = fish150()
    from torchsummary import summary
    summary(net, (3, 224, 224), device="cpu")