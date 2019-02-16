# Fishnet

PyTorch implementation of `FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction`


### [[arxiv]](http://arxiv.org/abs/1901.03495) [[Official Code]](https://github.com/kevin-ssy/FishNet)

Authors : [Jihyeong Yoo](https://github.com/YooJiHyeong), [Daewoong Ahn](https://github.com/zsef123)

<hr>

 # Notes
There might be some difference between Original Paper and its original implementation

Original implementation seems to have features below:
 1. Regular connections have no residual shortcut (FishTail, TransferBlock, or Non-DRBlock/URBlock parts of FishBody/FishHead)
 2. In FishHead, Shortcut is not used in DRBlock

However, according the original paper, probably, they should be like below:
 1. Regular connections should have residual shortcuts
 2. In FishHead, Shortcut should be used in DRBlock
 
<hr>

## How to use:
```
```
