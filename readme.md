# Fishnet

PyTorch implementation of `FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction`


### [[arxiv]](http://arxiv.org/abs/1901.03495) [[Official Code]](https://github.com/kevin-ssy/FishNet)

Authors : [Jihyeong Yoo](https://github.com/YooJiHyeong), [Daewoong Ahn](https://github.com/zsef123)

<hr>

 # Notes
There might be some difference between Original Paper and its original implementation

Original implementation seems to have features below:
 1. In ResBlock of FishTail, bn_relu_conv shortcut is used
 2. In ResBlock of FishHead, bn_relu_conv shortcut is not used

However, according the original paper, probably, they should be like below:
 1. bn_relu_conv shortcut should not be used in ResBlock of FishTail
 2. bn_relu_conv shortcut should be used in ResBlock of FishHead
