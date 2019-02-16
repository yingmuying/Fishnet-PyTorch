import os
import argparse

import torch
import torch.nn as nn

from models import net_factory
from datas import get_dataloader
from utils import accuracy, AverageMeter
from Logger import Logger
logger = Logger(".")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', default='fish99',
                        choices=["fish99", "fish150"],
                        help='models architecture')
    parser.add_argument("--num_cls", default=1000, type=int,
                        help="output of the fishnet")
    parser.add_argument('--cpus', default=4, type=int,
                        help='number of data loading workers')

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")

    
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=32, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate models on validation set')

    parser.add_argument('--input_size', default=224, type=int, help='img crop size')
    parser.add_argument('--image_size', default=256, type=int, help='ori img size')

    parser.add_argument('--model_name', default='', type=str, help='name of the models')
    return parser.parse_args()

# def train(train_loader, model, criterion, optimizer, epoch):
def train(args, epoch, train_loader, net, loss_fn, optim, torch_device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    for i, (img, label) in enumerate(train_loader):
        img, label = img.to(torch_device), label.to(torch_device)

        # compute output
        output = net(img)

        loss = loss_fn(output, label)
        prec1, prec5 = accuracy(output.cpu(), label.cpu(), topk=(1, 5))

        # measure accuracy and record loss
        reduced_prec1 = prec1.clone()
        reduced_prec5 = prec5.clone()

        top1.update(reduced_prec1[0])
        top5.update(reduced_prec5[0])

        reduced_loss = loss.data.clone()
        losses.update(reduced_loss)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i % 20 == 0:
            logger.log_write("train", epoch=epoch,
                             loss_val=losses.val, loss_avg=losses,avg,
                             top1_val=top1.val, top1_avg=top1.avg, 
                             top5_val=top5.val, top5_avg=top5.avg)


def valid(args, epoch, val_loader, net, loss_fn, torch_device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (img, label) in enumerate(train_loader):
        img, label = img.to(torch_device), label.to(torch_device)

        # compute output
        output = net(img)
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.cpu(), label.cpu(), topk=(1, 5))
        losses.update(loss, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        logger.log_write("val", epoch=epoch,
                         loss_val=losses.val, loss_avg=losses,avg,
                         top1_val=top1.val, top1_avg=top1.avg, 
                         top5_val=top5.val, top5_avg=top5.avg)
    return top1.avg

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch_device = torch.device("cuda")


    train_loader, val_loader = get_dataloader(args, "./data")

    if args.model == "fish99":
        net = net_factory.fish99(args.num_cls)
    elif args.model == "fish150":
        net = net_factory.fish150(args.num_cls)

    net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_avg = -1
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, net, loss, optim, torch_device)    
        top1_avg = valid(args, epoch, val_loader, net, loss, torch_device)
        print("Epoch[%d] top1_avg : %f"%(epoch, top1_avg))

        if best_avg < top1_avg:
            torch.save(net.state_dict(), "checkpoint.pth.tar")
    
