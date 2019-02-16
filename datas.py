import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# https://github.com/kevin-ssy/FishNet/blob/master/utils/data_aug.py#L4
class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def get_dataloader(args, data_path):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        data_path + "/train",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        data_path + "/val",
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader