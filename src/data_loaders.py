import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import transforms


__all__ = ['CIFAR10DataLoader', 'ImageNetDataLoader']


class CIFAR10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
        else:
            train = False

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


if __name__ == '__main__':
    data_loader = ImageNetDataLoader(
        data_dir='/home/hchen/Projects/vat_contrast/data/ImageNet',
        split='val',
        image_size=384,
        batch_size=16,
        num_workers=0)

    for images, targets in data_loader:
        print(targets)
