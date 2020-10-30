from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from datasets import ImageNetDownSampleDataset
from torchvision.transforms import transforms


__all__ = ['CIFAR10DataLoader', 'ImageNetDownSampleDataLoader']


class CIFAR10DataLoader(DataLoader):
    def __init__(self, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
        else:
            train = False

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.dataset = CIFAR10(root='./data', train=train, transform=transform, download=True)

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class ImageNetDownSampleDataLoader(DataLoader):
    def __init__(self, split='val', image_size=224, batch_size=16, num_workers=8):

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.dataset = ImageNetDownSampleDataset(root='./data', split=split, transform=transform)

        super(ImageNetDownSampleDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


