import os
import numpy as np
from torch.utils.data import Dataset

__all__ = ['ImageNetDownSampleDataset']


class ImageNetDownSampleDataset(Dataset):
    def __init__(self, root='./data', split='val', transform=None):
        data = np.load(os.path.join(root, 'imagenet64', '{}_data.npz'.format(split)), allow_pickle=True)
        self.data = data['data']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].reshape(64, 64, 3)
        label = np.asarray(self.labels[index], dtype=np.int64)

        if self.transform is not None:
            image = self.transform(image)

        return image, label



if __name__ == '__main__':
    dataset = ImageNetDownSampleDataset()

    for i in range(len(dataset)):
        image, label = dataset.__getitem__(i)
        print(image.shape)



