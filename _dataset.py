import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(96, padding_mode='symmetric'),
        transforms.RandomRotation(45),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ], p=0.8),
        transforms.RandomApply([
            transforms.RandomGrayscale(p=1.0),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), fillcolor=0)
        ], p=0.8),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
}

class ImgDataset(Dataset):

    def __init__(self, paths, labels, pic_size, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.pic_size = pic_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = cv2.imread(self.paths[index])
        X = cv2.resize(X, (self.pic_size, self.pic_size))
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)
