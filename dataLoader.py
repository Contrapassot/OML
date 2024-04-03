import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# dataset site : https://www.cs.toronto.edu/~kriz/cifar.html

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


DIR = './data/cifar-10-batches-py/'


def load_batches():
    print(f'{DIR}data_batch_{1}')
    train_batches = [unpickle(f'{DIR}data_batch_{i}') for i in range(1, 6)]
    test_batch = unpickle(f'{DIR}test_batch')

    return train_batches, test_batch


def load_images_labels(n_batches=5):
    train_batches, test_batch = load_batches()

    train_images = normalize_image(
        np.concatenate([np.array(train_batches[i][b'data']) for i in range(n_batches)], axis=0)).astype(np.float32)
    train_labels = np.concatenate([np.array(train_batches[i][b'labels']) for i in range(n_batches)], axis=0).astype(
        np.float32)

    test_images = normalize_image(np.array(test_batch[b'data'])).astype(np.float32)
    test_labels = np.array(test_batch[b'labels']).astype(np.float32)

    # print(train_images[0]) test

    # make labels to one_hot encoding
    encoder = OneHotEncoder()
    one_hot_train_labels = encoder.fit_transform(train_labels.reshape(-1, 1)).toarray()
    one_hot_test_labels = encoder.transform(test_labels.reshape(-1, 1)).toarray()

    train_dataset = CIFAR10Dataset(train_images, one_hot_train_labels)
    test_dataset = CIFAR10Dataset(test_images, one_hot_test_labels)

    return train_dataset, test_dataset


def normalize_image(image):
    return image / 255.0


if __name__ == '__main__':
    print(" ")
