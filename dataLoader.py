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
     # values of means and std : https://github.com/kuangliu/pytorch-cifar/issues/19
     transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
DIR = './data/cifar-10-batches-py/'


class CIFAR10Dataset(Dataset):
    """
    Custom dataset class for CIFAR-10 to handle images and labels.
    """

    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def unpickle(file):
    """
    Loads the CIFAR-10 batch file and returns the dictionary.
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batches():
    """
    Loads all train and test batches of CIFAR-10 dataset.

    Returns:
        tuple: Contains the list of training batches and the test batch.
    """
    print(f'{DIR}data_batch_{1}')
    train_batches = [unpickle(f'{DIR}data_batch_{i}') for i in range(1, 6)]
    test_batch = unpickle(f'{DIR}test_batch')

    return train_batches, test_batch


def load_images_labels(n_batches=5):
    """
    Loads and prepares the CIFAR-10 images and labels for training and testing.

    Args:
        n_batches (int, optional): The number of training batches to load.

    Returns:
        tuple: A tuple containing the training dataset and the testing dataset,
               with images normalized and labels one-hot encoded.
   """
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
    """
    Normalizes the pixel values of an image to the range [0, 1].
    """
    return image / 255.0


if __name__ == '__main__':
    print(" ")
