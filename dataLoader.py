import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch

# dataset site : https://www.cs.toronto.edu/~kriz/cifar.html


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def load_batches():
    batch_1 = unpickle('./cifar-10-batches-py/data_batch_1')
    batch_2 = unpickle('./cifar-10-batches-py/data_batch_2')
    batch_3 = unpickle('./cifar-10-batches-py/data_batch_3')
    batch_4 = unpickle('./cifar-10-batches-py/data_batch_4')
    batch_5 = unpickle('./cifar-10-batches-py/data_batch_5')

    test_batch = unpickle('./cifar-10-batches-py/test_batch')

    train_batches = [batch_1, batch_2, batch_3, batch_4, batch_5]
    return train_batches, test_batch


def load_images_labels(n_batches = 5):
    train_batches, test_batch = load_batches()

    train_images = normalize_image(np.concatenate([np.array(train_batches[i][b'data']) for i in range(n_batches)], axis=0)).astype(np.float32)
    train_labels = np.concatenate([np.array(train_batches[i][b'labels']) for i in range(n_batches)], axis=0).astype(np.float32)

    test_images = normalize_image(np.array(test_batch[b'data'])).astype(np.float32)
    test_labels = np.array(test_batch[b'labels']).astype(np.float32)
    
 
    
    # print(train_images[0])
    
    # make labels to one_hot encoding
    encoder = OneHotEncoder()
    one_hot_train_labels = encoder.fit_transform(train_labels.reshape(-1,1)).toarray()
    one_hot_test_labels = encoder.transform(test_labels.reshape(-1,1)).toarray()
    
    train_dataset = CIFAR10Dataset(train_images, one_hot_train_labels)
    test_dataset = CIFAR10Dataset(test_images, one_hot_test_labels)
    

    return train_dataset, test_dataset


def normalize_image(image):
    return image/255.0

if __name__ == '__main__':

    a,b = load_images_labels(1)

    print(a[0][0][0])
    
 
    
    
    # images = batch_1[b'data']
    # labels = batch_1[b'labels']

    # print(images.shape, len(labels))