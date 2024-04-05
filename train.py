import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOPPING_CRITERION = 1e-3
STOPPING_CRITERION_EPOCHS = 5


class MLP_1(nn.Module):
    """
    Multilayer perceptron to classify CIFAR-10 images.

    Attributes:
        seq (nn.Sequential): The sequential layer of the model: 3072 -> 512 (ReLU) -> 256 (ReLU) -> 128 (ReLU) -> 10.
        loss (nn.CrossEntropyLoss): The Cross Entropy loss function.
        optimizer (torch.optim): The optimizer: SGD, AdaGrad or FullGD.
    """
    def __init__(self):
        super(MLP_1, self).__init__()
        self.seq = nn.Sequential(nn.Linear(3072, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 10), nn.BatchNorm1d(10, affine=False,
                                                                    momentum=0))  

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.seq(x)
        return x


def get_model(name):
    """
    Returns a model instance based on the name provided.
    """
    if name == "MLP_1":
        return MLP_1().to(device)


def learn(model, model_name, batch_size, learning_rate, optimizer, tensorboard_path="./tensorboard",
          iteration_number=0):
    """
    Trains a model with the given hyperparameters, logs the results to TensorBoard, and returns the trained model.

    The model is tested after each epoch. Training stops when the stopping criterion is reached, i.e. the loss difference
    between two consecutive epochs is less STOPPING_CRITERION than for STOPPING_CRITERION_EPOCHS consecutive epochs.

    Args:
        model (nn.Module): The model to train.
        model_name (str): The name of the model to train.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for training.
        optimizer (str): The optimizer to use for training.
        tensorboard_path (str): The path to save tensorboard logs.
        iteration_number (int): The iteration number to differentiate models.

    Returns:
        model (nn.Module): The trained model.
    """
    if optimizer == 'AdaGrad':
        model.optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        model.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer not supported")

    train_dataset, test_dataset = dataLoader.load_images_labels(5)

    print(model)
    print(train_dataset[0][0][0])

    num_epochs = 20
    loss_values = []
    test_accuracy = []
    train_accuracy = []
    test_loss = 0

    tb_path = tensorboard_path + "/" + str(model_name) + "_lr_" + str(learning_rate) + "_batch_" + str(
        batch_size) + "_opt_" + str(optimizer) + "_iter_" + str(iteration_number)
    print(tb_path)
    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)

    tb_writer = SummaryWriter(tb_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    stopping_criterion_counter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        if epoch == num_epochs - 1:
            print("Final epoch reached, did not converge.")  # TODO throw error
            # break
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, labels)
            loss.backward()
            model.optimizer.step()

            epoch_loss += loss.item()
            num_batches = i + 1

        loss_values.append(epoch_loss / num_batches)

        if epoch > 1 and epoch % 10 == 0:
            tb_writer.add_scalar("Loss", loss_values[-1], epoch)
            tb_writer.add_scalar("Test accuracy", test_accuracy[-1], epoch)
            tb_writer.add_scalar("Test loss", test_loss, epoch)

        if epoch > 1:
            if abs(loss_values[-1] - loss_values[-2]) < STOPPING_CRITERION:
                stopping_criterion_counter += 1
        else:
            stopping_criterion_counter = 0

        if stopping_criterion_counter == STOPPING_CRITERION_EPOCHS:
            print(
                f"Stopping criterion reached at epoch {epoch} with loss difference {abs(loss_values[-1] - loss_values[-2])}")
            break

        print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy.append(100 * correct / total)
        test_loss = loss.item()
        print(f"Test accuracy: {100 * correct / total}")

    tb_writer.flush()
    tb_writer.close()
    return model


if __name__ == '__main__':
    learn('MLP_1', 128, 1e-4, 'SGD', tensorboard_path="./tensorboard", iteration_number=0)
