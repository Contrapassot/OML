import torch
import dataLoader
from torch.utils.data import DataLoader
from train import MLP_1, learn
import torch.nn.functional as F
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TRIALS = 50
EPSILON = 5


def generate_samples(model):
    """
    Computes NUM_TRIALS losses for a perturbed version of the given model.

    Args:
        model (nn.Module): The model to generate samples for.

    Returns:
        list: A list of losses for each sample.
              Format: [original_loss, perturbed_loss_1, ..., perturbed_loss_NUM_TRIALS]
    """
    original_params = get_model_params(model)
    sample_losses = []
    train_dataset, _ = dataLoader.load_images_labels(5)

    data = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    for i in range(NUM_TRIALS):
        if i == 0:
            sample_losses.append(get_loss(model, data))
            continue
        perturbed_params = get_perturbed_params(original_params)
        new_model = set_model_params(model, perturbed_params)
        sample_losses.append(get_loss(new_model, data))

        # set_model_params(model, original_params)

        # print(sample_losses)

    print(sample_losses)
    return sample_losses


def get_model_params(model):
    """
    Returns the parameters of the model.
    """
    return list(model.parameters())


def get_perturbed_params(original_params):
    """
    Returns the perturbed parameters of the model.
    Parameters are perturbed by adding a random rescaled unit vector to the original parameters.

    Args:
        original_params (list): The original parameters of the model.

    Returns:
        perturbed_params (list): The perturbed parameters of the model.
    """
    all_params = torch.cat([param.flatten() for param in original_params])

    random_vector = torch.from_numpy(np.random.randn(len(all_params))).to(device)  # use numpy to save random seed for NN
    normalized_random_vector = random_vector / torch.linalg.vector_norm(random_vector)

    perturbed_params_vector = all_params + normalized_random_vector * EPSILON
    # perturbed_params_vector = all_params + random_vector * EPSILON

    perturbed_params = []
    start_index = 0
    for param in original_params:
        end_index = start_index + param.numel()
        perturbed_params.append(perturbed_params_vector[start_index:end_index].view_as(param))
        start_index = end_index

    return perturbed_params


def set_model_params(model, params):
    """
    Sets the parameters of the model.

    Args:
        model (nn.Module): The model to set parameters for.
        params (list): The parameters to set.

    Returns:
        new_model (nn.Module): The model with the new parameters.
    """
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for model_param, param in zip(new_model.parameters(), params):
            model_param.copy_(param)
    return new_model


def get_loss(model, data):
    """
    Returns the loss of the model on the given data.

    Args:
        model (nn.Module): The model to evaluate.
        data (DataLoader): The data to evaluate the model on.

    Returns:
        loss (float): The loss of the model on the data.
    """
    loss = 0
    for _, data in enumerate(data):  # batch size is the size of the dataset so just one iteration
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # print(outputs[0:2], labels[0])
        loss += model.loss(outputs, labels).item()
        # loss += model.loss(F.softmax(outputs, dim=1), labels).item()

    return loss


def get_avg_sharpness(model):
    """
    Returns the average sharpness of the model over NUM_TRIALS measurements.
    """
    sample_losses = generate_samples(model)
    return torch.tensor(sample_losses[1:]).mean() - sample_losses[0]


def get_sharpness(model):
    """
    Returns a list of NUM_TRIALS measurements of sharpness for the model.
    """
    sample_losses = generate_samples(model)
    return torch.tensor(sample_losses[1:]) - sample_losses[0]


if __name__ == '__main__':
    # model = MLP_1().to(device)
    # model = model.type(torch.float32)
    model = learn('MLP_1', 128, 1e-3, 'SGD')
    sample_losses = generate_samples(model)
    sharpness = get_sharpness(model)

    print(f"Sharpness: {sharpness}")
