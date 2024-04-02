import torch
import dataLoader
from torch.utils.data import DataLoader
from train import MLP_1, learn
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_TRIALS = 5
EPSILON = 5


def generate_samples(model):
    original_params = get_model_params(model)
    sample_losses = []
    train_dataset, _ =  dataLoader.load_images_labels(5)
   
    data =  DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    
    for i in range(NUM_TRIALS):
        if i == 0:
            sample_losses.append(get_loss(model, data))
            continue
        perturbed_params = get_perturbed_params(original_params)
        new_model = set_model_params(model, perturbed_params)
        sample_losses.append(get_loss(new_model, data))
        
        # set_model_params(model, original_params)
        
        #print(sample_losses)
    
    print(sample_losses)
    return sample_losses



def get_model_params(model):
    return list(model.parameters())

def get_perturbed_params(original_params):
    all_params = torch.cat([param.flatten() for param in original_params])
    random_vector = torch.randn(all_params.size()).to(device)
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
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for model_param, param in zip(new_model.parameters(), params):
            model_param.copy_(param)
    return new_model
        
def get_loss(model, data):
    loss = 0
    for _, data in enumerate(data):   # batch size is the size of the dataset so just one iteration
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        #print(outputs[0:2], labels[0])
        loss += model.loss(outputs, labels).item()
        # loss += model.loss(F.softmax(outputs, dim=1), labels).item()
        
    return loss
    
def get_sharpness(model):
    sample_losses = generate_samples(model)
    return torch.tensor(sample_losses[1:]).mean() - sample_losses[0]



if __name__ == '__main__':
    # model = MLP_1().to(device)
    # model = model.type(torch.float32)
    model = learn(MLP_1, 128, 1e-3)
    sample_losses = generate_samples(model)
    sharpness = get_sharpness(model)
    
    print(f"Sharpness: {sharpness}")