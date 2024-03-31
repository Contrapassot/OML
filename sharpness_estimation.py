import torch
import dataLoader
from torch.utils.data import DataLoader
from train import MLP_1, learn
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_TRIALS = 5
EPSILON = 0.5


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
        
        print(sample_losses)



def get_model_params(model):
    return list(model.parameters())

def get_perturbed_params(original_params):
    perturbed_params = []
    for param in original_params:
        random_vector = torch.randn(param.size()).to(device)
        # print(torch.linalg.vector_norm(random_vector))
        perturbed_params.append(param.to(device) + random_vector/torch.linalg.vector_norm(random_vector) * EPSILON)
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
        print(outputs[0:2], labels[0])
        loss += model.loss(outputs, labels).item()
        # loss += model.loss(F.softmax(outputs, dim=1), labels).item()
        
    return loss
    
    

if __name__ == '__main__':
    # model = MLP_1().to(device)
    # model = model.type(torch.float32)
    model = learn()
    generate_samples(model)