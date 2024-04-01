# Start by testing if the different models are already trained (i.e. have .pt files) and if not, train them.

import torch
import os
from train import learn, get_model
import copy
from sharpness_estimation import get_sharpness
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



LEARNING_RATES = [1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2, 5*1e-2, 1e-1]
BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 50_000]

# LEARNING_RATES = [1e-4, 1e-3, 1e-2]
# BATCH_SIZES = [128, 256, 512, 1024]

EFFECT_VARIABLES = [LEARNING_RATES, BATCH_SIZES]

BASELINE_LEARNING_RATE = 1e-4
BASELINE_BATCH_SIZE = 128

BASELINE_EFFECT = [BASELINE_LEARNING_RATE, BASELINE_BATCH_SIZE]


def get_model_ready(class_name = 'MLP_1'):
    models = dict() # key: model name, value: model
    hyperparameters = [] # [lr, batch_size]
    
    for i, var in enumerate(EFFECT_VARIABLES):
        for k, value in enumerate(var):
            model_name = class_name
            for j, baseline_effect in enumerate(BASELINE_EFFECT):
                if i == j:
                    model_name += "_" + str(value) 
                    hyperparameters.append(value)
                else:
                    model_name += "_" + str(baseline_effect)
                    hyperparameters.append(baseline_effect)
                    
            model_name += ".pt"
                
                
            if os.path.exists(model_name):
                model = get_model(class_name)
                model.load_state_dict(torch.load(model_name))
                models[model_name] = copy.deepcopy(model)
            else:
                model = learn(model_name=class_name, batch_size = hyperparameters[1], learning_rate = hyperparameters[0]) 
                torch.save(model.state_dict(), model_name)
                models[model_name] = copy.deepcopy(model)
    
    return models


def show_results(effect, models, effect_name, model_class_name = 'MLP_1'):
    i = EFFECT_VARIABLES.index(effect)
    
    list_of_sharpness = []
    list_of_values = []

    for value in effect:
        model_name = model_class_name
        for j, baseline_effect in enumerate(BASELINE_EFFECT):
            if i == j:
                model_name += "_" + str(value) 
            else:
                model_name += "_" + str(baseline_effect)
                    
        model_name += ".pt"
        
        model = models[model_name]
        
        sharpness = get_sharpness(model)
        list_of_sharpness.append(sharpness.item())
        list_of_values.append(value)
    
    print(effect_name, list_of_values, list_of_sharpness)
    sns.lineplot(x=list_of_values, y=list_of_sharpness)
    plt.xlabel(effect_name)
    plt.ylabel("Sharpness")
    plt.show()
    
    
    
        
        

if __name__ == "__main__":
    models = get_model_ready('MLP_1')
    # show_results(LEARNING_RATES, models, "Learning Rate", "MLP_1")
    show_results(BATCH_SIZES, models, "Batch Size")
        
    
 
    