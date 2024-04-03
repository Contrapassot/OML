# Start by testing if the different models are already trained (i.e. have .pt files) and if not, train them.

import torch
import os
from train import learn, get_model
import copy
from sharpness_estimation import get_sharpness
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#LEARNING_RATES = [1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2, 5*1e-2, 1e-1]
#BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 50_000]
OPTIMIZERS = ['SGD', 'AdaGrad']

LEARNING_RATES = [1e-4, 1e-3, 1e-2]
BATCH_SIZES = [128, 256, 512, 1024]

EFFECT_VARIABLES = [LEARNING_RATES, BATCH_SIZES]

BASELINE_LEARNING_RATE = 1e-4
BASELINE_BATCH_SIZE = 128


BASELINE_EFFECT = [BASELINE_LEARNING_RATE, BASELINE_BATCH_SIZE]


def iterate_over_models(n_iteration, model_class_name = 'MLP_1'):
    models = dict() # key: model name, value: model
    for i in range(n_iteration):
        models = models|get_model_ready(model_class_name, i)
    
    return models

def get_model_ready(class_name = 'MLP_1', iteration_number = 0):
    models = dict() # key: model name, value: model
    
    for opt in OPTIMIZERS:
        for i, var in enumerate(EFFECT_VARIABLES):
            for k, value in enumerate(var):
                hyperparameters = [] # [lr, batch_size]
                model_name = "models/" + class_name
                for j, baseline_effect in enumerate(BASELINE_EFFECT):
                    if i == j:
                        model_name += "_" + str(value) 
                        hyperparameters.append(value)
                    else:
                        model_name += "_" + str(baseline_effect)
                        hyperparameters.append(baseline_effect)
                        
                model_name += f"_{opt}_{iteration_number}.pt"
                    
                    
                if os.path.exists(model_name):
                    model = get_model(class_name)
                    model.load_state_dict(torch.load(model_name))
                    models[model_name] = copy.deepcopy(model)
                else:
                    model = learn(model_name=class_name, batch_size = hyperparameters[1], learning_rate = hyperparameters[0],
                                optimizer = opt) 
                    torch.save(model.state_dict(), model_name)
                    models[model_name] = copy.deepcopy(model)
    
    return models


def show_results(effect, models, effect_name, model_class_name = 'MLP_1', optimizer = 'SGD', n_iterations = 5):
    i = EFFECT_VARIABLES.index(effect)
    
    list_of_sharpness = []
    list_of_values = []
    errors = []

    for value in effect:
        model_name = "models/" + model_class_name 
        for j, baseline_effect in enumerate(BASELINE_EFFECT):
            if i == j:
                model_name += "_" + str(value) 
            else:
                model_name += "_" + str(baseline_effect)
                    
        model_name += f"_{optimizer}"

        
        average_sharpness, left_interval, right_interval = get_sharpness_stats(model_name, models, n_iterations)
        list_of_sharpness.append(average_sharpness)
        list_of_values.append(value)
        errors.append((right_interval - left_interval)/2)

    # print(effect_name, list_of_values, list_of_sharpness)
    plt.figure(figsize=(10, 6))
    plt.errorbar(list_of_values, list_of_sharpness, yerr=errors, fmt='o', ecolor='red', capsize=5)
    sns.lineplot(x=list_of_values, y=list_of_sharpness)
    plt.xlabel(effect_name)
    plt.ylabel("Sharpness")
    plt.show()
    

def get_sharpness_stats(model_name, models, n_iterations):
    '''
    Computes the mean and confidence interval of sharpness
    Returns: mean, left interval, right interval
    '''
    list_of_sharpness = []
    for i in range(n_iterations):
        complete_model_name = model_name + f"_{i}.pt"
        model = models[complete_model_name]
        sharpness = get_sharpness(model)
        print(sharpness.tolist())
        list_of_sharpness.extend(sharpness.tolist())

    sharpness_values = np.array(list_of_sharpness)
    mean_sharpness = np.mean(sharpness_values)
    std_sharpness = np.std(sharpness_values, ddof=1) # delta degree of freedom = 1 because we are working with sampled data
    n = len(sharpness_values)

    t_value = stats.t.ppf(0.975, n - 1)  # two-tailed 95% confidence interval
    print(t_value)
    margin_error = t_value * (std_sharpness / np.sqrt(n))
    left_interval = mean_sharpness - margin_error
    right_interval = mean_sharpness + margin_error
 
    return mean_sharpness, left_interval, right_interval

    
    
    
    
        
        

if __name__ == "__main__":
    n_iterations = 3
    models = iterate_over_models(3, 'MLP_1')
    show_results(LEARNING_RATES, models, "Learning Rate", "MLP_1", 'AdaGrad', 3)
    #show_results(BATCH_SIZES, models, "Batch Size", "MLP_1", 'SGD', 3)
        
    
 