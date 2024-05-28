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
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import hashlib
import warnings
from resultsSaver import sharpnessResultsSaver

warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    mp.set_start_method('spawn', force=True)

LEARNING_RATES = [1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2, 5*1e-2, 1e-1]
BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 50_000]
OPTIMIZERS = ['SGD', 'AdaGrad']

# LEARNING_RATES = [1e-4, 1e-3, 1e-2]
# BATCH_SIZES = [128, 256, 512, 1024]

EFFECT_VARIABLES = [LEARNING_RATES, BATCH_SIZES]

BASELINE_LEARNING_RATE = 1e-4
BASELINE_BATCH_SIZE = 128

BASELINE_EFFECT = [BASELINE_LEARNING_RATE, BASELINE_BATCH_SIZE]


def iterate_over_models(array_iterations, model_class_name='MLP_1', base_seed=42, unit_test=False):
    """
    Prepares n_iterations instances of all models.
    Each model with fixed hyperparameters is prepared n_iterations times with a different seed.

    Args:
        n_iterations (int): Number of iterations to prepare models for.
        model_class_name (str): Class name of the model to prepare.

    Returns:
        dict: A dictionary containing model names as keys and model instances as values.
    """
    models = dict()  # key: model name, value: model
    for i in array_iterations:
        models = models | get_model_ready(model_class_name, i, base_seed=base_seed, unit_test=unit_test)

    return models



def get_model_ready(class_name='MLP_1', iteration_number=0, base_seed=42, unit_test=False):
    """
    Prepares all models of a given class name with a specific seed.

    Iterates over predefined optimizer types and sets of hyperparameters (learning rates and
    batch sizes) to prepare a model. If the model has already been trained, it is loaded from a .pt file.

    Args:
        class_name (str): Class name of the model to be prepared.
        iteration_number (int): The iteration number used to modify the seed, ensuring
                                different initializations for different iterations.
        base_seed (int): The base seed for random number generation.

    Returns:
        dict: A dictionary containing the model name as the key and the model instance as the value.
    """
    # torch.manual_seed(base_seed + iteration_number)
    np.random.seed(base_seed + iteration_number)
    models = dict()  # key: model name, value: model

    for opt in OPTIMIZERS:
        for i, var in enumerate(EFFECT_VARIABLES):
            for k, value in enumerate(var):
                hyperparameters = []  # [lr, batch_size]
                model_name = "models/" + class_name
                unit_test_model_name = "unit_test_models/" + class_name
                for j, baseline_effect in enumerate(BASELINE_EFFECT):
                    if i == j:
                        model_name += "_" + str(value)
                        unit_test_model_name += "_" + str(value)
                        hyperparameters.append(value)
                    else:
                        model_name += "_" + str(baseline_effect)
                        unit_test_model_name += "_" + str(baseline_effect)
                        hyperparameters.append(baseline_effect)

                model_name += f"_{opt}_{iteration_number}.pt"
                unit_test_model_name += f"_{opt}_{iteration_number}.pt"

                seed = int(hashlib.sha1(model_name.encode()).hexdigest(), 16) % (2 ** 32)
                torch.manual_seed(seed + base_seed)

                model = get_model(class_name)
                model = model.type(torch.float32)

                # save the model initialization for unit testing
                if not os.path.exists(unit_test_model_name):
                    torch.save(model.state_dict(), unit_test_model_name)

                # Unit test
                if unit_test:
                    if model_name not in models.keys():
                        print(model_name in models.keys(), model_name, models.keys())
                        assert torch.all(
                            model.state_dict()[list(model.state_dict().keys())[0]] == torch.load(unit_test_model_name)[
                                list(model.state_dict().keys())[0]])
                        print(model.state_dict()[list(model.state_dict().keys())[0]])
                        print("Unit test passed")

                if os.path.exists(model_name):
                    model.load_state_dict(torch.load(model_name))

                else:
                    model = learn(model, model_name=class_name, batch_size=hyperparameters[1],
                                  learning_rate=hyperparameters[0],
                                  optimizer=opt, iteration_number=iteration_number)
                    torch.save(model.state_dict(), model_name)

                models[model_name] = copy.deepcopy(model)

    return models


def run(effect, effect_name, model_class_name='MLP_1', optimizer='SGD', array_iterations=[0,1,2]):
    """
    Plots the sharpness of the models for a given hyperparameter effect.

    Args:
        effect (list): A list of hyperparameter values to examine.
        models (dict): A dictionary of models.
        effect_name (str): The name of the hyperparameter.
        model_class_name (str): The class name of the model.
        optimizer (str): The optimizer used in the model.
        n_iterations (int): The number of iterations for the model preparation.
    """
    i = EFFECT_VARIABLES.index(effect)
    n_iterations = len(array_iterations)

    

    list_of_sharpness = []
    list_of_values = []
    errors = []
    all_sharpness_measures = []
    
    saver = sharpnessResultsSaver(model_class_name=model_class_name, optimizer=optimizer, effect_name = effect_name)
    
    if saver.results is None:
        
        models = iterate_over_models(array_iterations=array_iterations, model_class_name=model_class_name, base_seed=42, unit_test=True) 
        print(models.keys())

        for value in effect:
            model_name = "models/" + model_class_name
            for j, baseline_effect in enumerate(BASELINE_EFFECT):
                if i == j:
                    model_name += "_" + str(value)
                else:
                    model_name += "_" + str(baseline_effect)

            model_name += f"_{optimizer}"

            average_sharpness, left_interval, right_interval, all_measures = get_sharpness_stats(model_name, models, n_iterations)
            list_of_sharpness.append(average_sharpness)
            list_of_values.append(value)
            errors.append((right_interval - left_interval) / 2)
            all_sharpness_measures.append(all_measures)
            print(all_sharpness_measures)
        
        saver = sharpnessResultsSaver(effect = effect, effect_name = effect_name, list_of_sharpness_mean = list_of_sharpness, 
                                      list_of_sharpness = all_sharpness_measures, list_of_values = list_of_values, 
                                      errors = errors, model_class_name=model_class_name, 
                                      optimizer=optimizer, n_iterations=n_iterations, save = True)
        
    
    assert saver.results is not None
    
  


def plot_results(effect_name, model_class_name, optimizer, show_plot):
    
    saver = sharpnessResultsSaver(model_class_name=model_class_name, optimizer=optimizer, effect_name = effect_name)
    
    list_of_sharpness_mean = saver.results['list_of_sharpness_mean']
    list_of_values = saver.results['list_of_values']
    errors = saver.results['errors']
   
    plt.figure(figsize=(10, 6))
    plt.errorbar(list_of_values, list_of_sharpness_mean, yerr=errors, fmt='o', ecolor='red', capsize=5)
    sns.lineplot(x=list_of_values, y=list_of_sharpness_mean)
    plt.xlabel(effect_name)
    plt.ylabel("Sharpness")
    
    #save the plot
    plt.savefig(f"results/{model_class_name}_{optimizer}_{effect_name}.png")
    
    if show_plot:
        plt.show()    
    
    


def get_sharpness_stats(model_name, models, n_iterations):
    """
    Computes the mean and confidence interval of sharpness for given models.

    Args:
        model_name (str): The base name of the model to calculate sharpness for.
        models (dict): A dictionary containing models.
        n_iterations (int): Number of iterations to calculate sharpness over.

    Returns:
        tuple: Mean sharpness, left interval, right interval, and the list of sharpness values.
    """
    list_of_sharpness = []
    for i in range(n_iterations):
        complete_model_name = model_name + f"_{i}.pt"
        model = models[complete_model_name]
        sharpness = get_sharpness(model)
        print(sharpness.tolist())
        list_of_sharpness.extend(sharpness.tolist())

    sharpness_values = np.array(list_of_sharpness)
    mean_sharpness = np.mean(sharpness_values)
    std_sharpness = np.std(sharpness_values, ddof=1)  # delta degree of freedom = 1 because we are working with sampled data
    n = len(sharpness_values)

    t_value = stats.t.ppf(0.975, n - 1)  # two-tailed 95% confidence interval
    # print(t_value)
    margin_error = t_value * (std_sharpness / np.sqrt(n))
    left_interval = mean_sharpness - margin_error
    right_interval = mean_sharpness + margin_error

    return mean_sharpness, left_interval, right_interval, sharpness_values.tolist()


if __name__ == '__main__':
    
    model_class_name = 'MLP_1'
    optimizer = 'AdaGrad'
    effect_name = "LearningRate"
    array_iterations = [0, 1, 2, 3, 4, 5]
    effect = LEARNING_RATES
    
    run(effect, effect_name, model_class_name, optimizer, array_iterations)
    plot_results(effect_name, model_class_name, optimizer, show_plot = False)

    #--------------------------------------------------------------------------------------------

    model_class_name = 'MLP_1'
    optimizer = 'AdaGrad'
    effect_name = "Batchsize"
    array_iterations = [0, 1, 2, 3, 4, 5]
    effect = BATCH_SIZES
        
    run(effect, effect_name, model_class_name, optimizer, array_iterations)
    plot_results(effect_name, model_class_name, optimizer, show_plot = False)

    
    #--------------------------------------------------------------------------------------------

    
    model_class_name = 'MLP_1'
    optimizer = 'SGD'
    effect_name = "LearningRate"
    array_iterations = [0, 1, 2, 3, 4, 5]
    effect = LEARNING_RATES
    
    run(effect, effect_name, model_class_name, optimizer, array_iterations)
    plot_results(effect_name, model_class_name, optimizer, show_plot = False)
    
    #--------------------------------------------------------------------------------------------

    
    model_class_name = 'MLP_1'
    optimizer = 'SGD'
    effect_name = "Batchsize"
    array_iterations = [0, 1, 2, 3, 4, 5]
    effect = BATCH_SIZES
    
    run(effect, effect_name, model_class_name, optimizer, array_iterations)
    plot_results(effect_name, model_class_name, optimizer, show_plot = False)