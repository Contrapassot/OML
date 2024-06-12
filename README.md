# How Learning Rate, Batch Size, and Optimizer Influence Model Sharpness

## Table of Contents
- [How to use](#how-to-use)
- [Data Loader](#dataloader)
- [Train Model](#train)
- [Sharpness Estimation](#sharpness_estimation)
- [Results Saver](#resultssaver)



### How to use

(read what follows before running the experiment)

In `run.py`, choose your experiment parameters and called the functions `run()` and `plot_results()`

Example usage (already set up in `run.py`):

```
model_class_name = 'MLP_1'
optimizer = 'AdaGrad'
effect_name = "LearningRate"
array_iterations = [0, 1, 2, 3, 4, 5]
effect = LEARNING_RATES 

run(effect, effect_name, model_class_name, optimizer, array_iterations)
plot_results(effect_name, model_class_name, optimizer, show_plot = False)
```

-------------------------------------------------------------------------------------------


The results are saved in json files in the folder `./results`. Therefore, the `run()` function will not do anything (besides confirming that the results are indeed in said file). If you want to reproduce the results, simply remove those JSON files from the folder. 

To produce the results, the models need to be trained already. For that, the function `get_model_ready()` will train the model if it doesn't exist in the `./models` folder. 

Training the models takes **more than 24 hours** since we train until convergence and have a lot of models. You can consider training fewer models with hand selected hyperparameters by changing the LEARNING_RATES and BATCH_SIZES variables at the beginning of the run.py script. You can also change the array_iterations variable called by the `run()` function to have fewer iterations (the numbers in the array are here for reproducibility, but you can use any number you want, what matters for the number of networks trained is the length of the array). 

Estimating the local minimum sharpness also takes some time (15-30 minutes) which can be reduced by having fewer measures (change the variable NUM_TRIALS at the beginning of the script `sharpness_estimation.py`).

Finally, the plots produced will have a different layout than the ones in the report. Uncomment the `plot_results_2()` function from the `if __name__ == '__main__':` part of `run.py` to see them.



### dataLoader

This script is responsible for loading and preprocessing the CIFAR-10 dataset for both training and testing.

Key functions include:

`load_images_labels(n_batches=5)`: This function loads the specified number of training batches (default is 5) and a single test batch from the CIFAR-10 dataset. It normalizes the image data and converts the labels to np.float32 data type. The function returns a tuple containing the training and testing datasets.

Example usage:
```train_data, test_data = load_images_labels(n_batches=3)```

### train

This script contains the implementation of a Multilayer Perceptron (MLP) for classifying CIFAR-10 images and a learning function for training the model.

Key components include:

```MLP_1```: This class defines the MLP model. It consists of a sequential layer with the structure 3072 -> 512 (ReLU) -> 256 (ReLU) -> 128 (ReLU) -> 10.

```learn``` function: This function is responsible for training the model. It uses the optimizer defined in the MLP_1 class to update the model parameters. The function also implements a stopping criterion that ends the learning when the model has converged.

```STOPPING_CRITERION``` and ```STOPPING_CRITERION_EPOCHS```: These constants define the stopping criterion for the learning function. If the improvement in loss is less than STOPPING_CRITERION for STOPPING_CRITERION_EPOCHS consecutive epochs, the training stops.

### sharpness_estimation

This scripts contains the code necessary to sample the losses of the model close to the local minimum. This is used to estimate the sharpness. 

### resultsSaver

This scripts is useful to visualize and change the layout of the plots quickly. It saves the measurements from the experiment in JSON files, which allows us to do them only once and save time during development. 

