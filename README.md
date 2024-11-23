main.py


Main function to train a ResNet model on a custom dataset.

    This function performs the following steps:
    1. Loads a dataset with random data and targets.
    2. Initializes a DataLoader for batching and shuffling the dataset.
    3. Loads a ResNet model with a specified number of output classes.
    4. Sets up an Adam optimizer and CrossEntropyLoss criterion.
    5. Runs a training loop for a specified number of epochs, performing the following in each iteration:
        - Forward pass: Computes model outputs from inputs.
        - Loss computation: Calculates the loss using the criterion.
        - Backward pass: Computes gradients by backpropagation.
        - Optimization step: Updates model parameters using the optimizer.
        - Model saving: Saves the model's state dictionary at the end of each epoch.

    
resnet.py


For creating resnet 50

base_model.py

