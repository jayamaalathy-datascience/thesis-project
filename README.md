main.py


To train a ResNet model on a custom dataset.

    Performs the following tasks:
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


resnetBatchNorm.py

For creating resnet 50.
Perform batchNorm on the resnet50 model.

resnetGroupNorm.py, mainGroupNorm.py

Perform Group normalization.

resnetLayerNorm.py, mainLayerNorm.py

Perform Layer normalization.

base_model.py

BaseModel is an abstract base class for neural network models.

CIFAR100.py is a custom class for handling the CIFAR-100 dataset.

This class loads the CIFAR-100 dataset using torchvision.

