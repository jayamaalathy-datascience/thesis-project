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

Topic Outline

 

Various normalization methods have been suggested in machine learning for application with neural networks. Normalization methods are usually non-linear operations ( during training, though they act as linear operators during inference ). Normalization methods are many but can be broadly categorized into two groups :-  a) Hand-crafted normalization such as batch or group normalization ( remember that this list is not exhaustive ) and b) Searched normalization methods which are found using neural architecture search.  While there are many studies on the relative performance of various normalization strategies, it is clear that the choice is crucial with respect to metrics such as training time and validation/test accuracy in image classification.  The objective of your thesis is 

a) Make a concerted analysis of the choice of normalization function on various image classification datasets ( e.g:- single-label datasets such as CIFAR100 and multi-label datasets such as COCO and BigEarthNet ). 

b) The normalization functions should include both groups ( as described above ) and analysis should be qualitative (i.e visual analysis of the impact of usage on training time, accuracy and explainability ) and also quantiative ( involving benchmark figures ). 

 

c) Explore the possibility of replacing activation functions with non-linear normalization methods ( which usually fall into the category b ) for image classification. 

 

 

The above is the core of the thesis topic and the contribution would involve : 

 

a) An elaborate codebase of your experiments involving implementation of various normalization functions. 

 

b) A clear analysis of various normalization layers.  To the best of my search and knowledge, there is no such concerted study of normalization functions involving both the groups together. 

 

 

 
