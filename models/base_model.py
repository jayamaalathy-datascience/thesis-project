import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    BaseModel is an abstract base class for neural network models.

    Attributes:
        backbone (nn.Module): The feature extraction part of the model.
        classifier (nn.Linear): The classification layer of the model.

    Methods:
        __init__(num_classes):
            Initializes the BaseModel with the specified number of output classes.
        
        forward(x):
            Defines the forward pass of the model.
        
        _build_backbone():
            Abstract method to build the backbone of the model. Subclasses should implement this method.
    """
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = self._build_backbone()
        self.classifier = nn.Linear(self.backbone.out_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def _build_backbone(self):
        raise NotImplementedError("Subclasses should implement this!")
