import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(nn.Module):
    """
    CustomResNet is a customizable ResNet model that allows for the replacement of BatchNorm layers with a specified normalization layer.
    Args:
        model_type (str): The type of ResNet model to use. Options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        num_classes (int): The number of output classes for the final fully connected layer.
        normalization (nn.Module, optional): The normalization layer to replace BatchNorm2d with. Defaults to nn.BatchNorm2d.
    Attributes:
        backbone (nn.Module): The ResNet backbone model with the specified normalization layers and modified final fully connected layer.
    Methods:
        _get_resnet_model(model_type):
            Retrieves the specified ResNet model with pretrained weights.
        _replace_batch_norm(module, normalization):
            Recursively replaces all BatchNorm2d layers in the given module with the specified normalization layer.
        forward(x):
            Defines the forward pass of the model.
    """
    def __init__(self, model_type, num_classes, normalization=nn.BatchNorm2d):
        super(CustomResNet, self).__init__()
        self.backbone = self._get_resnet_model(model_type)
        
        # Replace all BatchNorm layers
        self._replace_batch_norm(self.backbone, normalization)

        # Modify the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _get_resnet_model(self, model_type):
        resnets = {
            'resnet18': models.resnet18(pretrained=True),
            'resnet34': models.resnet34(pretrained=True),
            'resnet50': models.resnet50(pretrained=True),
            'resnet101': models.resnet101(pretrained=True),
            'resnet152': models.resnet152(pretrained=True)
        }
        return resnets[model_type]

    def _replace_batch_norm(self, module, normalization):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, normalization(child.num_features))
            else:
                self._replace_batch_norm(child, normalization)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Example usage: creating a ResNet50 with Instance Normalization
    model = CustomResNet(model_type='resnet50', num_classes=10, normalization=nn.InstanceNorm2d)
    print(model)
