import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.resnetGroupNorm import CustomResNet
from data.cifar100 import CIFAR100Dataset  # Assuming you save the class in cifar100_dataset.py
import os
import torch



def calculate_accuracy(output, target, top_k=(1,)):
    """
    Calculates the Top-K accuracies.
    
    Args:
        output (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size).
        top_k (tuple): Tuple of top-k values for which to compute accuracy.
    
    Returns:
        List of accuracies for each top-k value.
    """
    max_k = max(top_k)
    batch_size = target.size(0)

    # Get the top-k predictions
    _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()  # Transpose to shape (max_k, batch_size)
    
    # Check if predictions match the targets
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # Calculate accuracies for each top-k
    top_k_accuracies = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        top_k_accuracies.append((correct_k / batch_size).item())

    return top_k_accuracies

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_top1_accuracy = 0.0
        val_top5_accuracy = 0.0
        total_samples = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Calculate Top-1 and Top-5 accuracies
            top1_acc, top5_acc = calculate_accuracy(outputs, labels, top_k=(1, 5))
            val_top1_accuracy += top1_acc * inputs.size(0)
            val_top5_accuracy += top5_acc * inputs.size(0)
            total_samples += inputs.size(0)


            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy,val_top1_accuracy,val_top5_accuracy

def main():
    """
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
    """
    
# Initialize variables for tracking the best model
    best_top1_accuracy = 0.0
    best_model_path = "best_model.pth"

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR100Dataset(train=True, transform=transform)
    test_dataset=CIFAR100Dataset(train=False, transform=transform)
    train_size = len(train_dataset)
    val_size = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    



    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(model_type="resnet50", num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Create directory for saving models
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy,val_top1_accuracy,val_top5_accuracy = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Top-1 Accuracy: {val_top1_accuracy:.4f}, Top-5 Accuracy: {val_top5_accuracy:.4f}")

        # Save the model if it has the highest Top-1 accuracy so far
        if val_top1_accuracy > best_top1_accuracy:
            best_top1_accuracy = val_top1_accuracy
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_top1_accuracy,
                        }, best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with Top-1 Accuracy: {best_top1_accuracy:.4f}")
        
        # Save the model's state dictionary
        

if __name__ == "__main__":
    main()
