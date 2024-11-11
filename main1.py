import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.resnet2 import CustomResNet
from data.cifar100 import CIFAR100Dataset  # Assuming you save the class in cifar100_dataset.py
import os

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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

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
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR100Dataset(train=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(model_type="resnet50", num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Create directory for saving models
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save the model's state dictionary
        torch.save(model.state_dict(), os.path.join(save_dir, f'resnet_epoch_{epoch}.pth'))

if __name__ == "__main__":
    main()