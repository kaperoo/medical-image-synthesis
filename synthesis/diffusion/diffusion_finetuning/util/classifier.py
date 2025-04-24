import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

TRAIN_PATH = 'train_data'
TEST_PATH = 'test_data'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_CLASSES = 7
LR = 1e-3
EPOCHS = 100

class Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, model_name='resnet18', pretrained=True):
        super(Model, self).__init__()
        base_model = getattr(models, model_name)(pretrained=pretrained)
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # Keep all layers except FC
        
        # Additional convolutional layers
        self.extra_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.extra_conv(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_data(): 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    dataloaders = {'train': train_loader, 'val': test_loader}
    classes = train_dataset.classes
    return dataloaders, classes

# Load Pretrained ResNet Model
def get_resnet_model(num_classes, model_name='resnet18', pretrained=True):
    model = getattr(models, model_name)(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Define Training Loop
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss, correct, total = 0.0, 0, 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
            
            epoch_loss = running_loss / total
            epoch_acc = correct.double() / total
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

# Model Evaluation
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
# Example usage
if __name__ == "__main__":
    dataloaders, class_names = load_data()
    
    # Get model and define loss/optimizer
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # Train model
    trained_model = train_model(model, dataloaders, criterion, optimizer, DEVICE, EPOCHS)
    
    # Evaluate model
    evaluate_model(trained_model, dataloaders['val'], DEVICE, class_names)
