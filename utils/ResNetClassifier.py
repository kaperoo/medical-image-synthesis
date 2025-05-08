# Set up CUDA in OS
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Import libabries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import time
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Check version of Pytorch
print(torch. __version__)


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Find out if a GPU is available
use_cuda = torch.cuda.is_available()

# Set up paths for data after downloading
#train_dir and test_dir should have following structure 
#train_dir/<class_name>/images
#so for instance
#train_dir/ERM/erm_1043186_1.jpg
train_dir = ""
test_dir = ""


# Display image for reference
def show_image(imgPath):
    white_torch = torchvision.io.read_image(imgPath)
    print("This is Epiretinal Membrane")
    image = T.ToPILImage()(white_torch)
    image.show()

def preproccesing():
    # Create transform function
    #Resnet model only accepts height and width of 224
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader, train_dataset, test_dataset

def train_model(train_dataloader, test_dataloader, train_dataset, test_dataset):
    model = models.resnet18(pretrained=True)

    num_features = model.fc.in_features

    #set depending on number of classes of OCT used
    num_classes = len(train_dataset.classes)

    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_loss=[]
    train_accuary=[]
    test_loss=[]
    test_accuary=[]

    num_epochs = 30   #(set no of epochs)
    start_time = time.time() #(for showing time)
    # Start loop
    for epoch in range(num_epochs): #(loop for every epoch)
        print("Epoch {} running".format(epoch)) #(printing message)
        """ Training Phase """
        model.train()    #(training model)
        running_loss = 0.   #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.
        # Append result
        train_loss.append(epoch_loss)
        train_accuary.append(epoch_acc)
        # Print progress
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))
        """ Testing Phase """
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset) * 100.
            # Append result
            test_loss.append(epoch_loss)
            test_accuary.append(epoch_acc)
            # Print progress
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))

    #set path to which to save classifier to
    save_path = 'OCT-classifier_resnet_18_final.pth'
    torch.save(model.state_dict(), save_path)

    plt.figure(figsize=(6,6))
    plt.plot(np.arange(1,num_epochs+1), train_accuary,'-o')
    plt.plot(np.arange(1,num_epochs+1), test_accuary,'-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.title('Train vs Test Accuracy over time')
    plt.show()