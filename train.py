# Imports

import numpy as np

import pandas as pd

import torch

import torchvision

from torchvision import datasets, transforms, models

from torch import nn

from torch import optim

import torch.nn.functional as F

import matplotlib.pyplot as plt

from PIL import Image

import json

from collections import OrderedDict

import time

import torchvision.models as models

from PIL import Image

import json

from matplotlib.ticker import FormatStrFormatter


#Load the data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
image_datasets = [datasets.ImageFolder(train_dir, transform=train_transform),
                  datasets.ImageFolder(valid_dir, transform=valid_transform),
                  datasets.ImageFolder(test_dir, transform=test_transform)]

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

### TODO: Build and train your network

### Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)

model = models.vgg16(pretrained= True)

#print(model)


### Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(25088,5120)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.2)),
                           ('fc2',nn.Linear(5120,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.2)),
                           ('fc3',nn.Linear(512,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))


model.classifier = classifier
model = model.to('cuda')
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


### Train the classifier layers using backpropagation using the pre-trained network to get the features
### Track the loss and accuracy on the validation set to determine the best hyperparameters

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to('cuda')

epochs = 10
steps = 0

print_every = 20

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in dataloaders[0]:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders[1]:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    validation_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(dataloaders[1]):.3f}.. "
                  f"Accuracy: {accuracy/len(dataloaders[1]):.3f}")
            running_loss = 0
            model.train()

# TODO: Do validation on the test set

model.to('cuda')
test_loss = 0
total = 0
accuracy = 0
with torch.no_grad():
    for inputs, labels in dataloaders[2]:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(dataloaders[2]):.3f}")


#The idea behind saving and loading the trained model is so that you can reuse it without having to train it again. 
#Therefore, we save the the components of the trained model, like classifier, epochs, state dictionary, optimizer, learning rate.

# TODO: Save the checkpoint 


model.class_to_idx = image_datasets[0].class_to_idx

checkpoint_path = 'vgg16_checkpoint.pth'

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'classifier': classifier,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
