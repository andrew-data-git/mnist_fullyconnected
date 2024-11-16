'''Basic script demonstrating MNIST classifier implementation'''

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create an FC Net
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

def check_accuracy(loader, model):
    '''Infer using model and calculate accuracy'''
    if loader.dataset.train:
        print('Check acc. on train')
    else:
        print('Check acc on test')
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for X,y in loader:
            X = X.to(device=device)
            y = y.to(device=device)
            X = X.reshape(X.shape[0],-1) # coerce to 64*10 size

            # Compute scores
            scores = model(X)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        # Print with tensors converted to floats
        print(f'Got {num_correct}/{num_samples} with acc. {float(num_correct)/float(num_samples)*100:2f}%')
    model.train()

# Set device and params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784
num_classes = 10

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Dataloaders
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=False)
train_dataloader =  DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=False)
test_dataloader =  DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Other
model = NN(input_size=input_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss() # objective function to minimise
optimiser = optim.Adam(model.parameters(), lr=learning_rate) # to modify params while training

# Train
model.train()
print(f'Training initiated, with learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}')
print(f'Running on {device}.')
for epoch in range(num_epochs):
    print(f'Epoch {epoch} of {num_epochs}')
    for batch_idx, (data, targets) in tqdm(enumerate(train_dataloader), unit='batch', total=len(train_dataloader)):
        data = data.to(device=device)
        targets = targets.to(device=device)
        data = data.reshape(data.shape[0],-1)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimiser.zero_grad()
        loss.backward() # compute x.grad = dloss/dx for each param x with required_grad == True

        # Gradient descent / Adam step
        optimiser.step() # update weights in loss.backward()
    
    check_accuracy(train_dataloader, model)

# Check against test set
check_accuracy(test_dataloader, model)
