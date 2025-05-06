import torch
import torchvision
from torchvision import transforms 
import sklearn 
from sklearn import scikitlearn
import numpy
import pandas
import matplotlib
import tqdm
import os
from tqdm import tqdm

#test for get data
def get_data(batch_size, test_batch_size=16, num_workers=2, mean=None, std=None, num_train_samples=None):
    #for small size we can use 16 or 32 

    # Compute mean and standard deviation on training set if not given
    if mean is None or std is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        full_training_data = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)

        images = torch.stack([image for image, _ in full_training_data], dim=3)
        mean = torch.mean(images)
        std = torch.std(images)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    # Load data
    full_training_data = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int(num_samples * 0.7 + 1) if num_train_samples is None else num_train_samples
    validation_samples = num_samples - training_samples
    training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples, validation_samples])

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(validation_data, test_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
