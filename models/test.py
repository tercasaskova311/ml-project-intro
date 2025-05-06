import torch
import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sklearn
import numpy
import pandas
import matplotlib
import os
import tqdm
from tqdm import tqdm

#test for get data

root = os.path.join(os.path.dirname(__file__), '..', 'data')
root_dir_train = os.path.join(root, 'training')
root_dir_test  = os.path.join(root, 'test')


def get_data(batch_size, test_batch_size=16, num_workers=2, mean=None, std=None, num_train_samples=None):
    # for small size we can use 16 or 32 
    # num_workers is the number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. We can use 1 or 2 because we have a small size.

    # Compute mean and standard deviation on training set if not given
    target_size = (224, 224)
    if mean is None or std is None:

        transform = transforms.Compose([
            transforms.Resize((target_size)),
            transforms.ToTensor()
        ])
        
        train_data = ImageFolder(root_dir_train, transform=transform)
        #loading images organized by “one folder per class"
        #Builds a list of (image_path, class_index) pairs

        images = torch.stack([image for image, _ in train_data], dim=0)
        #dimension 0 because we want to stack the images in a batch. 
        mean = torch.mean(images)
        std = torch.std(images)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((target_size)),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    # Load data
    train_data = ImageFolder(root_dir_train, transform=transform)    
    test_data = ImageFolder(root_dir_test, transform=transform)

    # Create train and validation splits
    num_samples = len(train_data)
    training_samples = int(num_samples * 0.7 + 1) if num_train_samples is None else num_train_samples
    validation_samples = num_samples - training_samples
    training_data, validation_data = torch.utils.data.random_split(train_data, [training_samples, validation_samples])

    # Initialize dataloaders
    train_loader = DataLoader(training_data, batch_size, shuffle=True, num_workers=num_workers)
    val_loader =DataLoader(validation_data, test_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


#main guard — prevents recursive spawning on Windows
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()            # optional, silences a Windows warning

    train_loader, val_loader, test_loader = get_data(16)

    i = iter(train_loader)
    a = next(i)
    for t in a:
        print(t)
