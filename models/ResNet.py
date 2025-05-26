import os
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml-project-intro/
data_dir = os.path.join(BASE_DIR, 'data')
train_dir = os.path.join(data_dir, 'training')
test_query_dir = os.path.join(data_dir, 'test', 'query')
test_gallery_dir = os.path.join(data_dir, 'test', 'gallery')


fine_tune = True  # Set to False to skip training and only extract features
k=10
batch_size = 64
num_epochs = 6
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_version = "resnet50"

# print(os.path.exists(train_dir))
# print(os.listdir(train_dir))  # will list subfolders/classes for training set
# print(os.listdir(test_query_dir))


def initialize_model(resnet_version=resnet_version, pretrained=True, feature_extract=True):
    from torchvision.models import ResNet50_Weights

    # Get the model constructor from torchvision.models
    model_fn = getattr(models, resnet_version)
    
    # Load with proper weights
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = model_fn(weights=weights)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Identity()

    return model.to(device)



class ImageDatasetWithoutLabels(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name  # returning image and filename (no label)


data_transforms = {
    'training': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# For training data (has subfolders/classes)
train_dataset = datasets.ImageFolder(train_dir, data_transforms['training'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# For query data (NO subfolders/classes)
query_dataset = ImageDatasetWithoutLabels(test_query_dir, transform=data_transforms['test'])
query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

# For gallery data (depends on structure) --> no subfolders/classes
gallery_dataset = ImageDatasetWithoutLabels(test_gallery_dir, data_transforms['test'])
gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)


def train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        final_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {final_loss:.4f}")

    return final_loss


def fine_tune_model(model, num_classes, learning_rate):
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #for param in model.fc.parameters():
        #param.requires_grad = False   # unfreeze only new head
    # Safely get in_features BEFORE replacing the classifier

    num_features = model.fc.in_features if hasattr(model.fc, "in_features") else 2048

    # Replace classification head
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    ).to(device)

    return model


def extract_features(model, loader):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
    return np.vstack(features)

def calculate_similarity(query_features, gallery_features):
    similarities = cosine_similarity(query_features, gallery_features)
    return similarities

def get_all_images(dataloader):
    all_images = []
    for images, _ in dataloader:
        all_images.append(images.cpu())
    return torch.cat(all_images, dim=0)

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


# Precompute true indices once
true_indices = []
for q_name in query_dataset.image_files:
    try:
        idx = gallery_dataset.image_files.index(q_name)
        true_indices.append(idx)
    except ValueError:
        print(f"[WARNING] Query image {q_name} not found in gallery. Assigning -1")
        true_indices.append(-1)  # or handle differently


def calculate_accuracy(similarities, true_indices, k):
    correct = 0
    for i, query_sim in enumerate(similarities):
        top_k_indices = np.argsort(query_sim)[-k:][::-1]  # Top k indices by similarity
        if true_indices[i] in top_k_indices:
            correct += 1
    return correct / len(similarities)




def save_metrics_json(
    model_name,
    top_k_accuracy,
    batch_size,
    is_finetuned,
    num_classes=None,
    runtime=None,
    loss_function="CrossEntropyLoss",
    num_epochs=None,
    final_loss=None
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml-project-intro/
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")

    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": 10,
        "top_k_accuracy": round(top_k_accuracy, 4),
        "batch_size": batch_size,
        "is_finetuned": is_finetuned,
        "num_classes": num_classes,
        "runtime_seconds": round(runtime, 2) if runtime else None,
        "loss_function": loss_function,
        "num_epochs": num_epochs,
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DEBUG] Metrics saved to: {os.path.abspath(out_path)}")

start_time = time.time()

# 1. Initialize model
resnet = initialize_model(resnet_version, pretrained=True, feature_extract=not fine_tune)
model = resnet
# 2. Optional fine-tuning
if fine_tune:
    num_classes = len(train_dataset.classes)
    model = fine_tune_model(model, num_classes=num_classes, learning_rate=learning_rate)
    final_loss = train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)
else:
    model.fc = nn.Identity()
    final_loss = None  # No training, no loss


# 3. Extract features
query_features = extract_features(model, query_loader)
gallery_features = extract_features(model, gallery_loader)

# 4. Compute similarities and top-k indices
similarities = calculate_similarity(query_features, gallery_features)
I = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]

# 5. Get image paths
query_paths = [os.path.join(test_query_dir, name) for name in query_dataset.image_files]
gallery_paths = [os.path.join(test_gallery_dir, name) for name in gallery_dataset.image_files]

# 6. Build submission format
submission = {}
for qi, qpath in enumerate(query_paths):
    qname = os.path.basename(qpath)
    retrieved = [os.path.basename(gallery_paths[i]) for i in I[qi]]
    submission[qname] = retrieved

# 7. Write JSON submission
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml-project-intro/
sub_dir = os.path.join(project_root, "submissions")
os.makedirs(sub_dir, exist_ok=True)
out_path = os.path.join(sub_dir, f"submission_{datetime.now().strftime('%Y%m%d-%H%M')}.json")
with open(out_path, 'w') as f:
    json.dump(submission, f, indent=2)
print(f"[DEBUG] Submission saved to: {out_path}")



# 8. Evaluate accuracy
top_k_acc = calculate_accuracy(similarities, true_indices=true_indices, k=k)

# 9. Save metrics
runtime = time.time() - start_time
save_metrics_json(
    model_name=resnet_version,
    top_k_accuracy=top_k_acc,
    batch_size=batch_size,
    is_finetuned=fine_tune,
    num_classes=len(train_dataset.classes),
    runtime=runtime,
    loss_function="CrossEntropyLoss",  # fisso, se usi sempre questo
    num_epochs=num_epochs,
    final_loss=final_loss
)


