import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json
from datetime import datetime
from torch.utils.data import Dataset
from PIL import Image


# Config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml-project-intro/
data_dir = os.path.join(BASE_DIR, 'data')
train_dir = os.path.join(data_dir, 'training')
test_query_dir = os.path.join(data_dir, 'test', 'query')
test_gallery_dir = os.path.join(data_dir, 'test', 'gallery')


fine_tune = False  # Set to False to skip training and only extract features
resnet_version = 'resnet34'  # Change to: 'resnet18', 'resnet34', 'resnet50', or 'resnet101'
k=10
batch_size = 16
num_epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(os.path.exists(train_dir))
# print(os.listdir(train_dir))  # will list subfolders/classes for training set
# print(os.listdir(test_query_dir))



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

import os

class ImageDatasetWithoutLabels(Dataset):
    def __init__(self, folder, transform=None):
        self.image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]
    


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


def get_test_loader(dir_path):
    dataset = datasets.ImageFolder(dir_path, data_transforms['test'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

query_dataset = ImageDatasetWithoutLabels(test_query_dir, transform=data_transforms['test'])
query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)


def initialize_model(resnet_version, pretrained=True, feature_extract=True):
    model_fn = getattr(models, resnet_version)
    model = model_fn(pretrained=pretrained)

    if feature_extract:
        # For feature extraction, keep the backbone and remove the classification head
        model.fc = nn.Identity()

    return model.to(device)

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


def fine_tune_model(model, num_classes):
    # Safely extract in_features from Linear layer
    if isinstance(model.fc, nn.Sequential):
        # Search for the first Linear layer in the Sequential block
        for layer in model.fc:
            if isinstance(layer, nn.Linear):
                num_features = layer.in_features
                break
        else:
            raise ValueError("No Linear layer found in model.fc Sequential block.")
    elif isinstance(model.fc, nn.Linear):
        num_features = model.fc.in_features
    else:
        raise TypeError("model.fc must be either nn.Linear or nn.Sequential")

    # Replace with a new Sequential block for classification
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

def calculate_accuracy(similarities, k):
    correct = 0
    for i, query_sim in enumerate(similarities):
        top_k_indices = np.argsort(query_sim)[-k:][::-1]
        if i in top_k_indices:
            correct += 1
    accuracy = correct / len(similarities)
    print(f"Top-{k} Accuracy: {accuracy:.4f}")
    return accuracy

def visualize_retrieved_images(query_loader, gallery_loader, similarities, k):
    query_images = get_all_images(query_loader)
    gallery_images = get_all_images(gallery_loader)

    for i, query_sim in enumerate(similarities):
        top_k_indices = np.argsort(query_sim)[-k:][::-1]
        plt.figure(figsize=(10, 2))
        plt.subplot(1, k + 1, 1)
        plt.imshow(denormalize(query_images[i]).permute(1, 2, 0))
        plt.title("Query")
        plt.axis('off')
        for j, idx in enumerate(top_k_indices):
            plt.subplot(1, k + 1, j + 2)
            plt.imshow(denormalize(gallery_images[idx]).permute(1, 2, 0))
            plt.title(f"Top {j+1}")
            plt.axis('off')
        plt.show()


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





import time
start_time = time.time()

# 1. Initialize model
model = initialize_model(resnet_version, pretrained=True, feature_extract=not fine_tune)

# 2. Optional fine-tuning
if fine_tune:
    num_classes = len(train_dataset.classes)
    model = fine_tune_model(model, num_classes=num_classes)
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
out_path = os.path.join(sub_dir, f"sub_{resnet_version}.json")
print(f"[DEBUG] Submission saved to: {out_path}")

with open(out_path, "w") as f:
    json.dump(submission, f, indent=2)

print(f"Done! {len(submission)} queries written to: {out_path}")


# 8. Evaluate accuracy
top_k_acc = calculate_accuracy(similarities, k)

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


# 10. Optional: Visualize retrieval results
visualize_retrieved_images(query_loader, gallery_loader, similarities, k)
