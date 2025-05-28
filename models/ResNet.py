import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, 'data')
train_dir = os.path.join(data_dir, 'training')
test_query_dir = os.path.join(data_dir, 'test', 'query')
test_gallery_dir = os.path.join(data_dir, 'test', 'gallery')

fine_tune = True
resnet_version = 'resnet34' # Change to 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'...
k = 10
batch_size = 32
num_epochs = 10
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET & TRANSFORMS ----------------
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

train_dataset = datasets.ImageFolder(train_dir, data_transforms['training'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
query_dataset = ImageDatasetWithoutLabels(test_query_dir, transform=data_transforms['test'])
query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
gallery_dataset = ImageDatasetWithoutLabels(test_gallery_dir, transform=data_transforms['test'])
gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

# ---------------- MODEL ----------------
def initialize_model(resnet_version, pretrained=True, feature_extract=True):
    model_fn = getattr(models, resnet_version)
    model = model_fn(weights="DEFAULT" if pretrained else None)
    if feature_extract:
        model.fc = nn.Identity()
    return model.to(device)

def fine_tune_model(model, num_classes):
    if isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
    else:
        raise TypeError("model.fc must be nn.Linear")
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    ).to(device)
    return model

def train_model(model, dataloader):
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

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(model, loader):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
    return np.vstack(features)

# ---------------- ACCURACY EVALUATION ----------------
def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

def calculate_top_k_accuracy(query_paths, gallery_paths, similarities, train_lookup, k):
    correct = 0
    total = 0
    for i, query_sim in enumerate(similarities):
        query_filename = os.path.basename(query_paths[i])
        query_class = extract_class(query_filename, train_lookup)
        if query_class == "unknown":
            continue
        top_k_indices = np.argsort(query_sim)[-k:][::-1]
        retrieved_filenames = [os.path.basename(gallery_paths[j]) for j in top_k_indices]
        retrieved_classes = [extract_class(name, train_lookup) for name in retrieved_filenames]
        if query_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"Top-{k} Accuracy (valid queries only): {acc:.4f}")
    return acc

# ---------------- METRICS SAVE ----------------
def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")
    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": k,
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

# ---------------- MAIN SCRIPT ----------------
start_time = time.time()
model = initialize_model(resnet_version, pretrained=True, feature_extract=not fine_tune)
if fine_tune:
    num_classes = len(train_dataset.classes)
    model = fine_tune_model(model, num_classes=num_classes)
    final_loss = train_model(model, train_loader)
else:
    model.fc = nn.Identity()
    final_loss = None
query_features = extract_features(model, query_loader)
gallery_features = extract_features(model, gallery_loader)
similarities = cosine_similarity(query_features, gallery_features)
I = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
query_paths = [os.path.join(test_query_dir, name) for name in query_dataset.image_files]
gallery_paths = [os.path.join(test_gallery_dir, name) for name in gallery_dataset.image_files]

# TRAIN_LOOKUP for class inference
TRAIN_LOOKUP = {}
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.jpg', '.png')):
            TRAIN_LOOKUP[fname] = class_name

submission = {}
for qi, qpath in enumerate(query_paths):
    qname = os.path.basename(qpath)
    retrieved = [os.path.basename(gallery_paths[i]) for i in I[qi]]
    submission[qname] = retrieved
sub_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(sub_dir, exist_ok=True)
out_path = os.path.join(sub_dir, f"sub_{resnet_version}.json")
with open(out_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Done! {len(submission)} queries written to: {out_path}")
top_k_acc = calculate_top_k_accuracy(query_paths, gallery_paths, similarities, TRAIN_LOOKUP, k)
runtime = time.time() - start_time
save_metrics_json(
    model_name=resnet_version,
    top_k_accuracy=top_k_acc,
    batch_size=batch_size,
    is_finetuned=fine_tune,
    num_classes=len(train_dataset.classes),
    runtime=runtime,
    loss_function="CrossEntropyLoss",
    num_epochs=num_epochs,
    final_loss=final_loss
)

# submission is the dictionary that we have to give to result function
