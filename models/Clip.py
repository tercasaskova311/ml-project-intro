import os
import json
import time
import torch
import clip
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests

from utils.metrics import top_k_accuracy, precision_at_k

# ---------------- CONFIGURATION ----------------
k = 10
batch_size = 2
FINE_TUNE = True
TRAIN_LAST_LAYER_ONLY = True
epochs = 5
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- MODEL & TRANSFORM ----------------
model, preprocess = clip.load("ViT-L/14", device=device)

if not FINE_TUNE:
    for param in model.parameters():
        param.requires_grad = False
elif TRAIN_LAST_LAYER_ONLY:
    for name, param in model.named_parameters():
        if "proj" in name or "visual.proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    for param in model.parameters():
        param.requires_grad = True

class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        self.fc = torch.nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip.encode_image(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return self.fc(x)

class ImageFolderDataset(Dataset):
    def __init__(self, root_folder, transform):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        for idx, class_name in enumerate(sorted(os.listdir(root_folder))):
            class_path = os.path.join(root_folder, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        self.img_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, os.path.basename(self.img_paths[idx])

def fine_tune_clip(train_loader, model, epochs=epochs, lr=lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {final_loss:.4f}")
    return final_loss

def encode_images(image_folder, model, preprocess, batch_size=2):
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    ])
    features = []
    valid_paths = []

    model.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        current_valid_paths = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                batch_images.append(img)
                current_valid_paths.append(p)
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {p}")
        if not batch_images:
            continue
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor).float()
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            features.append(batch_features.cpu())
            valid_paths.extend(current_valid_paths)
        torch.cuda.empty_cache()
    return torch.cat(features, dim=0).to(device), valid_paths

def retrieve(query_features, gallery_features, query_paths, gallery_paths, k):
    similarities = query_features @ gallery_features.T
    topk_values, topk_indices = similarities.topk(k, dim=-1)
    results = {}
    for i in range(query_features.shape[0]):
        query_filename = os.path.basename(query_paths[i])
        retrieved_filenames = [os.path.basename(gallery_paths[idx]) for idx in topk_indices[i].cpu().numpy()]
        results[query_filename] = retrieved_filenames
    return results

def extract_class(filename, *_):
    return filename.split('_')[0]

def save_metrics_json(model_name, top_k_accuracy, precision, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None):
    """
    Save evaluation metrics to a JSON file in the 'results' directory.
    """
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
        "precision_at_k": round(precision, 4),
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
    print(f"[LOG] Metrics saved to: {os.path.abspath(out_path)}")

def submit(results, groupname="stochastic thr", url="http://65.108.245.177:3001/retrieval/"):
    """
    Submit results to evaluation server.
    """
    payload = {
        "groupname": groupname,
        "images": results
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"[SUBMIT] Server response - Accuracy: {result.get('accuracy')}")
    except requests.RequestException as e:
        print(f"[SUBMIT] Submission failed: {e}")


# ---------------- MAIN EXECUTION ----------------
start_time = time.time()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
training_dir = os.path.join(DATA_DIR, "training")
query_dir = os.path.join(DATA_DIR, "test", "query")
gallery_dir = os.path.join(DATA_DIR, "test", "gallery")

train_dataset = ImageFolderDataset(training_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

classifier = ClipClassifier(model, num_classes=len(train_dataset.class_to_idx)).to(device)
final_loss = fine_tune_clip(train_loader, classifier) if FINE_TUNE else None

with torch.no_grad():
    gallery_features, gallery_paths = encode_images(gallery_dir, model, preprocess)
    query_features, query_paths = encode_images(query_dir, model, preprocess)

retrieval_results = retrieve(query_features, gallery_features, query_paths, gallery_paths, k)

submission = {os.path.basename(q): retrieval_results[os.path.basename(q)] for q in query_paths}
sub_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "sub_clip.json")
with open(sub_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Submission saved to: {os.path.abspath(sub_path)}")

top_k_acc = top_k_accuracy(query_paths, retrieval_results, k=k)
prec_at_k = precision_at_k(query_paths, retrieval_results, k=k)
runtime = time.time() - start_time

save_metrics_json(
    model_name="clip-vit-b32",
    top_k_accuracy=top_k_acc,
    precision=prec_at_k,
    batch_size=batch_size,
    is_finetuned=FINE_TUNE,
    num_classes=len(train_dataset.class_to_idx),
    runtime=runtime,
    loss_function="CrossEntropyLoss",
    num_epochs=epochs,
    final_loss=final_loss
)

