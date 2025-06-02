import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import json
from torchvision.datasets import ImageFolder
import torch.optim as optim
from datetime import datetime
from collections import defaultdict

# CONFIG
k = 10
batch_size = 32
image_size = (224, 224)
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
FINE_TUNE = True
TRAIN_LAST_LAYER_ONLY = True
epochs = 5
learning_rate = 1e-5
USE_GEM_POOLING = True  # 👈 Toggle this to True to use GeM instead of GAP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
# ADDITION: GeM class
class GeM(torch.nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                            (x.size(-2), x.size(-1))).pow(1./self.p)

# ─────────────────────────────────────────────────────────────────────────────

# Paths
cwd = os.getcwd()
if os.path.isdir(os.path.join(cwd, 'data', 'training')):
    project_root = cwd
elif os.path.isdir(os.path.join(cwd, '..', 'data', 'training')):
    project_root = os.path.abspath(os.path.join(cwd, '..'))
else:
    raise RuntimeError("Folder 'data/training' not found. Run from the project root.")

# Build TRAIN_LOOKUP from training folder
TRAIN_LOOKUP = {}
train_root = os.path.join(project_root, 'data', 'training')
for class_name in os.listdir(train_root):
    class_dir = os.path.join(train_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.png')):
            TRAIN_LOOKUP[img_name] = class_name

# Paths
root_dir_test = os.path.join(project_root, 'data', 'test')
query_folder = os.path.join(root_dir_test, 'query')
gallery_folder = os.path.join(root_dir_test, 'gallery')

# Dataset
class ImagePathDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

# Transform
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

# GoogLeNet model
def build_googlenet_extractor(device):
    weights = torchvision.models.GoogLeNet_Weights.DEFAULT
    model = torchvision.models.googlenet(weights=weights, aux_logits=True)
    model.aux1 = torch.nn.Identity()
    model.aux2 = torch.nn.Identity()
    if USE_GEM_POOLING:
        model.avgpool = GeM()
    else:
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    model.fc = torch.nn.Identity()
    return model.to(device).eval()

# Fine-tuning
def fine_tune_model(model, train_loader, device, num_classes, train_last_layer_only=True, epochs=epochs, learning_rate=learning_rate):
    print(f"[FT] Starting fine-tuning for {epochs} epochs...")
    model.train()
    model = model.to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)

    if train_last_layer_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    final_loss = None

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        final_loss = running_loss / len(train_loader)
        print(f"[FT] Epoch {epoch+1}: Loss={final_loss:.3f}, Acc={acc:.2f}%")

    print("[FT] Fine-tuning complete.")
    return model.eval(), final_loss

# Feature extraction
@torch.no_grad()
def extract_embeddings(loader, model, device):
    embeddings = []
    for images, _ in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model(images)
        embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)

# Retrieval
def retrieve_topk(query_embs, gallery_embs, query_paths, gallery_paths, k):
    query_embs = F.normalize(query_embs, dim=1)
    gallery_embs = F.normalize(gallery_embs, dim=1)
    sim_matrix = query_embs @ gallery_embs.T
    topk_indices = sim_matrix.topk(k, dim=1, largest=True)[1]

    results = {}
    for i, indices in enumerate(topk_indices):
        query_filename = os.path.basename(query_paths[i])
        top_filenames = [os.path.basename(gallery_paths[j]) for j in indices.tolist()]
        results[query_filename] = top_filenames
    return results

# Accuracy calculation
def calculate_top_k_accuracy(results):
    def extract_class(filename):
        if "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2 and not parts[0].isdigit():
                return "_".join(parts[:-1])
        base = os.path.basename(filename)
        return TRAIN_LOOKUP.get(base, "unknown")

    correct = 0
    total = len(results)

    for query_filename, retrieved_list in results.items():
        query_class = extract_class(query_filename)
        retrieved_classes = [extract_class(fn) for fn in retrieved_list]
        if query_class in retrieved_classes:
            correct += 1

    acc = correct / total
    print(f" Top-{k} Accuracy: {acc:.4f}")
    return acc

# Metrics saving
def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned, num_classes=None, runtime=None, loss_function="MSELoss", num_epochs=None, final_loss=None, pooling_type=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        "pooling_type": "GeM" if USE_GEM_POOLING else "GAP"
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {os.path.abspath(out_path)}")

import requests
import json
def submit(results, groupname, url="http://65.108.245.177:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# Main
if __name__ == '__main__':
    import time
    start_time = time.time()

    query_dataset = ImagePathDataset(query_folder, transform=transform)
    gallery_dataset = ImagePathDataset(gallery_folder, transform=transform)
    query_loader = DataLoader(query_dataset, batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size, shuffle=False)

    if FINE_TUNE:
        print("[FT] Loading training data...")
        train_dir = os.path.join(project_root, 'data', 'training')
        train_dataset = ImageFolder(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT, aux_logits=True)
        model.aux1 = torch.nn.Identity()
        model.aux2 = torch.nn.Identity()
        if USE_GEM_POOLING:
            model.avgpool = GeM()
        else:
            model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        model, final_loss = fine_tune_model(model, train_loader, device, num_classes=len(train_dataset.classes), train_last_layer_only=TRAIN_LAST_LAYER_ONLY, epochs=epochs, learning_rate=learning_rate)
    else:
        model = build_googlenet_extractor(device)
        final_loss = None

    query_embs = extract_embeddings(query_loader, model, device)
    gallery_embs = extract_embeddings(gallery_loader, model, device)

    results = retrieve_topk(query_embs, gallery_embs, query_dataset.paths, gallery_dataset.paths, k)

    output_dir = os.path.join(project_root, 'submissions')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'sub_googlenet.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    print(f" Total runtime: {total_time:.2f} seconds")
    print(f" Submission saved to: {output_file}")

    print("[] Calculating Top-K accuracy...")
    topk_acc = calculate_top_k_accuracy(results)

    total_time = time.time() - start_time
    print(f"⏱ Total runtime: {total_time:.2f} seconds")

    num_classes = len(train_dataset.classes) if FINE_TUNE else None
    save_metrics_json(
        model_name="googlenet",
        top_k_accuracy=topk_acc,
        batch_size=batch_size,
        is_finetuned=FINE_TUNE,
        num_classes=num_classes,
        runtime=total_time,
        loss_function="CrossEntropyLoss" if FINE_TUNE else "None",
        num_epochs=epochs if FINE_TUNE else None,
        final_loss=final_loss
    )
    submit(results, groupname="Stochastic_thr")  