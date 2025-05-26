import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from datetime import datetime

# ----- Config -----
K = 10
FINE_TUNE = False
USE_GEM = False
batch_size = 64
epochs = 5

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Image preprocessing -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- GeM Pooling -----
class GeM(torch.nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1. / self.p)

# ----- EfficientNet with GeM -----
class EfficientNetWithGeM(torch.nn.Module):
    def __init__(self, num_classes=None, fine_tune=False):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.gem_pool = GeM()
        self.flatten = torch.nn.Flatten()
        self.out_dim = base.classifier[1].in_features
        self.classifier = None
        if fine_tune and num_classes is not None:
            self.classifier = torch.nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gem_pool(x)
        x = self.flatten(x)
        if self.classifier:
            x = self.classifier(x)
        return x

# ----- Model Loaders -----
def load_model(num_classes=None, fine_tune=False):
    base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    if fine_tune and num_classes is not None:
        in_features = base_model.classifier[1].in_features
        base_model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return base_model.to(DEVICE)

def load_model_GEM(num_classes=None, fine_tune=False):
    return EfficientNetWithGeM(num_classes=num_classes, fine_tune=fine_tune).to(DEVICE)

# ----- Dataset -----
class ImagePathDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.img_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
                          if fname.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# ----- Feature Extraction -----
@torch.no_grad()
def extract_features(model, dataloader):
    features = []
    filenames = []
    for imgs, fnames in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        feats = model(imgs)
        if feats.dim() == 4:
            feats = feats.view(feats.size(0), -1)
        features.append(feats.cpu().numpy())
        filenames.extend(fnames)
    return np.vstack(features), filenames

# ----- Accuracy Calculation (safe version) -----
def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

def calculate_top_k_accuracy(results, train_lookup, k=10):
    correct = 0
    total = 0
    for query_filename, retrieved_list in results.items():
        query_class = extract_class(query_filename, train_lookup)
        if query_class == "unknown":
            continue
        retrieved_classes = [extract_class(fn, train_lookup) for fn in retrieved_list]
        if query_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"Top-{k} Accuracy (valid queries only): {acc:.4f}")
    return acc

# ----- Metrics Saver -----
def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="MSELoss",
                      num_epochs=None, final_loss=None, pooling_type=None):
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
        "pooling_type": pooling_type
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {os.path.abspath(out_path)}")

# ----- Main Execution -----
if __name__ == "__main__":
    start_time = time.time()

    # Build TRAIN_LOOKUP
    TRAIN_LOOKUP = {}
    train_root = os.path.join(DATA_DIR, 'training')
    for class_name in os.listdir(train_root):
        class_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.png')):
                TRAIN_LOOKUP[img_name] = class_name

    # Load model
    model = load_model_GEM() if USE_GEM else load_model()

    # Load data
    gallery_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/gallery"), transform)
    query_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/query"), transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size)
    query_loader = DataLoader(query_dataset, batch_size)

    gallery_feats, gallery_names = extract_features(model, gallery_loader)
    query_feats, query_names = extract_features(model, query_loader)

    # Similarity and JSON
    result = {}
    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    for i, qname in enumerate(query_names):
        topk_idx = np.argsort(sim_matrix[i])[::-1][:K]
        samples = [gallery_names[idx] for idx in topk_idx]
        result[qname] = samples

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submissions"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sub_efficientnet.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Submission saved to: {output_path}")

    topk_acc = calculate_top_k_accuracy(result, TRAIN_LOOKUP, k=K)

    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

    save_metrics_json(
        model_name="efficientnet_b0",
        top_k_accuracy=topk_acc,
        batch_size=batch_size,
        is_finetuned=FINE_TUNE,
        num_classes=None,
        runtime=total_time,
        loss_function="NotApplicable" if not FINE_TUNE else "MSELoss",
        num_epochs=epochs if FINE_TUNE else 0,
        final_loss=None,
        pooling_type="GeM" if USE_GEM else "GAP"
    )
