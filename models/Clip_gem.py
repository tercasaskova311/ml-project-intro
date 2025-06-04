import os
import json
import torch
import clip
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

# -------------------- CONFIG --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
k = 10
MODEL_NAME = "ViT-L/14"
BATCH_SIZE = 16

# -------------------- GeM Pooling --------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                                              (x.size(-2), x.size(-1))).pow(1. / self.p)

# -------------------- Dataset --------------------
class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.img_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.img_paths[idx]

# -------------------- Feature Extraction --------------------
@torch.no_grad()
def extract_features(model, dataloader, gem_pool):
    model.eval()
    all_features, all_paths = [], []
    for imgs, paths in tqdm(dataloader):
        imgs = imgs.to(device)
        feats = model.encode_image(imgs)  # [B, D]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        feats = gem_pool(feats).squeeze(-1).squeeze(-1)  # [B, D]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu())
        all_paths.extend(paths)
    return torch.cat(all_features, dim=0), all_paths

# -------------------- Retrieval --------------------
def retrieve(query_features, gallery_features, query_paths, gallery_paths, k):
    sims = query_features @ gallery_features.T  # cosine similarity
    topk_indices = sims.topk(k, dim=-1).indices
    results = {}
    for i, qpath in enumerate(query_paths):
        results[os.path.basename(qpath)] = [os.path.basename(gallery_paths[idx])
                                            for idx in topk_indices[i]]
    return results

# -------------------- Evaluation --------------------
def extract_class(filename):
    return filename.split("_")[0]

def top_k_accuracy(retrievals, query_paths):
    correct = 0
    for qpath in query_paths:
        qname = os.path.basename(qpath)
        qclass = extract_class(qname)
        retrieved = retrievals[qname]
        retrieved_classes = [extract_class(r) for r in retrieved]
        if qclass in retrieved_classes:
            correct += 1
    return correct / len(query_paths)

# -------------------- Main --------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    train_dir = os.path.join(DATA_DIR, "training")
    query_dir = os.path.join(DATA_DIR, "test", "query")
    gallery_dir = os.path.join(DATA_DIR, "test", "gallery")

    # Load CLIP
    model, preprocess = clip.load(MODEL_NAME, device=device)
    gem = GeM().to(device)

    # Datasets and loaders
    query_set = ImageDataset(query_dir, preprocess)
    gallery_set = ImageDataset(gallery_dir, preprocess)

    query_loader = DataLoader(query_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Feature extraction
    print("Extracting gallery features...")
    gallery_feats, gallery_paths = extract_features(model, gallery_loader, gem)

    print("Extracting query features...")
    query_feats, query_paths = extract_features(model, query_loader, gem)

    # Retrieval
    print("Retrieving top-k...")
    retrievals = retrieve(query_feats, gallery_feats, query_paths, gallery_paths, k)

    # Evaluation
    acc = top_k_accuracy(retrievals, query_paths)
    print(f"Top-{k} accuracy: {acc:.4f}")

    # Submission saving
    submission = {os.path.basename(q): retrievals[os.path.basename(q)] for q in query_paths}
    sub_dir = os.path.join(BASE_DIR, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    sub_path = os.path.join(sub_dir, f"sub_clip.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to: {sub_path}")