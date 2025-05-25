import os
import json
import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

k=10  # Number of top results to retrieve
num_workers = 2  # Number of workers for data loading
batch_size = 8 # Batch size for data loading
image_size = (224, 224)
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
FINE_TUNE = False  # Set to False to skip training
TRAIN_LAST_LAYER_ONLY = True  # Set to False to fine-tune entire model
epochs = 5
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#
# ── 1. FLAT DATASET ───────────────────────────────────────────────
#
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

#
# ── 2. AUTO DETECTION ─────────────────────────────────────────────
#
def load_dataset_auto(root, transform):
    has_subfolders = any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root))
    if has_subfolders:
        return ImageFolder(root, transform=transform)
    else:
        return FlatFolderDataset(root, transform=transform)

#
# ── 3. DATA LOADER ────────────────────────────────────────────────
#
def get_data_loaders(batch_size, data_root="data", target_size=(518,518), num_workers=num_workers):
    mean = normalize_mean
    std  = normalize_std

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    gallery_ds      = load_dataset_auto(os.path.join(data_root, "training"), transform)
    query_ds        = load_dataset_auto(os.path.join(data_root, "test", "query"), transform)
    gallery_test_ds = load_dataset_auto(os.path.join(data_root, "test", "gallery"), transform)

    return (
        DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(gallery_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

#
# ── 4. GEOMETRIC MEAN POOLING ────────────────────────────────────
#
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1,1))
        return x.pow(1.0 / self.p)

#
# ── 5. MODEL BUILDER ─────────────────────────────────────────────
#
def build_model(backbone_name="resnet50", pretrained=True, device=device):
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
    model.gem_pool = GeM()
    model.flatten  = nn.Flatten(1)
    return model.to(device).eval()

@torch.no_grad()
def extract_features(dataloader, model, device=device):
    features, paths = [], []
    for imgs, img_paths in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs)
        if feats.ndim == 4:
            feats = model.gem_pool(feats)
            feats = model.flatten(feats)
        elif feats.ndim == 3:
            feats = feats[:, 0, :]
        feats = F.normalize(feats, p=2, dim=1)
        features.append(feats.cpu().numpy())
        paths.extend(img_paths)
    return np.vstack(features), paths

#
# ── 6. BRUTE FORCE SEARCH ────────────────────────────────────────
#
def build_index(vectors): return vectors.astype('float32')

def search(index_vectors, query_vecs, k):
    q, g = query_vecs.astype('float32'), index_vectors.astype('float32')
    sim = np.dot(q, g.T)
    I = np.argsort(-sim, axis=1)[:, :k]
    D = sim[np.arange(sim.shape[0])[:, None], I]
    return D, I

#
# ── 7. MAIN EXECUTION ─────────────────────────────────────────────
#
if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load data
    train_loader, query_loader, gallery_loader = get_data_loaders(
        batch_size=batch_size,
        data_root=data_root,
        target_size=(518,518),
        num_workers=num_workers
    )

    # Build models
    model_resnet = build_model("resnet50", pretrained=True, device=device)
    model_dino   = build_model("vit_small_patch14_dinov2", pretrained=True, device=device)

    # Extract gallery features
    feats_r, paths_r = extract_features(gallery_loader, model_resnet, device)
    feats_d, paths_d = extract_features(gallery_loader, model_dino, device)

    index_r = build_index(feats_r)
    index_d = build_index(feats_d)

    # Extract query features
    query_r, query_paths = extract_features(query_loader, model_resnet, device)
    query_d, _           = extract_features(query_loader, model_dino, device)

    # Search
    D_r, I_r = search(index_r, query_r, k)
    D_d, I_d = search(index_d, query_d, k)

    # Write submission files
    for name, I, paths in [
        ("resnet_mix", I_r, paths_r),
        ("dino_mix",   I_d, paths_d),
    ]:
        submission = {}
        for qi, qpath in enumerate(query_paths):
            qname = os.path.basename(qpath)
            retrieved = [os.path.basename(paths[i]) for i in I[qi]]
            submission[qname] = retrieved


        out_path = os.path.join(os.path.dirname(__file__), "..", "submissions", f"sub_{name}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(submission, f, indent=2)
        print(f"Done! {len(submission)} queries written to {out_path}")




