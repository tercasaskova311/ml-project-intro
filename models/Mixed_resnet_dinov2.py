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
def get_data_loaders(batch_size, data_root="data", target_size=(518,518), num_workers=2):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

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
def build_model(backbone_name="resnet50", pretrained=True, device="cuda"):
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
    model.gem_pool = GeM()
    model.flatten  = nn.Flatten(1)
    return model.to(device).eval()

@torch.no_grad()
def extract_features(dataloader, model, device="cuda"):
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

def search(index_vectors, query_vecs, topk=10):
    q, g = query_vecs.astype('float32'), index_vectors.astype('float32')
    sim = np.dot(q, g.T)
    I = np.argsort(-sim, axis=1)[:, :topk]
    D = sim[np.arange(sim.shape[0])[:, None], I]
    return D, I

#
# ── 7. MAIN EXECUTION ─────────────────────────────────────────────
#
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load data
    train_loader, query_loader, gallery_loader = get_data_loaders(
        batch_size=64,
        data_root=data_root,
        target_size=(518,518),
        num_workers=4
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
    D_r, I_r = search(index_r, query_r, topk=10)
    D_d, I_d = search(index_d, query_d, topk=10)

    # Write submission files
    for name, I, paths in [
        ("resnet", I_r, paths_r),
        ("dino",   I_d, paths_d),
    ]:
        submission = []
        for qi, qpath in enumerate(query_paths):
            qname = os.path.basename(qpath)
            retrieved = [os.path.basename(paths[i]) for i in I[qi]]
            submission.append({"filename": qname, "samples": retrieved})

        out_path = os.path.join(os.path.dirname(__file__), "..", "submissions", f"submission_{name}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(submission, f, indent=2)
        print(f"Done! {len(submission)} queries written to {out_path}")




# import os
# import json
# import torch
# import timm
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F

# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder

# #
# # ── 1. DATA LOADER (with paths) ───────────────────────────────────────────────
# #
# class PathImageFolder(ImageFolder):
#     """Like ImageFolder, but returns (img_tensor, image_path) instead of (img, label)."""
#     def __getitem__(self, index):
#         path, _ = self.samples[index]
#         img = self.loader(path)
#         if self.transform:
#             img = self.transform(img)
#         return img, path


# def get_data(batch_size,
#              data_root="data",
#              target_size=(518,518),
#              num_workers=2):
#     root_dir_train = os.path.join(data_root, 'training')
#     root_dir_test  = os.path.join(data_root, 'test')

#     # Using ImageNet stats:
#     mean = [0.485, 0.456, 0.406]
#     std  = [0.229, 0.224, 0.225]

#     transform = transforms.Compose([
#         transforms.Resize(target_size),
#         transforms.CenterCrop(target_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])

#     train_ds = PathImageFolder(root_dir_train, transform=transform)
#     test_ds  = PathImageFolder(root_dir_test,  transform=transform)

#     train_loader = DataLoader(train_ds, batch_size=batch_size,
#                               shuffle=False, num_workers=num_workers)
#     test_loader  = DataLoader(test_ds,  batch_size=batch_size,
#                               shuffle=False, num_workers=num_workers)

#     return train_loader, test_loader

# #
# # ── 2. GEOMETRIC MEAN (GeM) POOLING ────────────────────────────────────────────
# #
# class GeM(nn.Module):
#     def __init__(self, p=3.0, eps=1e-6):
#         super().__init__()
#         self.p = nn.Parameter(torch.ones(1) * p)
#         self.eps = eps

#     def forward(self, x):
#         # x: (B, C, H, W)
#         x = x.clamp(min=self.eps).pow(self.p)
#         x = F.adaptive_avg_pool2d(x, (1,1))
#         return x.pow(1.0 / self.p)

# #
# # ── 3. MODEL + EMBEDDING HELPER ────────────────────────────────────────────────
# #
# def build_model(backbone_name="resnet50", pretrained=True, device="cuda"):
#     # Load timm model with no head:
#     model = timm.create_model(
#         backbone_name,
#         pretrained=pretrained,
#         num_classes=0,      # strip final fc
#         global_pool=""     # strip default pooling
#     )
#     # Attach pooling and flatten as attributes
#     model.gem_pool = GeM()
#     model.flatten  = nn.Flatten(1)
#     return model.to(device).eval()

# @torch.no_grad()
# def extract_features(dataloader, model, device="cuda"):
#     features = []
#     paths    = []
#     for imgs, img_paths in dataloader:
#         imgs = imgs.to(device)
#         # forward
#         feats = model(imgs)  # may be 4D (CNN) or 3D (ViT) or already 2D
#         # Handle different output dims:
#         if feats.ndim == 4:
#             # CNN: apply GeM pooling + flatten
#             feats = model.gem_pool(feats)
#             feats = model.flatten(feats)
#         elif feats.ndim == 3:
#             # ViT: take the [CLS] token
#             feats = feats[:, 0, :]
#         # feats is now (B, D)
#         # ℓ₂-normalise
#         feats = F.normalize(feats, p=2, dim=1)
#         features.append(feats.cpu().numpy())
#         paths.extend(img_paths)
#     features = np.vstack(features)  # (N, D)
#     return features, paths

# #
# # ── 4. BRUTE-FORCE INDEX + SEARCH ─────────────────────────────────────────────
# #
# def build_index(vectors):
#     """No-op index: just store the gallery vectors."""
#     return vectors.astype('float32')


# def search(index_vectors, query_vecs, topk=10):
#     """
#     Brute-force cosine similarity search:
#     - index_vectors: (N, D) numpy array of gallery features (ℓ₂-normalized)
#     - query_vecs:    (M, D) numpy array of query features (ℓ₂-normalized)
#     Returns distances D and indices I of shape (M, topk).
#     """
#     # Ensure float32
#     q = query_vecs.astype('float32')
#     g = index_vectors.astype('float32')
#     # Compute similarity matrix (M, N)
#     sim = np.dot(q, g.T)
#     # For each query, retrieve topk indices
#     I = np.argsort(-sim, axis=1)[:, :topk]
#     rows = np.arange(sim.shape[0])[:, None]
#     D    = sim[rows, I]
#     return D, I

# #
# # ── 5. MAIN PIPELINE ─────────────────────────────────────────────────────────
# #
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 1. load data
#     train_loader, test_loader = get_data(
#         batch_size=64,
#         data_root=os.path.join(os.path.dirname(__file__), "..", "data"),
#         target_size=(518,518),
#         num_workers=4
#     )

#     # 2. build models
#     model_resnet = build_model("resnet50", pretrained=True, device=device)
#     model_dino   = build_model("vit_small_patch14_dinov2", pretrained=True, device=device)

#     # 3. extract gallery (train) features
#     feats_r, paths_r = extract_features(train_loader, model_resnet, device)
#     feats_d, paths_d = extract_features(train_loader, model_dino,   device)

#     # 4. build indices (no FAISS)
#     index_r = build_index(feats_r)
#     index_d = build_index(feats_d)

#     # 5. extract query (test) features
#     query_r, query_paths = extract_features(test_loader, model_resnet, device)
#     query_d, _           = extract_features(test_loader, model_dino,   device)

#     # 6. search each index
#     D_r, I_r = search(index_r, query_r, topk=10)
#     D_d, I_d = search(index_d, query_d, topk=10)

#     # 7. write out two submissions (or pick the better one)
#     for name, I, paths in [
#         ("resnet", I_r, paths_r),
#         ("dino",   I_d, paths_d),
#     ]:
#         submission = []
#         for qi, qpath in enumerate(query_paths):
#             qname = os.path.basename(qpath)
#             retrieved = [os.path.basename(paths[i]) for i in I[qi]]
#             submission.append({
#                 "filename": qname,
#                 "samples":  retrieved
#             })
#         out_path = os.path.join("..", "submissions", f"submission_{name}.json")
#         with open(out_path, "w") as f:
#             json.dump(submission, f, indent=2)
#         print(f"Done! Wrote {len(submission)} queries to {out_path}")
