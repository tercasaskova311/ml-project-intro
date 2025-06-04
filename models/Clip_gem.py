import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from datetime import datetime

# -------------------- CONFIG --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
k = 10
BATCH_SIZE = 16
MODEL_NAMES = ["ViT-L/14", "ViT-B/16"]
TARGET_DIM = 512  # Common embedding size for stacking

# -------------------- Projection --------------------
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

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
        return self.transform(img), self.img_paths[idx]

# -------------------- Augmentation --------------------
def augment(img_tensor):
    flipped = torch.flip(img_tensor, dims=[2])  # horizontal flip
    return (img_tensor + flipped) / 2.0

# -------------------- Feature Extraction --------------------
@torch.no_grad()
def extract_features(models, preprocess, dataloader, projectors):
    all_features, all_paths = [], []
    for imgs, paths in tqdm(dataloader):
        imgs = imgs.to(device)
        imgs_aug = augment(imgs)

        feats = []
        for model, projector in zip(models, projectors):
            emb = model.encode_image(imgs_aug).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = projector(emb)  # project to TARGET_DIM
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb)

        combined = torch.stack(feats).mean(dim=0)
        combined = combined / combined.norm(dim=-1, keepdim=True)
        all_features.append(combined.cpu())
        all_paths.extend(paths)
    return torch.cat(all_features, dim=0), all_paths

# -------------------- Retrieval & Reranking --------------------
def retrieve_with_reranking(query_feats, gallery_feats, query_paths, gallery_paths, k):
    sims = query_feats @ gallery_feats.T
    topk_indices = sims.topk(k=k, dim=-1).indices
    results = {}
    for i, qpath in enumerate(query_paths):
        topk_paths = [gallery_paths[idx] for idx in topk_indices[i]]
        sub_gallery = gallery_feats[topk_indices[i]]
        refined_sims = torch.nn.functional.cosine_similarity(query_feats[i].unsqueeze(0), sub_gallery)
        reranked = torch.argsort(refined_sims, descending=True)
        results[os.path.basename(qpath)] = [os.path.basename(topk_paths[j]) for j in reranked]
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
    query_dir = os.path.join(DATA_DIR, "test", "query")
    gallery_dir = os.path.join(DATA_DIR, "test", "gallery")

    print("Loading models...")
    models, preprocesses, projectors = [], [], []
    for name in MODEL_NAMES:
        model, preprocess = clip.load(name, device=device)
        model.eval()
        models.append(model)
        preprocesses.append(preprocess)
        out_dim = model.visual.output_dim
        projector = Projector(out_dim, TARGET_DIM).to(device)
        projector.eval()  # no training, just matching dimensions
        projectors.append(projector)

    preprocess = preprocesses[0]

    print("Preparing datasets...")
    query_set = ImageDataset(query_dir, preprocess)
    gallery_set = ImageDataset(gallery_dir, preprocess)

    query_loader = DataLoader(query_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Extracting gallery features...")
    gallery_feats, gallery_paths = extract_features(models, preprocess, gallery_loader, projectors)

    print("Extracting query features...")
    query_feats, query_paths = extract_features(models, preprocess, query_loader, projectors)

    print("Retrieving and reranking top-k...")
    retrievals = retrieve_with_reranking(query_feats, gallery_feats, query_paths, gallery_paths, k)

    acc = top_k_accuracy(retrievals, query_paths)
    print(f"Top-{k} accuracy: {acc:.4f}")

    submission = {os.path.basename(q): retrievals[os.path.basename(q)] for q in query_paths}
    sub_dir = os.path.join(BASE_DIR, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    sub_path = os.path.join(sub_dir, f"sub_clip.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to: {sub_path}")
