import os
import json
import torch
import clip
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from tqdm import tqdm
import random

# -------------------- CONFIGURATION --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
k = 10
batch_size_train = 32 #This controls the number of triplets (anchor, positive, negative) that are processed simultaneously during training.
batch_size_encode = 16 #This controls the number of images processed at once during feature extraction.
EPOCHS = 10
LR = 5e-5
MODEL_NAME = "ViT-L/14"
TEMPERATURE = 0.07  # This is the scaling factor used in the InfoNCE contrastive loss to divide the cosine similarity logits before applying softmax.

# -------------------- WHY CONTRASTIVE LOSS --------------------
# Contrastive Loss (InfoNCE) maximizes similarity between
# embeddings of images from the same class (positives),
# and minimizes it for those from different classes (negatives).
# It creates a more discriminative embedding space for retrieval.

# -------------------- AUGMENTATIONS --------------------
# Augmentation increases generalization and helps the model
# learn invariant representations (e.g., same class under flip/crop).
train_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# -------------------- DATASET --------------------
class ContrastiveDataset(Dataset):
    def __init__(self, root, preprocess):
        self.data = []
        self.class_to_images = {}
        self.preprocess = preprocess

        for class_idx, class_name in enumerate(sorted(os.listdir(root))):
            folder = os.path.join(root, class_name)
            if os.path.isdir(folder):
                images = [os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.class_to_images[class_name] = images
                for img_path in images:
                    self.data.append((img_path, class_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, cls = self.data[idx]
        positive_path = random.choice([p for p in self.class_to_images[cls] if p != img_path])
        negative_class = random.choice([c for c in self.class_to_images.keys() if c != cls])
        negative_path = random.choice(self.class_to_images[negative_class])

        anchor = self.preprocess(Image.open(img_path).convert("RGB"))
        positive = self.preprocess(Image.open(positive_path).convert("RGB"))
        negative = self.preprocess(Image.open(negative_path).convert("RGB"))
        return anchor, positive, negative

# -------------------- MODEL WRAPPER --------------------
class CLIPEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x):
        with torch.no_grad():
            x = self.clip.encode_image(x).float()
        return nn.functional.normalize(x, dim=-1)

# -------------------- CONTRASTIVE LOSS (InfoNCE) --------------------
def contrastive_loss(anchor, positive, negative, temperature=TEMPERATURE):
    # Normalize embeddings
    anchor = nn.functional.normalize(anchor, dim=-1)
    positive = nn.functional.normalize(positive, dim=-1)
    negative = nn.functional.normalize(negative, dim=-1)

    # Compute similarities
    pos_sim = torch.exp(torch.sum(anchor * positive, dim=-1) / temperature)
    neg_sim = torch.exp(torch.sum(anchor * negative, dim=-1) / temperature)

    # InfoNCE loss: -log (sim(anchor, positive) / (sim(anchor, positive) + sim(anchor, negative)))
    loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
    return loss

# -------------------- TRAINING --------------------
def train_encoder(model, loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for anchor, pos, neg in loader:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            emb_anchor = model(anchor)
            emb_pos = model(pos)
            emb_neg = model(neg)

            loss = contrastive_loss(emb_anchor, emb_pos, emb_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Contrastive Loss: {total_loss / len(loader):.4f}")

# -------------------- ENCODING --------------------
def encode_folder(folder, model, preprocess):
    paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    features, valid_paths = [], []
    model.eval()

    for i in range(0, len(paths), batch_size_encode):
        batch_paths = paths[i:i + batch_size_encode]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            emb = model(batch)
        features.append(emb.cpu())
        valid_paths.extend(batch_paths)

    return torch.cat(features, dim=0), valid_paths

# -------------------- RETRIEVAL --------------------
def retrieve(query_features, gallery_features, query_paths, gallery_paths):
    sims = query_features @ gallery_features.T
    topk_indices = sims.topk(k, dim=-1).indices
    results = {}
    for i, qpath in enumerate(query_paths):
        results[os.path.basename(qpath)] = [os.path.basename(gallery_paths[idx]) for idx in topk_indices[i]]
    return results

def extract_class(fname):
    return fname.split("_")[0]

def top_k_accuracy(query_paths, retrievals):
    correct = 0
    for q in query_paths:
        qclass = extract_class(os.path.basename(q))
        retrieved_classes = [extract_class(name) for name in retrievals[os.path.basename(q)]]
        if qclass in retrieved_classes:
            correct += 1
    return correct / len(query_paths)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Load CLIP model
    clip_model, preprocess = clip.load(MODEL_NAME, device=device)
    model = CLIPEncoder(clip_model).to(device)

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "data")
    train_dir = os.path.join(data_dir, "training")
    query_dir = os.path.join(data_dir, "test", "query")
    gallery_dir = os.path.join(data_dir, "test", "gallery")

    # Dataset and training
    train_dataset = ContrastiveDataset(train_dir, train_aug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)
    train_encoder(model, train_loader)

    # Feature extraction
    query_features, query_paths = encode_folder(query_dir, model, preprocess)
    gallery_features, gallery_paths = encode_folder(gallery_dir, model, preprocess)

    # Retrieval
    retrievals = retrieve(query_features, gallery_features, query_paths, gallery_paths)
    acc = top_k_accuracy(query_paths, retrievals)
    print(f"Top-{k} accuracy: {acc:.4f}")

    # Submission
    sub_dir = os.path.join(BASE_DIR, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "sub_clip.json")
    with open(sub_path, "w") as f:
        json.dump({os.path.basename(q): retrievals[os.path.basename(q)] for q in query_paths}, f, indent=2)
    print(f"Submission saved to: {sub_path}")
