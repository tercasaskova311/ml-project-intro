import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

# -------------------- CONFIG --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
k = 10
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
USE_GEM = True  # Flag to choose between GeM and GAP
GEM_P = 3.0  # Initial p value for GeM pooling
FINE_TUNE = True  # Flag to control fine-tuning
MODEL_NAME = "ViT-B/16"  # Default model

# -------------------- Pooling Layers --------------------
class GeM(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
    
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))

# -------------------- CLIP Classifier --------------------
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, use_gem=USE_GEM):
        super().__init__()
        self.clip = clip_model
        self.use_gem = use_gem
        
        # Initialize pooling layer based on flag
        self.pooling = GeM() if use_gem else GAP()
        
        # Projection head
        self.fc = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, clip_model.visual.output_dim),
            nn.LayerNorm(clip_model.visual.output_dim),
            nn.ReLU(),
            nn.Linear(clip_model.visual.output_dim, num_classes)
        )

    def forward(self, x):
        # Ensure input has the correct dtype
        expected_dtype = self.clip.visual.conv1.weight.dtype
        x = x.to(dtype=expected_dtype)
        
        # Extract features through CLIP's visual encoder
        features = self.extract_features(x)
        return self.fc(features)

    def extract_features(self, x):
        # Remove no_grad to allow proper fine-tuning
        x = self.clip.encode_image(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

# -------------------- Dataset --------------------
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
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.img_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, os.path.basename(self.img_paths[idx])

# -------------------- Training --------------------
def fine_tune_clip(train_loader, model, epochs=EPOCHS, lr=LR):
    model.train()
    
    # Different learning rates for different parts of the model
    optimizer = torch.optim.AdamW([
        {'params': model.clip.parameters(), 'lr': lr * 0.1},  # Lower LR for CLIP
        {'params': model.fc.parameters(), 'lr': lr}  # Higher LR for new layers
    ])
    
    ce_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = ce_loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    return epoch_loss

# -------------------- Feature Extraction --------------------
@torch.no_grad()
def extract_features(model, dataloader):
    model.eval()
    all_features = []
    all_paths = []
    
    for images, _, paths in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model.extract_features(images)
        all_features.append(features.cpu())
        all_paths.extend(paths)
    
    return torch.cat(all_features, dim=0), all_paths

# -------------------- Retrieval --------------------
def retrieve(query_features, gallery_features, query_paths, gallery_paths, k):
    similarities = query_features @ gallery_features.T
    topk_values, topk_indices = similarities.topk(k, dim=-1)
    results = {}
    for i in range(query_features.shape[0]):
        query_filename = os.path.basename(query_paths[i])
        retrieved_filenames = [os.path.basename(gallery_paths[idx]) for idx in topk_indices[i].cpu().numpy()]
        results[query_filename] = retrieved_filenames
    return results

# -------------------- Main --------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data_animals")
    training_dir = os.path.join(DATA_DIR, "training")
    query_dir = os.path.join(DATA_DIR, "test", "query")
    gallery_dir = os.path.join(DATA_DIR, "test", "gallery")

    # Load CLIP model
    model, preprocess = clip.load(MODEL_NAME, device=device)
    
    # Load datasets
    train_dataset = ImageFolderDataset(training_dir, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    query_dataset = ImageFolderDataset(query_dir, transform=preprocess)
    gallery_dataset = ImageFolderDataset(gallery_dir, transform=preprocess)
    
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize classifier
    classifier = ClipClassifier(model, num_classes=len(train_dataset.class_to_idx), use_gem=USE_GEM).to(device)
    
    # Fine-tune if enabled
    if FINE_TUNE:
        print("Fine-tuning model...")
        final_loss = fine_tune_clip(train_loader, classifier)
    
    # Extract features
    print("Extracting features...")
    gallery_features, gallery_paths = extract_features(classifier, gallery_loader)
    query_features, query_paths = extract_features(classifier, query_loader)
    
    # Retrieve results
    print("Performing retrieval...")
    results = retrieve(query_features.to(device), gallery_features.to(device), query_paths, gallery_paths, k)
    
    # Save results
    submission = {os.path.basename(q): results[os.path.basename(q)] for q in query_paths}
    sub_dir = os.path.join(BASE_DIR, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    sub_path = os.path.join(sub_dir, f"sub_clip_gem_{timestamp}.json")
    
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to: {sub_path}")
