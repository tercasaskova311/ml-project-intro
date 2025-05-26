# models/dino2_retrieval_finetune.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

# ---------------- CONFIGURATION ----------------
K = 5                          # Top-k images to retrieve
EPOCHS = 5                     # Number of training epochs
BATCH_SIZE = 16                # Batch size during training/inference
LR = 1e-4                      # Learning rate for optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to dataset and output
MODEL_NAME = "facebook/dinov2-base"  # Pretrained ViT-B/14 model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "training")
QUERY_DIR = os.path.join(BASE_DIR, "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data", "test", "gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "submissions", "sub_dino2_finetune.json")

# ---------------- TRANSFORM ----------------
# Apply same transformations that DINOv2 expects (resizing, normalization)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ---------------- MODEL: DINO + HEAD ----------------
# Define a simple classification model on top of frozen DINOv2
class DinoClassifier(nn.Module):
    def __init__(self, dinov2, num_classes):
        super().__init__()
        self.dino = dinov2
        # Freeze the pretrained DINO backbone
        for param in self.dino.parameters():
            param.requires_grad = False
        # Add a trainable linear layer on top (for class prediction)
        self.head = nn.Linear(self.dino.config.hidden_size, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.dino(x).last_hidden_state.mean(dim=1)
        return self.head(feats)

    def extract_features(self, x):
        # Reusable feature extractor (for retrieval)
        with torch.no_grad():
            return self.dino(x).last_hidden_state.mean(dim=1)

# ---------------- TRAINING FUNCTION ----------------
# Trains only the linear classification head
def train_head(model, dataloader, optimizer, criterion):
    model.train()
    final_loss = None
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        final_loss = running_loss / len(dataloader)
        print(f"Loss: {final_loss:.4f}")
    return final_loss


# ---------------- RETRIEVAL DATASET ----------------
# Custom dataset to load query/gallery images by filename
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpg", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), os.path.basename(path)

# ---------------- FEATURE EXTRACTOR ----------------
# Used for both query and gallery
def extract_features(model, folder):
    dataset = ImagePathDataset(folder, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    features, filenames = [], []

    model.eval()
    with torch.no_grad():
        for images, names in tqdm(loader, desc=f"Extracting from {os.path.basename(folder)}"):
            images = images.to(DEVICE)
            feats = model.extract_features(images).cpu().numpy()
            features.append(feats)
            filenames.extend(names)

    return np.vstack(features), filenames

import os
import json
from datetime import datetime

def save_metrics_json(
    model_name,
    top_k_accuracy,
    batch_size,
    is_finetuned,
    num_classes=None,
    runtime=None,
    loss_function="CrossEntropyLoss",
    num_epochs=None,
    final_loss=None
):
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
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
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📁 Metrics saved to: {os.path.abspath(out_path)}")



# ---------------- MAIN LOGIC ----------------
def main():
    start_time = time.time()

    print("[1] Loading pretrained DINOv2 model...")
    dinov2 = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    print("[2] Loading training dataset...")
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(train_dataset.classes)

    print("[3] Building model with classification head...")
    model = DinoClassifier(dinov2, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("[4] Fine-tuning head on training set...")
    final_loss = train_head(model, train_loader, optimizer, criterion)

    print("[5] Extracting features for retrieval...")
    q_feats, q_names = extract_features(model, QUERY_DIR)
    g_feats, g_names = extract_features(model, GALLERY_DIR)

    print("[6] Performing top-k retrieval for each query...")
    submission = {}
    correct = 0
    for i, qf in enumerate(q_feats):
        sims = cosine_similarity(qf.reshape(1, -1), g_feats)[0]
        topk_idx = np.argsort(sims)[::-1][:K]
        retrieved = [g_names[j] for j in topk_idx]

        query_filename = q_names[i]
        submission[query_filename] = retrieved

        # Accuracy check
        q_class = query_filename.split("_")[0]
        retrieved_classes = [name.split("_")[0] for name in retrieved]
        if q_class in retrieved_classes:
            correct += 1


    top_k_acc = correct / len(q_names)
    print(f"Top-{K} Accuracy: {top_k_acc:.4f}")

    print(f"[7] Saving output JSON to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(submission, f, indent=2)

    runtime = time.time() - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    print("[8] Saving metrics JSON...")
    save_metrics_json(
    model_name="dinov2-base",
    top_k_accuracy=top_k_acc,
    batch_size=BATCH_SIZE,
    is_finetuned=True,
    num_classes=num_classes,
    runtime=runtime,
    loss_function="CrossEntropyLoss",
    num_epochs=EPOCHS,
    final_loss=final_loss
)


    print("Retrieval pipeline with metrics logging complete.")


# ---------------- RUN SCRIPT ----------------
if __name__ == "__main__":
    main()
