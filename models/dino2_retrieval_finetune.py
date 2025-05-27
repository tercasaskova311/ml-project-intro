import os
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import torch.nn as nn

# ---------------- CONFIGURATION ----------------
K = 10
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "facebook/dinov2-base"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "training")
QUERY_DIR = os.path.join(BASE_DIR, "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data", "test", "gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "submissions", "sub_dino2_finetune.json")

# ---------------- TRANSFORM ----------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ---------------- MODEL: DINO + HEAD ----------------
class DinoClassifier(nn.Module):
    def __init__(self, dinov2, num_classes):
        super().__init__()
        self.dino = dinov2
        for param in self.dino.parameters():
            param.requires_grad = False
        self.head = nn.Linear(self.dino.config.hidden_size, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.dino(x).last_hidden_state.mean(dim=1)
        return self.head(feats)

    def extract_features(self, x):
        with torch.no_grad():
            return self.dino(x).last_hidden_state.mean(dim=1)

# ---------------- TRAINING FUNCTION ----------------
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

# ---------------- CLASS EXTRACTION ----------------
def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

# ---------------- METRICS SAVE ----------------
def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None):
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
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {os.path.abspath(out_path)}")

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

    print("[5] Building TRAIN_LOOKUP dictionary...")
    train_lookup = {}
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".jpg", ".png")):
                train_lookup[img_name] = class_name

    print("[6] Extracting features for retrieval...")
    q_feats, q_names = extract_features(model, QUERY_DIR)
    g_feats, g_names = extract_features(model, GALLERY_DIR)

    print("[7] Performing top-k retrieval for each query...")
    submission = {}
    correct = 0
    total = 0
    for i, qf in enumerate(q_feats):
        sims = cosine_similarity(qf.reshape(1, -1), g_feats)[0]
        topk_idx = np.argsort(sims)[::-1][:K]
        retrieved = [g_names[j] for j in topk_idx]

        query_filename = q_names[i]
        submission[query_filename] = retrieved

        q_class = extract_class(query_filename, train_lookup)
        if q_class == "unknown":
            continue
        retrieved_classes = [extract_class(name, train_lookup) for name in retrieved]
        if q_class in retrieved_classes:
            correct += 1
        total += 1

    top_k_acc = correct / total if total > 0 else 0.0
    print(f"Top-{K} Accuracy (valid queries only): {top_k_acc:.4f}")

    print(f"[8] Saving output JSON to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(submission, f, indent=2)

    runtime = time.time() - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    print("[9] Saving metrics JSON...")
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

if __name__ == "__main__":
    main()
