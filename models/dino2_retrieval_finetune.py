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
import requests
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import top_k_accuracy, precision_at_k

# ---------------- CONFIGURATION ----------------
K = 9  # top-k for retrieval
EPOCHS = 10 # number of fine-tuning epochs
BATCH_SIZE = 32  # batch size for training and feature extraction
LR = 5e-5  # learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "facebook/dinov2-base"

# Base directory and dataset paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data_animals", "training")
QUERY_DIR = os.path.join(BASE_DIR, "data_animals", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data_animals", "test", "gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "submissions", "sub_dino2_finetune.json")

# ---------------- TRANSFORM ----------------
# Use model-specific normalization
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ---------------- MODEL DEFINITION ----------------
# Wrap DINOv2 with a classification head for fine-tuning and feature extraction
class DinoClassifier(nn.Module):
    def __init__(self, dinov2, num_classes):
        super().__init__()
        self.dino = dinov2
        for param in self.dino.parameters():
            param.requires_grad = False  # Freeze all backbone weights

        # Classification head (fine-tunable layer)
        self.head = nn.Linear(self.dino.config.hidden_size, num_classes)

    def forward(self, x):
        # During classification training: use frozen features + classification head
        with torch.no_grad():
            feats = self.dino(pixel_values=x).last_hidden_state.mean(dim=1)
        return self.head(feats)

    def extract_features(self, x):
        """
        Extract normalized embeddings from images (used during retrieval).
        This method runs the backbone and classification head (if attached),
        then normalizes the resulting embeddings.
        """
        self.eval()
        with torch.no_grad():
            # Forward through frozen DINO backbone
            feats = self.dino(pixel_values=x).last_hidden_state.mean(dim=1)

            # Pass through the classification head (optional but improves separation)
            feats = self.head(feats)

            # Normalize features to unit length for cosine similarity
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats


# ---------------- TRAINING ----------------
# Fine-tune the classification head using CrossEntropy loss
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

# ---------------- CUSTOM DATASET ----------------
# Loads image paths and returns (image_tensor, filename)
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

# ---------------- FEATURE EXTRACTION ----------------
# Extract embeddings using the feature extractor
def get_features_from_dir(model, folder):
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
# Used to extract class name from filenames
def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

# ---------------- METRICS JSON SAVE ----------------
# Saves all model metrics into a JSON file for analysis
def save_metrics_json(model_name, top_k_accuracy, precision, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results_animals")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")

    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": K,
        "top_k_accuracy": round(top_k_accuracy, 4),
        "precision_at_k": round(precision, 4),
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

# ---------------- SUBMIT TO SERVER ----------------
def submit(results, groupname, url="http://65.108.245.177:3001/retrieval/"):
    res = {"groupname": groupname, "images": results}
    response = requests.post(url, json=res)
    try:
        result = response.json()
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# ---------------- MAIN ----------------
def main():
    start_time = time.time()

    # Load pretrained DINOv2
    print("[1] Loading pretrained DINOv2 model...")
    dinov2 = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Load training data and build model
    print("[2] Loading training dataset...")
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(train_dataset.classes)

    print("[3] Building model with classification head...")
    model = DinoClassifier(dinov2, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Fine-tune
    print("[4] Fine-tuning classification head...")
    final_loss = train_head(model, train_loader, optimizer, criterion)

    # Prepare class lookup
    print("[5] Building TRAIN_LOOKUP dictionary...")
    train_lookup = {}
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".jpg", ".png")):
                train_lookup[img_name] = class_name

    # Extract features
    print("[6] Extracting features for retrieval...")
    q_feats, q_names = get_features_from_dir(model, QUERY_DIR)
    g_feats, g_names = get_features_from_dir(model, GALLERY_DIR)

    # Retrieval + submission dict
    print("[7] Performing top-k retrieval for each query...")
    submission = {}
    for i, qf in enumerate(q_feats):
        sims = cosine_similarity(qf.reshape(1, -1), g_feats)[0]
        topk_idx = np.argsort(sims)[::-1][:K]
        retrieved = [g_names[j] for j in topk_idx]
        submission[q_names[i]] = retrieved

    # Evaluate metrics using standardized utility functions
    print("[8] Calculating metrics...")
    topk_acc = top_k_accuracy(q_names, submission, k=K)
    prec_at_k = precision_at_k(q_names, submission, k=K)
    print(f"Top-{K} Accuracy: {topk_acc:.4f}")
    print(f"Precision@{K}: {prec_at_k:.4f}")

    # Save submission file
    print("[9] Saving output JSON...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(submission, f, indent=2)

    # Save metrics
    runtime = time.time() - start_time
    print("[10] Saving metrics JSON...")
    save_metrics_json(
        model_name="dinov2-base",
        top_k_accuracy=topk_acc,
        precision=prec_at_k,
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
