import os
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import top_k_accuracy, precision_at_k

# -------------------- CONFIGURATION --------------------
K = 9 # Number of retrieved images to evaluate against (Top-K)
BATCH_SIZE = 32  # Batch size for DataLoader
MODEL_NAME = "facebook/dinov2-base"  # Pretrained DINOv2 model from HuggingFace
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_DIR = os.path.join(BASE_DIR, "data_animals", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data_animals", "test", "gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "submissions", "sub_dino2.json")

# -------------------- LOAD MODEL AND PREPROCESSOR --------------------
# Load DINOv2 model and its image processor (for normalization parameters)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()  # Set model to evaluation mode (no dropout, no weight updates)

# -------------------- TRANSFORM --------------------
# Preprocessing pipeline for images (resize, to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# -------------------- DATASET CLASS --------------------
# Custom dataset class to load images and return them along with filenames
class ImagePathDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), os.path.basename(self.image_paths[idx])

# -------------------- FEATURE EXTRACTION --------------------
# Extracts embeddings from a dataset using the DINOv2 model
@torch.no_grad()
def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features, all_filenames = [], []

    for images, filenames in tqdm(loader, desc="Extracting features"):
        images = images.to(DEVICE)
        # Take the mean of the last hidden state across all patches (global image representation)
        feats = model(images).last_hidden_state.mean(dim=1)
        all_features.append(feats.cpu().numpy())
        all_filenames.extend(filenames)

    return np.vstack(all_features), all_filenames

# -------------------- CLASS EXTRACTION --------------------
# Extract class name from the filename or lookup dictionary
def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

# -------------------- METRICS SAVER --------------------
# Save model run metadata (accuracy, runtime, loss, etc.) to JSON for analysis
def save_metrics_json(
    model_name,
    top_k_accuracy_value,
    precision_value,
    batch_size,
    is_finetuned,
    num_classes=None,
    runtime=None,
    loss_function="CrossEntropyLoss",
    num_epochs=None,
    final_loss=None
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results_animals")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")

    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": K,
        "top_k_accuracy": round(top_k_accuracy_value, 4),
        "precision_at_k": round(precision_value, 4),
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

# -------------------- MAIN EXECUTION --------------------
def main(k=K):
    start_time = time.time()
    print(f"[INFO] Starting DINOv2 Retrieval (top-{k})...")

    # Build TRAIN_LOOKUP from training folder
    TRAIN_LOOKUP = {}
    train_dir = os.path.join(BASE_DIR, "data", "training")
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.png')):
                TRAIN_LOOKUP[img_name] = class_name

    # Load datasets for query and gallery
    query_dataset = ImagePathDataset(QUERY_DIR, transform)
    gallery_dataset = ImagePathDataset(GALLERY_DIR, transform)

    # Extract features from both sets
    query_features, query_filenames = extract_features(query_dataset)
    gallery_features, gallery_filenames = extract_features(gallery_dataset)

    # Retrieve top-k most similar images for each query based on cosine similarity
    results = {}
    for i, q_feat in enumerate(query_features):
        sims = cosine_similarity(q_feat.reshape(1, -1), gallery_features)[0]
        topk = np.argsort(sims)[::-1][:k]
        retrieved = [gallery_filenames[j] for j in topk]
        results[query_filenames[i]] = retrieved

    # Compute evaluation metrics
    topk_acc = top_k_accuracy(query_filenames, results, k=k)
    prec_at_k = precision_at_k(query_filenames, results, k=k)

    print(f"Top-{k} Accuracy: {topk_acc:.4f}")
    print(f"Precision@{k}: {prec_at_k:.4f}")

    # Save submission to disk
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Submission saved to: {OUTPUT_PATH}")

    # Save evaluation metrics to JSON
    runtime = time.time() - start_time
    save_metrics_json(
        model_name="dinov2-base",
        top_k_accuracy_value=topk_acc,
        precision_value=prec_at_k,
        batch_size=BATCH_SIZE,
        is_finetuned=False,
        runtime=runtime,
        loss_function="None",
        num_epochs=None,
        final_loss=None
    )

if __name__ == "__main__":
    main()
