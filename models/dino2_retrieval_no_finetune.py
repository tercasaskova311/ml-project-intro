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

# -------------------- CONFIGURATION --------------------
K = 10
BATCH_SIZE = 32
MODEL_NAME = "facebook/dinov2-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_DIR = os.path.join(BASE_DIR, "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data", "test", "gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "submissions", "sub_dino2.json")

# -------------------- LOAD MODEL --------------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# -------------------- DATASET --------------------
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
@torch.no_grad()
def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features, all_filenames = [], []

    for images, filenames in tqdm(loader, desc="Extracting features"):
        images = images.to(DEVICE)
        feats = model(images).last_hidden_state.mean(dim=1)
        all_features.append(feats.cpu().numpy())
        all_filenames.extend(filenames)

    return np.vstack(all_features), all_filenames

# -------------------- CLASS EXTRACTION --------------------
def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

# -------------------- METRICS SAVE --------------------
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
<<<<<<< HEAD
    # Use the script's directory to resolve path robustly
=======
>>>>>>> 30k_dataset
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
<<<<<<< HEAD
=======

import requests
import json

def submit(results, groupname, url="http://65.108.245.177:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
>>>>>>> 30k_dataset

# -------------------- MAIN SCRIPT --------------------
def main(k=K):
    start_time = time.time()
    print(f"[INFO] Starting DINOv2 Retrieval (top-{k})...")

    # Build TRAIN_LOOKUP
    TRAIN_LOOKUP = {}
    train_dir = os.path.join(BASE_DIR, "data", "training")
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.png')):
                TRAIN_LOOKUP[img_name] = class_name

    # Load datasets
    query_dataset = ImagePathDataset(QUERY_DIR, transform)
    gallery_dataset = ImagePathDataset(GALLERY_DIR, transform)

    # Extract features
    query_features, query_filenames = extract_features(query_dataset)
    gallery_features, gallery_filenames = extract_features(gallery_dataset)

    # Compute top-k similarity
    results = {}
    correct = 0
    total = 0
    for i, q_feat in enumerate(query_features):
        sims = cosine_similarity(q_feat.reshape(1, -1), gallery_features)[0]
        topk = np.argsort(sims)[::-1][:k]
        retrieved = [gallery_filenames[j] for j in topk]
        results[query_filenames[i]] = retrieved

        q_class = extract_class(query_filenames[i], TRAIN_LOOKUP)
        if q_class == "unknown":
            continue
        retrieved_classes = [extract_class(fn, TRAIN_LOOKUP) for fn in retrieved]
        if q_class in retrieved_classes:
            correct += 1
        total += 1

<<<<<<< HEAD
    top_k_acc = correct / len(query_filenames)
    print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
=======
    top_k_acc = correct / total if total > 0 else 0.0
    print(f"Top-{k} Accuracy (valid queries only): {top_k_acc:.4f}")
>>>>>>> 30k_dataset

    # Save submission
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {OUTPUT_PATH}")

    # Save metrics
    runtime = time.time() - start_time
    save_metrics_json(
        model_name="dinov2-base",
        top_k_accuracy=top_k_acc,
        batch_size=BATCH_SIZE,
        is_finetuned=False,
        runtime=runtime,
        loss_function="None",
        num_epochs=None,
        final_loss=None
    )

    submit(results, groupname="Stochastic thr")

if __name__ == "__main__":
    main()