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
K = 5  # Number of top similar images to retrieve
BATCH_SIZE = 16  # Batch size for feature extraction
MODEL_NAME = "facebook/dinov2-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamically get paths
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
def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features, all_filenames = [], []

    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Extracting features"):
            images = images.to(DEVICE)
            feats = model(images).last_hidden_state.mean(dim=1)
            all_features.append(feats.cpu().numpy())
            all_filenames.extend(filenames)

    return np.vstack(all_features), all_filenames

# -------------------- METRICS SAVE --------------------
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
    # Use the script's directory to resolve path robustly
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

# -------------------- MAIN SCRIPT --------------------
def main(k=K):
    start_time = time.time()
    print(f"[INFO] Starting DINOv2 Retrieval (top-{k})...")

    # Load datasets
    query_dataset = ImagePathDataset(QUERY_DIR, transform)
    gallery_dataset = ImagePathDataset(GALLERY_DIR, transform)

    # Extract features
    query_features, query_filenames = extract_features(query_dataset)
    gallery_features, gallery_filenames = extract_features(gallery_dataset)

    # Compute top-k similarity
    results =  {}
    correct = 0
    for i, q_feat in enumerate(query_features):
        sims = cosine_similarity(q_feat.reshape(1, -1), gallery_features)[0]
        topk = np.argsort(sims)[::-1][:k]
        retrieved = [gallery_filenames[j] for j in topk]
        results[query_filenames[i]] = retrieved
        # Accuracy calculation
        q_class = query_filenames[i].split("_")[0]
        retrieved_classes = [name.split("_")[0] for name in retrieved]
        if q_class in retrieved_classes:
            correct += 1

    top_k_acc = correct / len(query_filenames)
    print(f"Top-{k} Accuracy: {top_k_acc:.4f}")

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
    loss_function="None",  # No loss function used
    num_epochs=None,
    final_loss=None
)


if __name__ == "__main__":
    main()
