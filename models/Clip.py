import os
import json
import time
import torch
import clip
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import top_k_accuracy, precision_at_k  # Import our custom metric functions

# =======================
#    CONFIGURATION
# =======================
k = 9  # Number of top-k retrieved images to evaluate per query
batch_size = 64  # Batch size
FINE_TUNE = True  # If True, fine-tune the CLIP model on the training dataset
TRAIN_LAST_LAYER_ONLY = False  # If True, only fine-tune the projection head
epochs = 10  # Number of training epochs
lr = 1e-4  # Learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available for performance

# =======================
#  MODEL INITIALIZATION
# =======================
# Load the CLIP model and its preprocessing pipeline
model, preprocess = clip.load("ViT-B/16", device=device)

# Configure the fine-tuning strategy by setting trainable parameters
if not FINE_TUNE:
    # Fully freeze the model — useful for zero-shot or frozen feature extraction
    for param in model.parameters():
        param.requires_grad = False
elif TRAIN_LAST_LAYER_ONLY:
    # Train only the projection layer (last part of the visual encoder)
    for name, param in model.named_parameters():
        if "proj" in name or "visual.proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    # Enable full fine-tuning of all model weights
    for param in model.parameters():
        param.requires_grad = True

# =======================
#  CUSTOM CLASSIFIER
# =======================
# Wrap CLIP's visual encoder with a classification head for supervised training
class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        self.encoder = torch.nn.Sequential(
            clip_model.visual, 
            torch.nn.LayerNorm(clip_model.visual.output_dim)  
        )
        self.fc = torch.nn.Linear(clip_model.visual.output_dim, num_classes) 

    def forward(self, x):
    # Ensure input has the same dtype as the model (important for CLIP with float16 weights)
        expected_dtype = self.clip.visual.conv1.weight.dtype
        x = x.to(dtype=expected_dtype)

        # Encode image using CLIP's visual encoder
        with torch.no_grad():
            x = self.clip.encode_image(x).float()  # Note: we .float() here only if needed post-encoding

        x = x / x.norm(dim=-1, keepdim=True)
        return self.fc(x)

    def extract_features(self, x):
        expected_dtype = self.clip.visual.conv1.weight.dtype
        x = x.to(dtype=expected_dtype)
        with torch.no_grad():
            x = self.clip.encode_image(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x




# =======================
#     DATA LOADER
# =======================
# Custom dataset for loading images and their labels from a folder structure
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
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        self.img_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, os.path.basename(self.img_paths[idx])

# =======================
#    TRAINING LOGIC
# =======================
# Standard supervised training loop using CrossEntropyLoss
def fine_tune_clip(train_loader, model, epochs=epochs, lr=lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        final_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {final_loss:.4f}")
    return final_loss

# =======================
#   ENCODING IMAGES
# =======================
# Computes CLIP embeddings for a folder of images using batch inference
@torch.no_grad()
def encode_images(image_folder, model, preprocess, batch_size=64):
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    ])
    features = []
    valid_paths = []

    model.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        current_valid_paths = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                batch_images.append(img)
                current_valid_paths.append(p)
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {p}")

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        # Usa direttamente extract_features del classifier
        batch_features = model.extract_features(batch_tensor)
        features.append(batch_features.cpu())
        valid_paths.extend(current_valid_paths)

        torch.cuda.empty_cache()

    return torch.cat(features, dim=0).to(device), valid_paths


# =======================
#     RETRIEVAL STEP
# =======================
# Uses cosine similarity to compute top-k most similar images per query
def retrieve(query_features, gallery_features, query_paths, gallery_paths, k):
    similarities = query_features @ gallery_features.T
    topk_values, topk_indices = similarities.topk(k, dim=-1)
    results = {}
    for i in range(query_features.shape[0]):
        query_filename = os.path.basename(query_paths[i])
        retrieved_filenames = [os.path.basename(gallery_paths[idx]) for idx in topk_indices[i].cpu().numpy()]
        results[query_filename] = retrieved_filenames
    return results

def extract_class(filename, *_):
    return filename.split('_')[0]  # Assumes class name is prefix before '_'

# =======================
#        METRICS
# =======================
def save_metrics_json(model_name, top_k_accuracy, precision, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results_animals")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")

    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": k,
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
    print(f"[LOG] Metrics saved to: {os.path.abspath(out_path)}")


# Function to submit results to the competition server
def submit(results, groupname="stochastic thr", url="http://65.108.245.177:3001/retrieval/"):
    payload = {
        "groupname": groupname,
        "images": results
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"[SUBMIT] Server response - Accuracy: {result.get('accuracy')}")
    except requests.RequestException as e:
        print(f"[SUBMIT] Submission failed: {e}")

# =======================
#      MAIN SCRIPT
# =======================
start_time = time.time()

# Define root paths for training and test images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_animals")
training_dir = os.path.join(DATA_DIR, "training")
query_dir = os.path.join(DATA_DIR, "test", "query")
gallery_dir = os.path.join(DATA_DIR, "test", "gallery")

# Load training data
train_dataset = ImageFolderDataset(training_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Initialize classifier model
classifier = ClipClassifier(model, num_classes=len(train_dataset.class_to_idx)).to(device)
final_loss = fine_tune_clip(train_loader, classifier) if FINE_TUNE else None

# Feature extraction (with no gradient)
with torch.no_grad():
    # Usa il modello aggiornato (con il projection layer fine-tunato)
    gallery_features, gallery_paths = encode_images(gallery_dir, classifier, preprocess)
    query_features, query_paths = encode_images(query_dir, classifier, preprocess)

    # gallery_features, gallery_paths = encode_images(gallery_dir, model, preprocess)
    # query_features, query_paths = encode_images(query_dir, model, preprocess)

# Run retrieval
retrieval_results = retrieve(query_features, gallery_features, query_paths, gallery_paths, k)

# Format the submission file 
submission = {os.path.basename(q): retrieval_results[os.path.basename(q)] for q in query_paths}
sub_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "sub_clip.json")
with open(sub_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Submission saved to: {os.path.abspath(sub_path)}")

# Compute metrics, taken from utils.metrics
top_k_acc = top_k_accuracy(query_paths, retrieval_results, k=k)
prec_at_k = precision_at_k(query_paths, retrieval_results, k=k)
runtime = time.time() - start_time

# Save metrics as JSON
save_metrics_json(
    model_name="Clip-ViT-B-32",
    top_k_accuracy=top_k_acc,
    precision=prec_at_k,
    batch_size=batch_size,
    is_finetuned=FINE_TUNE,
    num_classes=len(train_dataset.class_to_idx),
    runtime=runtime,
    loss_function="CrossEntropyLoss",
    num_epochs=epochs,
    final_loss=final_loss
)
