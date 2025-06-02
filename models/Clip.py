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
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ---------------- CONFIGURATION ----------------
# General settings for the image retrieval task
k = 10  # Number of top similar images to retrieve
batch_size = 2  # Batch size for both training and feature extraction
FINE_TUNE = True  # Whether to fine-tune the CLIP model
TRAIN_LAST_LAYER_ONLY = True  # If True, only train the final linear layer
epochs = 5  # Number of training epochs
lr = 1e-4  # Learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# ---------------- MODEL & TRANSFORM ----------------
# Load the pre-trained CLIP model and image preprocessing pipeline
model, preprocess = clip.load("ViT-L/14", device=device)

# Freeze all or part of the model depending on the fine-tuning strategy
if not FINE_TUNE:
    for param in model.parameters():
        param.requires_grad = False
elif TRAIN_LAST_LAYER_ONLY:
    for name, param in model.named_parameters():
        # Enable gradients only for the projection layer
        if "proj" in name or "visual.proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    for param in model.parameters():
        param.requires_grad = True

# Custom module to append a classifier head on top of CLIP visual encoder
class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        self.fc = torch.nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        # Extract image features without updating the CLIP encoder
        with torch.no_grad():
            x = self.clip.encode_image(x).float()
        x = x / x.norm(dim=-1, keepdim=True)  # Normalize embeddings
        return self.fc(x)

# ---------------- DATASET ----------------
# Dataset that loads image paths and class labels from a folder structure
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

# ---------------- TRAINING ----------------
# Function to fine-tune a classifier head on top of the CLIP encoder.
# Although the final task is image retrieval, we use class supervision, via labels
# to learn a better feature representation aligned with semantic similarity.
# CrossEntropyLoss suits here because we're training a classification head.
def fine_tune_clip(train_loader, model, epochs=epochs, lr=lr):
    model.train()  # Set the model to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # We use CrossEntropyLoss to train the classifier head.
    # This helps the model learn to separate visual categories using class labels,
    # which improves the quality of the embeddings used for retrieval later.
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through the classifier
            outputs = model(images)

            # Compute classification loss
            loss = loss_fn(outputs, labels)

            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue

            # Weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        final_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {final_loss:.4f}")
    return final_loss

# ---------------- ENCODING ----------------
# This function computes visual embeddings for the folder of images
# These embeddings are then used for similarity-based retrieval
def encode_images(image_folder, model, preprocess, batch_size=2):
    # Collect all valid image paths
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    ])
    
    features = []       # List to store feature vectors
    valid_paths = []    # List to track successfully processed image paths

    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Iterate through images in mini-batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        current_valid_paths = []

        # Load and preprocess each image
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                batch_images.append(img)
                current_valid_paths.append(p)
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {p}")

        if not batch_images:
            continue

        # Stack the preprocessed images into a batch tensor
        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            # Forward pass through CLIP's image encoder
            batch_features = model.encode_image(batch_tensor).float()

            # Normalize embeddings to unit vectors for cosine similarity
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

            # Store embeddings and their corresponding filenames
            features.append(batch_features.cpu())
            valid_paths.extend(current_valid_paths)

        # Clear memory cache after each batch to avoid GPU OOM
        torch.cuda.empty_cache()

    # Return concatenated feature matrix and corresponding image paths
    return torch.cat(features, dim=0).to(device), valid_paths

# ---------------- RETRIEVAL ----------------
# Compute cosine similarity and retrieve top-k most similar images
def retrieve(query_features, gallery_features, query_paths, gallery_paths, k):
    similarities = query_features @ gallery_features.T
    topk_values, topk_indices = similarities.topk(k, dim=-1)
    results = {}
    for i in range(query_features.shape[0]):
        query_filename = os.path.basename(query_paths[i])
        retrieved_filenames = [os.path.basename(gallery_paths[idx]) for idx in topk_indices[i].cpu().numpy()]
        results[query_filename] = retrieved_filenames
    return results

# Extract class identifier from filename
def extract_class(filename, *_):
    return filename.split('_')[0]

# Evaluate top-k retrieval accuracy based on shared class in the filename
def calculate_top_k_accuracy(query_paths, retrievals, *_ , k=10):
    correct = 0
    total = 0
    for qname in query_paths:
        qfile = os.path.basename(qname)
        q_class = extract_class(qfile)
        retrieved_classes = [extract_class(name) for name in retrievals[qfile]]
        if q_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"Top-{k} Accuracy (valid queries only): {acc:.4f}")
    return acc

# ---------------- LOGGING ----------------
# Save performance metrics to a JSON file, to keep track of the metrics of each run
def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")
    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": k,
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

# Submit the results to the evaluation server
def submit(results, groupname="stochastic thr", url="http://65.108.245.177:3001/retrieval/"):
    res = {
        "groupname": groupname,
        "images": results
    }
    res = json.dumps(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# ---------------- MAIN EXECUTION ----------------
# Set paths
start_time = time.time()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
training_dir = os.path.join(DATA_DIR, "training")
query_dir = os.path.join(DATA_DIR, "test", "query")
gallery_dir = os.path.join(DATA_DIR, "test", "gallery")

# Load dataset and prepare training loader
train_dataset = ImageFolderDataset(training_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Initialize and train classifier
classifier = ClipClassifier(model, num_classes=len(train_dataset.class_to_idx)).to(device)
final_loss = fine_tune_clip(train_loader, classifier) if FINE_TUNE else None

# Extract features for gallery and query sets
with torch.no_grad():
    gallery_features, gallery_paths = encode_images(gallery_dir, model, preprocess)
    query_features, query_paths = encode_images(query_dir, model, preprocess)

# Retrieve top-k most similar gallery images for each query
retrieval_results = retrieve(query_features, gallery_features, query_paths, gallery_paths, k)

# Create lookup table for label-based accuracy evaluation
TRAIN_LOOKUP = {}
for class_name in os.listdir(training_dir):
    class_dir = os.path.join(training_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.png')):
            TRAIN_LOOKUP[img_name] = class_name

# Write submission JSON
submission = {os.path.basename(q): retrieval_results[os.path.basename(q)] for q in query_paths}
sub_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "sub_clip.json")
with open(sub_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Submission saved to: {os.path.abspath(sub_path)}")

# Submit to server and compute accuracy
submit(submission, groupname="Stochastic thr")
top_k_acc = calculate_top_k_accuracy(query_paths, retrieval_results, TRAIN_LOOKUP, k=k)
runtime = time.time() - start_time

# Save evaluation metrics to compare runs
save_metrics_json(
    model_name="clip-vit-b32",
    top_k_accuracy=top_k_acc,
    batch_size=batch_size,
    is_finetuned=FINE_TUNE,
    num_classes=len(train_dataset.class_to_idx),
    runtime=runtime,
    loss_function="CrossEntropyLoss",
    num_epochs=epochs,
    final_loss=final_loss
)