import os
import json
import time
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIGURATION ----------------
k = 10
batch_size = 64
FINE_TUNE = False
TRAIN_LAST_LAYER_ONLY = True
epochs = 5
lr = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- MODEL & TRANSFORM ----------------
model, preprocess = clip.load("ViT-B/32", device=device)

if not FINE_TUNE:
    for param in model.parameters():
        param.requires_grad = False
elif TRAIN_LAST_LAYER_ONLY:
    for name, param in model.named_parameters():
        if "proj" in name or "visual.proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    for param in model.parameters():
        param.requires_grad = True

# ---------------- DATASET ----------------
class ImageTextDataset(Dataset):
    def __init__(self, root_folder, transform):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        for label_name in os.listdir(root_folder):
            label_path = os.path.join(root_folder, label_name)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    self.img_paths.append(os.path.join(label_path, img_file))
                    self.labels.append(label_name)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, os.path.basename(self.img_paths[idx])

# ---------------- FUNCTIONS ----------------
def fine_tune_clip(train_loader, model, epochs=epochs, lr=lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = None
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            texts = clip.tokenize(labels).to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = image_features @ text_features.t()
            logits_per_text = text_features @ image_features.t()
            ground_truth = torch.arange(len(images), device=device)
            loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {final_loss:.4f}")
    return final_loss

def encode_images(image_folder, model, preprocess):
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
    images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    images = torch.stack(images).to(device)
    with torch.no_grad():
        features = model.encode_image(images)
        features /= features.norm(dim=-1, keepdim=True)
    return features, image_paths

def retrieve(query_features, gallery_features, query_paths, gallery_paths, k):
    similarities = query_features @ gallery_features.T
    topk_values, topk_indices = similarities.topk(k, dim=-1)
    results = {}
    for i in range(query_features.shape[0]):
        query_filename = os.path.basename(query_paths[i])
        retrieved_filenames = [os.path.basename(gallery_paths[idx]) for idx in topk_indices[i].cpu().numpy()]
        results[query_filename] = retrieved_filenames
    return results

def extract_class(filename, train_lookup):
    if "_" in filename:
        parts = filename.split("_")
        if len(parts) >= 2 and not parts[0].isdigit():
            return "_".join(parts[:-1])
    return train_lookup.get(os.path.basename(filename), "unknown")

def calculate_top_k_accuracy(query_paths, retrievals, train_lookup, k=10):
    correct = 0
    total = 0
    for qname in query_paths:
        qfile = os.path.basename(qname)
        q_class = extract_class(qfile, train_lookup)
        if q_class == "unknown":
            continue
        retrieved_classes = [extract_class(name, train_lookup) for name in retrievals[qfile]]
        if q_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"Top-{k} Accuracy (valid queries only): {acc:.4f}")
    return acc

def visualize_retrieval(query_paths, retrieval_results, k=5):
    for query_path in query_paths:
        qfile = os.path.basename(query_path)
        if qfile not in retrieval_results:
            continue
        plt.figure(figsize=(18, 4))
        plt.subplot(1, k + 1, 1)
        plt.title("Query")
        plt.axis("off")
        plt.imshow(Image.open(query_path))
        for j, img_name in enumerate(retrieval_results[qfile][:k]):
            gallery_img_path = os.path.join(GALLERY_DIR, img_name)
            if not os.path.exists(gallery_img_path):
                continue
            plt.subplot(1, k + 1, j + 2)
            plt.title(f"Rank {j + 1}")
            plt.axis("off")
            plt.imshow(Image.open(gallery_img_path))
        plt.tight_layout()
        plt.show()

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

# ---------------- MAIN EXECUTION ----------------
start_time = time.time()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
training_dir = os.path.join(DATA_DIR, "training")
query_dir = os.path.join(DATA_DIR, "test", "query")
gallery_dir = os.path.join(DATA_DIR, "test", "gallery")
GALLERY_DIR = gallery_dir

train_dataset = ImageTextDataset(training_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
if FINE_TUNE:
    final_loss = fine_tune_clip(train_loader, model)
else:
    final_loss = None

gallery_features, gallery_paths = encode_images(gallery_dir, model, preprocess)
query_features, query_paths = encode_images(query_dir, model, preprocess)
retrieval_results = retrieve(query_features, gallery_features, query_paths, gallery_paths, k)

TRAIN_LOOKUP = {}
for class_name in os.listdir(training_dir):
    class_dir = os.path.join(training_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.png')):
            TRAIN_LOOKUP[img_name] = class_name

submission = {}
for qname in query_paths:
    qfile = os.path.basename(qname)
    submission[qfile] = retrieval_results[qfile]

sub_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "sub_clip.json")
with open(sub_path, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Submission saved to: {os.path.abspath(sub_path)}")

top_k_acc = calculate_top_k_accuracy(query_paths, retrieval_results, TRAIN_LOOKUP, k=k)
runtime = time.time() - start_time
save_metrics_json(
    model_name="clip-vit-b32",
    top_k_accuracy=top_k_acc,
    batch_size=batch_size,
    is_finetuned=FINE_TUNE,
    num_classes=None,
    runtime=runtime,
    loss_function="CrossEntropyLoss",
    num_epochs=epochs,
    final_loss=final_loss
)
visualize_retrieval(query_paths, retrieval_results, k)
