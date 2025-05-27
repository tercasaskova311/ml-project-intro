import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from datetime import datetime
from timm import create_model

# ----- Config -----
K = 10
FINE_TUNE = True
batch_size = 32
epochs = 5
lr = 1e-5

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_vit_model(num_classes=None, fine_tune=False):
    model = create_model("vit_base_patch16_224", pretrained=True)
    if fine_tune and num_classes is not None:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    else:
        model.head = torch.nn.Identity()
    return model.to(DEVICE)

def finetune_model(model, dataloader, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    final_loss = None
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {final_loss:.4f}")
    model.eval()
    return final_loss

class ImagePathDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.img_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
                          if fname.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

@torch.no_grad()
def extract_features(model, dataloader):
    features = []
    filenames = []
    for imgs, fnames in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        feats = model.forward_features(imgs)
        if feats.ndim > 2:
            feats = feats.mean(dim=tuple(range(2, feats.ndim)))
        feats = feats.cpu().numpy()
        features.append(feats)
        filenames.extend(fnames)
    return np.vstack(features), filenames


def extract_class(filename, *_):
    return filename.split('_')[0]

def calculate_top_k_accuracy(query_paths, retrievals, *_ , k=10):
    correct = 0
    total = 0
    for qname in query_paths:
        q_class = extract_class(qname)
        retrieved_classes = [extract_class(name) for name in retrievals[qname]]
        if q_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"Top-{k} Accuracy (valid queries only): {acc:.4f}")
    return acc

def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned, num_classes,
                      runtime, loss_function, num_epochs, final_loss, pooling_type=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")
    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": K,
        "top_k_accuracy": round(top_k_accuracy, 4),
        "batch_size": batch_size,
        "is_finetuned": is_finetuned,
        "num_classes": num_classes,
        "runtime_seconds": round(runtime, 2),
        "loss_function": loss_function,
        "num_epochs": num_epochs,
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        "pooling_type": pooling_type
    }
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {os.path.abspath(out_path)}")

def main():
    start_time = time.time()

    if FINE_TUNE:
        train_dataset = ImageFolder(os.path.join(DATA_DIR, "training"), transform)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        model = load_vit_model(num_classes=len(train_dataset.classes), fine_tune=True)
        final_loss = finetune_model(model, train_loader, epochs, lr)
    else:
        model = load_vit_model()
        final_loss = None

    gallery_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/gallery"), transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size)
    gallery_feats, gallery_names = extract_features(model, gallery_loader)

    query_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/query"), transform)
    query_loader = DataLoader(query_dataset, batch_size)
    query_feats, query_names = extract_features(model, query_loader)

    result = {}
    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    for i, qname in enumerate(query_names):
        topk_idx = np.argsort(sim_matrix[i])[::-1][:K]
        samples = [gallery_names[idx] for idx in topk_idx]
        result[qname] = samples

    submission = {qname: result[qname] for qname in query_names}

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submissions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sub_vit.json")
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to: {output_path}")

    topk_acc = calculate_top_k_accuracy(query_names, result)
    total_time = time.time() - start_time
    num_classes = len(train_dataset.classes) if FINE_TUNE else None
    save_metrics_json("vit_base_patch16_224", topk_acc, batch_size, FINE_TUNE, num_classes, total_time,
                      "CrossEntropyLoss" if FINE_TUNE else "NotApplicable", epochs if FINE_TUNE else 0, final_loss)

if __name__ == "__main__":
    main()
