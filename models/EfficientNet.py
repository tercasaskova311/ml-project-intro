import os
import json
import time
import torch
import timm
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import top_k_accuracy, precision_at_k

# ----- Config -----
# Configuration parameters
K = 10  # top-k for retrieval evaluation
FINE_TUNE = True  # whether to fine-tune the model on training data
USE_GEM = True  # whether to use GeM pooling or default global average pooling (GAP)
batch_size = 32  # batch size
epochs = 10  # number of fine-tuning epochs
lr = 5e-5  # learning rate for optimizer
MODEL_VARIANT = 'b0'  # efficientnet variant: choose between 'b0' and 'b3'

MODEL_NAME_MAP = {
    'b0': 'efficientnet_b0',
    'b3': 'efficientnet_b3'
}
INPUT_SIZE_MAP = {
    'b0': (224, 224),
    'b3': (300, 300)
}
MODEL_NAME = MODEL_NAME_MAP[MODEL_VARIANT]
IMAGE_SIZE = INPUT_SIZE_MAP[MODEL_VARIANT]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Image preprocessing -----
# Standard preprocessing for EfficientNet using ImageNet statistics
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- GeM Pooling -----
# GeM (Generalized Mean Pooling) layer for improved retrieval performance
class GeM(torch.nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1. / self.p)

# ----- EfficientNet with GeM -----
# Custom EfficientNet model using GeM pooling, optionally with classification head
class EfficientNetWithGeM(torch.nn.Module):
    def __init__(self, model_name, num_classes=None, fine_tune=False):
        super().__init__()
        base = timm.create_model(model_name, pretrained=True, num_classes=0)
        base = base.to(DEVICE)
        self.features = base.forward_features
        self.gem_pool = GeM().to(DEVICE)
        self.flatten = torch.nn.Flatten().to(DEVICE)
        self.out_dim = base.num_features
        self.classifier = None
        if fine_tune and num_classes is not None:
            self.classifier = torch.nn.Linear(self.out_dim, num_classes).to(DEVICE)

    def forward(self, x):
        x = self.features(x)
        x = self.gem_pool(x)
        x = self.flatten(x)
        if self.classifier:
            x = self.classifier(x)
        return x

    def get_embedding(self, x):
        x = self.features(x)
        x = self.gem_pool(x)
        x = self.flatten(x)
        return x

# ----- Model Loaders -----
# Load standard or GeM-pooled EfficientNet model
def load_model(num_classes=None, fine_tune=False):
    model = timm.create_model(MODEL_NAME, pretrained=True)
    if fine_tune and num_classes is not None:
        in_features = model.get_classifier().in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

def load_model_GEM(num_classes=None, fine_tune=False):
    model = EfficientNetWithGeM(MODEL_NAME, num_classes=num_classes, fine_tune=fine_tune)
    return model

# ----- Dataset -----
# Dataset that returns image tensors and filenames (used for gallery/query)
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

# ----- Feature Extraction -----
@torch.no_grad()  # Disable gradient computation for inference (faster and memory efficient)
def extract_features(model, dataloader):
    model.eval()  # Set the model to evaluation mode to deactivate dropout, batchnorm updates, etc.
    features = []  # To store the extracted feature vectors
    filenames = []  # To store corresponding image filenames

    for imgs, fnames in tqdm(dataloader):  # Iterate through image batches
        imgs = imgs.to(DEVICE)  # Move the batch to GPU (if available)
        feats = model(imgs)  # Forward pass through the model

        # If the output has 4 dimensions (e.g., B x C x H x W), flatten it
        # This ensures each image is represented as a single vector
        if feats.dim() == 4:
            feats = feats.view(feats.size(0), -1)

        features.append(feats.cpu().numpy())  # Move features to CPU and convert to NumPy
        filenames.extend(fnames)  # Track filenames for identification

    # Concatenate all feature vectors into a matrix of shape [N_images x feature_dim]
    return np.vstack(features), filenames

# ----- Fine-tuning Function -----

# Trains the model using classification as a proxy task
# The objective is to make embeddings more discriminative for retrieval
def fine_tune_model(model, train_loader, num_epochs=5, lr=1e-4):
    model.train()  # Enable training mode

    # Optimizer: Adam adapts learning rate per parameter and handles sparse gradients well
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # Update only trainable layers (e.g., classifier)
        lr=lr
    )
    # Loss: CrossEntropy is the standard for multi-class classification.
    # Though our final task is retrieval, classification helps learn separable embeddings.
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = 0.0  # Will hold the loss of the final epoch

    for epoch in range(num_epochs):  # Repeat training for N epochs
        epoch_loss = 0.0

        for imgs, labels in tqdm(train_loader):  # Iterate through training batches
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(imgs)  # Get class predictions from the model
            loss = loss_fn(preds, labels)  # Compute how far predictions are from true labels

            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model weights

            epoch_loss += loss.item()  # Accumulate batch loss

        # Compute and report average loss for this epoch
        final_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {final_loss:.4f}")

    return final_loss  # Return final epoch loss for metric logging

# ----- Class Extraction -----
def extract_class(filename):
    return filename.split("_")[0]

# ----- Metrics Saver -----

# Saves metrics to a JSON file for analysis
def save_metrics_json(model_name, top_k_accuracy, precision, batch_size, is_finetuned,
                      num_classes=None, runtime=None, loss_function="CrossEntropyLoss",
                      num_epochs=None, final_loss=None, pooling_type=None):
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
        "precision_at_k": round(precision, 4),
        "batch_size": batch_size,
        "is_finetuned": is_finetuned,
        "num_classes": num_classes,
        "runtime_seconds": round(runtime, 2) if runtime else None,
        "loss_function": loss_function,
        "num_epochs": num_epochs,
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        "pooling_type": pooling_type
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {os.path.abspath(out_path)}")

# ----- Submission Function -----
def submit(results, groupname, url="http://65.108.245.177:3001/retrieval/"):
    res = {"groupname": groupname, "images": results}
    response = requests.post(url, json=res)
    try:
        result = response.json()
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# ----- Main Execution -----
if __name__ == "__main__":
    start_time = time.time()
    
    train_root = os.path.join(DATA_DIR, 'training')
    num_classes = len(os.listdir(train_root))

    # Load EfficientNet with or without GeM pooling
    model = load_model_GEM(num_classes=num_classes, fine_tune=FINE_TUNE) if USE_GEM else load_model(num_classes=num_classes, fine_tune=FINE_TUNE)

    if FINE_TUNE:
        train_dataset = ImageFolder(root=train_root, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        final_loss = fine_tune_model(model, train_loader, num_epochs=epochs, lr=lr)
    else:
        final_loss = None

    # Prepare gallery and query datasets
    gallery_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/gallery"), transform)
    query_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/query"), transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    #Extract features for retrieval
    gallery_feats, gallery_names = extract_features(model, gallery_loader)
    query_feats, query_names = extract_features(model, query_loader)
    
    # Compute cosine similarities and perform top-k retrieval
    result = {}
    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    for i, qname in enumerate(query_names):
        topk_idx = np.argsort(sim_matrix[i])[::-1][:K]
        samples = [gallery_names[idx] for idx in topk_idx]
        result[qname] = samples

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submissions"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sub_{MODEL_NAME}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Submission saved to: {output_path}")
    
    # Evaluate accuracy and precision
    topk_acc = top_k_accuracy(query_names, result, k=K)
    prec_at_k = precision_at_k(query_names, result, k=K)
    total_time = time.time() - start_time

    save_metrics_json(
        model_name=MODEL_NAME,
        top_k_accuracy=topk_acc,
        precision=prec_at_k,
        batch_size=batch_size,
        is_finetuned=FINE_TUNE,
        num_classes=num_classes,
        runtime=total_time,
        loss_function="CrossEntropyLoss",
        num_epochs=epochs if FINE_TUNE else 0,
        final_loss=final_loss,
        pooling_type="GeM" if USE_GEM else "GAP"
    )

    submit(result, groupname="Stochastic_thr")
