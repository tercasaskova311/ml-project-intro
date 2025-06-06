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
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import requests
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import top_k_accuracy, precision_at_k  # Import our custom metric functions

# =======================
#    CONFIGURATION
# =======================
k = 9  # Number of top-k retrieved images to evaluate per query
batch_size = 32  # Reduced batch size for stability
FINE_TUNE = True  # If True, fine-tune the CLIP model on the training dataset
TRAIN_LAST_LAYER_ONLY = False  # If True, only fine-tune the projection head
epochs = 20  # Number of training epochs
lr = 5e-5  # Reduced learning rate for stability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
temperature = 0.1  # Temperature parameter for contrastive loss
max_grad_norm = 1.0  # Maximum gradient norm for clipping

# =======================
#  MODEL INITIALIZATION
# =======================
# Load the CLIP model and its preprocessing pipeline
model, preprocess = clip.load("ViT-B/16", device=device)

def configure_model_parameters(classifier):
    """Configure which parameters should be trained based on settings."""
    if not FINE_TUNE:
        # Fully freeze the model — useful for zero-shot or frozen feature extraction
        for param in classifier.parameters():
            param.requires_grad = False
    elif TRAIN_LAST_LAYER_ONLY:
        # Freeze CLIP model
        for param in classifier.clip.parameters():
            param.requires_grad = False
        # Train projection head and classifier
        for param in classifier.projection.parameters():
            param.requires_grad = True
        for param in classifier.fc.parameters():
            param.requires_grad = True
    else:
        # Enable full fine-tuning of all model weights
        for param in classifier.parameters():
            param.requires_grad = True

    # Print trainable parameters for verification
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def get_optimizer(model):
    """Get optimizer with appropriate parameter groups and learning rates."""
    if TRAIN_LAST_LAYER_ONLY:
        return torch.optim.Adam([
            {'params': model.projection.parameters(), 'lr': lr},
            {'params': model.fc.parameters(), 'lr': lr}
        ])
    else:
        return torch.optim.Adam([
            {'params': model.clip.parameters(), 'lr': lr * 0.1},  # Lower learning rate for CLIP
            {'params': model.projection.parameters(), 'lr': lr},
            {'params': model.fc.parameters(), 'lr': lr}
        ])

# =======================
#    EARLY STOPPING
# =======================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def __call__(self, model, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(model)
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = current_loss
            self.save_checkpoint(model)
            self.counter = 0
        return self.should_stop

    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        self.best_state = model.state_dict()

    def load_best_model(self, model):
        """Load the best model."""
        model.load_state_dict(self.best_state)

# =======================
#  CUSTOM CLASSIFIER
# =======================
class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        # Add a projection head with batch normalization for fine-tuning
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(clip_model.visual.output_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256)
        )
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        expected_dtype = self.clip.visual.conv1.weight.dtype
        x = x.to(dtype=expected_dtype)
        
        # Get CLIP features
        with autocast():
            features = self.clip.encode_image(x).float()
        
        # Project features
        projected = self.projection(features)
        projected = F.normalize(projected, dim=-1)  # L2 normalize
        
        # Classification output
        return self.fc(projected)

    def extract_features(self, x):
        expected_dtype = self.clip.visual.conv1.weight.dtype
        x = x.to(dtype=expected_dtype)
        
        # Get CLIP features and project them
        with torch.set_grad_enabled(self.training):
            with autocast():
                features = self.clip.encode_image(x).float()
            projected = self.projection(features)
            projected = F.normalize(projected, dim=-1)  # L2 normalize
        
        return projected

def contrastive_loss(features, labels, temp=temperature):
    """Compute contrastive loss with temperature scaling"""
    batch_size = features.size(0)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    
    # Compute similarity matrix
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    
    # Mask out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(features.device),
        0
    )
    
    mask = mask * logits_mask
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temp
    
    # Compute log_prob with numerical stability
    exp_sim = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-7)
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
    
    return -mean_log_prob_pos.mean()

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
def fine_tune_clip(train_loader, model, epochs=epochs, lr=lr):
    model.train()
    optimizer = torch.optim.AdamW([
        {'params': model.clip.parameters(), 'lr': lr * 0.1},  # Lower learning rate for CLIP
        {'params': model.projection.parameters(), 'lr': lr},
        {'params': model.fc.parameters(), 'lr': lr}
    ], weight_decay=0.01)  # Add weight decay for regularization
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    ce_loss = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()  # For mixed precision training
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=3,
        min_delta=1e-4,
        path=f'checkpoint_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt'
    )

    final_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                # Get features and classification outputs
                features = model.extract_features(images)
                outputs = model.fc(features)
                
                # Compute both losses
                loss_ce = ce_loss(outputs, labels)
                loss_cont = contrastive_loss(features, labels)
                
                # Combined loss with weighting
                loss = loss_ce + 0.5 * loss_cont

            # Skip batch if loss is NaN
            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        final_loss = epoch_loss
        print(f"Epoch {epoch+1}/{epochs} - Total Loss: {epoch_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Early stopping check
        if early_stopping(model, epoch_loss):
            print(f"Early stopping triggered after epoch {epoch+1}")
            # Load the best model
            early_stopping.load_best_model(model)
            break

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
if __name__ == "__main__":
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
    configure_model_parameters(classifier)  # Configure which parameters to train
    
    # Fine-tune if enabled
    final_loss = None
    if FINE_TUNE:
        print(f"Fine-tuning model with {'all layers' if not TRAIN_LAST_LAYER_ONLY else 'last layer only'}")
        final_loss = fine_tune_clip(train_loader, classifier)

    # Feature extraction (with no gradient)
    print("Extracting features from gallery images...")
    gallery_features, gallery_paths = encode_images(gallery_dir, classifier, preprocess)
    print("Extracting features from query images...")
    query_features, query_paths = encode_images(query_dir, classifier, preprocess)

    # Run retrieval
    print("Performing image retrieval...")
    retrieval_results = retrieve(query_features, gallery_features, query_paths, gallery_paths, k)

    # Save submission file
    submission = {os.path.basename(q): retrieval_results[os.path.basename(q)] for q in query_paths}
    sub_dir = os.path.join(BASE_DIR, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sub_path = os.path.join(sub_dir, f"sub_clip_test.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to: {os.path.abspath(sub_path)}")

    # Compute and save metrics
    top_k_acc = top_k_accuracy(query_paths, retrieval_results, k=k)
    prec_at_k = precision_at_k(query_paths, retrieval_results, k=k)
    runtime = time.time() - start_time

    # Save metrics
    save_metrics_json(
        model_name="Clip-ViT-B-16",
        top_k_accuracy=top_k_acc,
        precision=prec_at_k,
        batch_size=batch_size,
        is_finetuned=FINE_TUNE,
        num_classes=len(train_dataset.class_to_idx),
        runtime=runtime,
        loss_function="CrossEntropyLoss + ContrastiveLoss",
        num_epochs=epochs,
        final_loss=final_loss
    )

    print(f"\nResults:")
    print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
    print(f"Precision@{k}: {prec_at_k:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
