import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import json
import torch.optim as optim
from datetime import datetime
import requests
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import top_k_accuracy, precision_at_k

# ---------------- CONFIGURATION ----------------
k = 10  # Top-K retrieval
batch_size = 32
image_size = (224, 224)  # GoogLeNet standard input size
normalize_mean = [0.485, 0.456, 0.406]  # Imagenet normalization mean
normalize_std = [0.229, 0.224, 0.225]   # Imagenet normalization std
FINE_TUNE = True
TRAIN_LAST_LAYER_ONLY = False  # Fine-tune only last layer for faster convergence
epochs = 10
learning_rate = 5e-5
USE_GEM_POOLING = False  # Use GeM pooling instead of Global Average Pooling (GAP)
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# ======= GeM Pooling Layer =========
# Generalized Mean Pooling (GeM) is a parameterized pooling layer that
# generalizes average and max pooling.
class GeM(torch.nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)  # Learnable pooling parameter
        self.eps = eps  # Prevent zero division

    def forward(self, x):
        # Clamp inputs > eps, raise to power p, average pool, then take p-th root
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1. / self.p)


# ======= PATHS ============
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir_test = os.path.join(project_root, 'data_animals', 'test')
query_folder = os.path.join(root_dir_test, 'query')
gallery_folder = os.path.join(root_dir_test, 'gallery')

#====== TRAIN LOOP - TRAIN FOLDER ======
TRAIN_LOOKUP = {}

train_root = os.path.join(project_root, 'data_animals', 'training') 

for class_name in os.listdir(train_root):
    class_dir = os.path.join(train_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.png')):
            TRAIN_LOOKUP[img_name] = class_name

# ======= Dataset Loading ========
# Dataset that reads images from folder paths and applies transform
class ImagePathDataset(Dataset):
    def __init__(self, folder, transform=None):
        # List all image files in the folder
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Open image and convert to RGB
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

#======= TRANSFORMATION - IMAGES ===========
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

# -------- Build GoogLeNet Model with Optional GeM Pooling --------
def build_googlenet_extractor(device):
    # Load pretrained GoogLeNet with default weights and disable auxiliary classifiers
    weights = torchvision.models.GoogLeNet_Weights.DEFAULT
    model = torchvision.models.googlenet(weights=weights, aux_logits=True)
    model.aux1 = torch.nn.Identity()
    model.aux2 = torch.nn.Identity()
    model._original_fc_in_features = model.fc.in_features

    # Replace pooling layer
    model.avgpool = GeM() if USE_GEM_POOLING else torch.nn.AdaptiveAvgPool2d((1, 1))

    # Remove final FC layer for embedding extraction
    model.fc = torch.nn.Identity()

    return model.to(device).eval()


# -------- Fine-tune Model on Training Data --------
def fine_tune_model(model, train_loader, device, num_classes,
                    train_last_layer_only=True, epochs=epochs, learning_rate=learning_rate):
    print(f"[FT] Starting fine-tuning for {epochs} epochs...")
    model.train().to(device)

    # Replace FC with new classifier for current dataset classes
    model.fc = torch.nn.Linear(model._original_fc_in_features, num_classes).to(device)

    # Freeze all parameters except the final FC layer if specified
    if train_last_layer_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Handle output shape if logits are inside a dict (in some models)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"[FT] Epoch {epoch+1}: Loss={running_loss / len(train_loader):.3f}, Acc={acc:.2f}%")

    print("[FT] Fine-tuning complete.")
    return model.eval(), running_loss / len(train_loader)

#======== extraction of embedings =========
def extract_embeddings(loader, model, device):
    embeddings = []
    for images, _ in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model(images)
        embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)

extract_embeddings = torch.no_grad()(extract_embeddings)

# -------- Retrieve Top-K Similar Images --------
def retrieve_topk(query_embs, gallery_embs, query_paths, gallery_paths, k):
    # Normalize embeddings to unit vectors (for cosine similarity)
    query_embs = F.normalize(query_embs, dim=1)
    gallery_embs = F.normalize(gallery_embs, dim=1)

    # Compute similarity matrix: cosine similarity = dot product of normalized vectors
    sim_matrix = query_embs @ gallery_embs.T
    # Get indices of top-k most similar gallery images for each query
    topk_indices = sim_matrix.topk(k, dim=1, largest=True)[1]

    results = {}
    for i, indices in enumerate(topk_indices):
        query_filename = os.path.basename(query_paths[i])
        top_filenames = [os.path.basename(gallery_paths[j]) for j in indices.tolist()]
        results[query_filename] = top_filenames
    return results

# ======= Metrics saving =======
def save_metrics_json(model_name, top_k_accuracy, precision, batch_size, is_finetuned, num_classes=None, runtime=None, loss_function="MSELoss", num_epochs=None, final_loss=None, pooling_type=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results_animals")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(results_dir, f"{model_name}_metrics_{timestamp}.json")
    
    metrics = {
        "model_name": model_name,
        "run_id": timestamp,
        "top_k": 10,
        "top_k_accuracy": round(top_k_accuracy, 4),
        "precision_at_k": round(precision, 4),
        "batch_size": batch_size,
        "is_finetuned": is_finetuned,
        "num_classes": num_classes,
        "runtime_seconds": round(runtime, 2) if runtime else None,
        "loss_function": loss_function,
        "num_epochs": num_epochs,
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        "pooling_type": "GeM" if USE_GEM_POOLING else "GAP"
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {os.path.abspath(out_path)}")



#====== SUBMIT =========
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


#======= Main ========
if __name__ == '__main__':
    import time

    # Create training lookup dict for accuracy evaluation
    TRAIN_LOOKUP = {}
    train_root = os.path.join(project_root, 'data_animals', 'training')
    for class_name in os.listdir(train_root):
        class_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.png')):
                TRAIN_LOOKUP[img_name] = class_name

    # Load query and gallery datasets
    query_dataset = ImagePathDataset(query_folder, transform=transform)
    gallery_dataset = ImagePathDataset(gallery_folder, transform=transform)
    query_loader = DataLoader(query_dataset, batch_size, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size, shuffle=False, num_workers=4)

    start_time = time.time()

    # Build and fine-tune GoogLeNet
    model = build_googlenet_extractor(device)
    final_loss = None
    if FINE_TUNE:
        print("[FT] Loading training data...")
        from torchvision.datasets import ImageFolder
        train_dataset = ImageFolder(train_root, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        model, final_loss = fine_tune_model(
            model=model,
            train_loader=train_loader,
            device=device,
            num_classes=len(train_dataset.classes),
            train_last_layer_only=TRAIN_LAST_LAYER_ONLY,
            epochs=epochs,
            learning_rate=learning_rate
        )

    # Extract embeddings
    gallery_embeddings = extract_embeddings(gallery_loader, model, device)
    query_embeddings = extract_embeddings(query_loader, model, device)

    # Retrieve top-k results
    gallery_paths = gallery_dataset.paths
    query_paths = query_dataset.paths
    result = retrieve_topk(query_embeddings, gallery_embeddings, query_paths, gallery_paths, k=k)

    # Evaluate Top-k Accuracy and Precision@k
    query_filenames = [os.path.basename(p) for p in query_paths]
    topk_acc = top_k_accuracy(query_filenames, result, k=k)
    prec_at_k = precision_at_k(query_filenames, result, k=k)

    # Save submission file
    output_dir = os.path.join(project_root, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sub_googlenet.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Submission saved to: {output_path}")

    # Save evaluation metrics to JSON
    total_time = time.time() - start_time
    save_metrics_json(
        model_name="googlenet",
        top_k_accuracy=topk_acc,
        precision=prec_at_k,
        batch_size=batch_size,
        is_finetuned=FINE_TUNE,
        num_classes=len(TRAIN_LOOKUP.values()),
        runtime=total_time,
        loss_function="CrossEntropyLoss",
        num_epochs=epochs if FINE_TUNE else 0,
        final_loss=final_loss if FINE_TUNE else None,
        pooling_type="GeM" if USE_GEM_POOLING else "GAP"
    )


    # Optional: submit to evaluation server
    #submit(result, groupname="Stochastic_thr")
