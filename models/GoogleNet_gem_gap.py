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

# ---------------- CONFIGURATION ----------------
k = 10  # Top-K retrieval
batch_size = 32
image_size = (224, 224)  # GoogLeNet standard input size
normalize_mean = [0.485, 0.456, 0.406]  # Imagenet normalization mean
normalize_std = [0.229, 0.224, 0.225]   # Imagenet normalization std
FINE_TUNE = True
TRAIN_LAST_LAYER_ONLY = True  # Fine-tune only last layer for faster convergence
epochs = 5
learning_rate = 1e-5
USE_GEM_POOLING = True  # Use GeM pooling instead of Global Average Pooling (GAP)
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
root_dir_test = os.path.join(project_root, 'data', 'test')
query_folder = os.path.join(root_dir_test, 'query')
gallery_folder = os.path.join(root_dir_test, 'gallery')

#====== TRAIN LOOP - TRAIN FOLDER ======
TRAIN_LOOKUP = {}
project_root = 'data'
train_root = os.path.join(project_root, 'training') 

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
    model.aux1 = torch.nn.Identity()  # Remove aux classifiers
    model.aux2 = torch.nn.Identity()
    # Replace pooling with GeM or GAP
    model.avgpool = GeM() if USE_GEM_POOLING else torch.nn.AdaptiveAvgPool2d((1, 1))
    # Remove final FC layer to get embeddings instead of class scores
    model.fc = torch.nn.Identity()
    return model.to(device).eval()

# -------- Fine-tune Model on Training Data --------
def fine_tune_model(model, train_loader, device, num_classes,
                    train_last_layer_only=True, epochs=epochs, learning_rate=learning_rate):
    print(f"[FT] Starting fine-tuning for {epochs} epochs...")
    model.train().to(device)

    # Replace FC with new classifier for current dataset classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)

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

# -------- Calculate Top-K Retrieval Accuracy --------
def calculate_top_k_accuracy(results):
    # Helper to extract class name from filename using training lookup
    def extract_class(filename):
        # If filename uses "_" to separate class info, use that, else fallback to TRAIN_LOOKUP
        if "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2 and not parts[0].isdigit():
                return "_".join(parts[:-1])
        base = os.path.basename(filename)
        return TRAIN_LOOKUP.get(base, "unknown")

    correct = 0
    total = len(results)

    for query_filename, retrieved_list in results.items():
        query_class = extract_class(query_filename)
        retrieved_classes = [extract_class(fn) for fn in retrieved_list]
        if query_class in retrieved_classes:
            correct += 1

    acc = correct / total
    print(f" Top-{k} Accuracy: {acc:.4f}")
    return acc


# ======= Metrics saving =======
def save_metrics_json(model_name, top_k_accuracy, batch_size, is_finetuned, num_classes=None, runtime=None, loss_function="MSELoss", num_epochs=None, final_loss=None, pooling_type=None):
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

    # Set project root based on current working directory and data folders
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, 'data', 'training')):
        project_root = cwd
    elif os.path.isdir(os.path.join(cwd, '..', 'data', 'training')):
        project_root = os.path.abspath(os.path.join(cwd, '..'))
    else:
        raise RuntimeError("Folder 'data/training' not found. Run from project root.")

    # Create training lookup dict for accuracy evaluation
    TRAIN_LOOKUP = {}
    train_root = os.path.join(project_root, 'data', 'training')
    for class_name in os.listdir(train_root):
        class_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.png')):
                TRAIN_LOOKUP[img_name] = class_name

    # Define query and gallery folders
    root_dir_test = os.path.join(project_root, 'data', 'test')
    query_folder = os.path.join(root_dir_test, 'query')
    gallery_folder = os.path.join(root_dir_test, 'gallery')

    # Load query and gallery datasets and dataloaders
    query_dataset = ImagePathDataset(query_folder, transform=transform)
    gallery_dataset = ImagePathDataset(gallery_folder, transform=transform)
    query_loader = DataLoader(query_dataset, batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size, shuffle=False)

    start_time = time.time()

    if FINE_TUNE:
        print("[FT] Loading training data...")
        from torchvision.datasets import ImageFolder
        train_dataset = ImageFolder