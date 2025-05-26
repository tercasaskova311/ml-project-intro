import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import json
from torchvision.datasets import ImageFolder
import torch.optim as optim

#CONFIG
k=5
batch_size= 16
image_size = (224, 224)
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
FINE_TUNE = False  # Set to False to skip training
TRAIN_LAST_LAYER_ONLY = True  # Set to False to fine-tune entire model
epochs = 5
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------
# 1. Define project paths
# ------------------------------
cwd = os.getcwd()
if os.path.isdir(os.path.join(cwd, 'data', 'training')):
    project_root = cwd
elif os.path.isdir(os.path.join(cwd, '..', 'data', 'training')):
    project_root = os.path.abspath(os.path.join(cwd, '..'))
else:
    raise RuntimeError("Folder 'data/training' not found. Run from the project root.")

# Test image folders
root_dir_test = os.path.join(project_root, 'data', 'test')
query_folder = os.path.join(root_dir_test, 'query')
gallery_folder = os.path.join(root_dir_test, 'gallery')

# ------------------------------
# 2. Custom Dataset for Image Paths
# ------------------------------
class ImagePathDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

# ------------------------------
# 3. Image transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean,  # ImageNet mean
                         std=normalize_std)   # ImageNet std
])

# ------------------------------
# 4. Build GoogLeNet feature extractor
# ------------------------------
def build_googlenet_extractor(device):
    weights = torchvision.models.GoogLeNet_Weights.DEFAULT
    model = torchvision.models.googlenet(weights=weights, aux_logits=True)
    model.aux1 = torch.nn.Identity()
    model.aux2 = torch.nn.Identity()
    model.fc = torch.nn.Identity()  # Output: 1024-d feature vector
    return model.to(device).eval()

# ------------------------------
# 4.5 . FINE TUNING
# ------------------------------

def fine_tune_model(model, train_loader, device, num_classes, train_last_layer_only=True, epochs=epochs, learning_rate=learning_rate):
    print(f"[FT] Starting fine-tuning for {epochs} epochs...")
    
    model.train()

    if train_last_layer_only:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    final_loss = None

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        final_loss = running_loss / len(train_loader)
        print(f"[FT] Epoch {epoch+1}: Loss={final_loss:.3f}, Acc={acc:.2f}%")

    print("[FT] Fine-tuning complete.")
    return model.eval(), final_loss




# ------------------------------
# 5. Extract image embeddings
# ------------------------------
@torch.no_grad()
def extract_embeddings(loader, model, device):
    embeddings = []
    for images, _ in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model(images)
        embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)

# ------------------------------
# 6. Compute top-k similar images
# ------------------------------
def retrieve_topk(query_embs, gallery_embs, query_paths, gallery_paths, k):
    query_embs = F.normalize(query_embs, dim=1)
    gallery_embs = F.normalize(gallery_embs, dim=1)
    sim_matrix = query_embs @ gallery_embs.T  # Cosine similarity
    topk_indices = sim_matrix.topk(k, dim=1, largest=True)[1]

    results = {}
    for i, indices in enumerate(topk_indices):
        query_filename = os.path.basename(query_paths[i])
        top_filenames = [os.path.basename(gallery_paths[j]) for j in indices.tolist()]
        results[query_filename] = top_filenames
    return results


def calculate_top_k_accuracy(results):
    def extract_class(filename):
        return filename.split("_")[0]  # Adatta alla tua convenzione (es: class42_img12.jpg)

    correct = 0
    total = len(results)

    for query_filename, retrieved_list in results.items():
        query_class = extract_class(query_filename)
        retrieved_classes = [extract_class(fn) for fn in retrieved_list]
        if query_class in retrieved_classes:
            correct += 1

    acc = correct / total
    print(f"Top-{k} Accuracy: {acc:.4f}")
    return acc



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
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
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


# ------------------------------
# 7. Main script execution
# ------------------------------
if __name__ == '__main__':
    
    import time
    start_time = time.time()

    # Load datasets
    query_dataset = ImagePathDataset(query_folder, transform=transform)
    gallery_dataset = ImagePathDataset(gallery_folder, transform=transform)

    query_loader = DataLoader(query_dataset, batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size, shuffle=False)

    # Build the GoogLeNet feature extractor WITH AND WITHOUT FINETUNING
    if FINE_TUNE: 
        print("[FT] Loading training data...")
        train_dir = os.path.join(project_root, 'data', 'training')
        train_dataset = ImageFolder(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Rebuild model for training
        model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT, aux_logits=True)
        model.aux1 = torch.nn.Identity()
        model.aux2 = torch.nn.Identity()

        model, final_loss = fine_tune_model(model, train_loader, device, num_classes=len(train_dataset.classes), train_last_layer_only=TRAIN_LAST_LAYER_ONLY,epochs=epochs, learning_rate=learning_rate)

    else:
        model = build_googlenet_extractor(device)
        final_loss = None


    # Extract features
    query_embs = extract_embeddings(query_loader, model, device)
    gallery_embs = extract_embeddings(gallery_loader, model, device)

    # Perform retrieval
    results = retrieve_topk(query_embs, gallery_embs, query_dataset.paths, gallery_dataset.paths, k)

    # Save output JSON to the submissions folde
    output_dir = os.path.join(project_root, 'submissions')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'sub_googlenet.json')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

    print(f"Submission saved to: {output_file}")

    print("Calculating Top-K accuracy...")
    topk_acc = calculate_top_k_accuracy(results)

    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

    num_classes = len(train_dataset.classes) if FINE_TUNE else None
    save_metrics_json(
    model_name="googlenet",
    top_k_accuracy=topk_acc,
    batch_size=batch_size,
    is_finetuned=FINE_TUNE,
    num_classes=num_classes,
    runtime=total_time,
    loss_function="CrossEntropyLoss" if FINE_TUNE else "None",
    num_epochs=epochs if FINE_TUNE else None,
    final_loss=final_loss
)

