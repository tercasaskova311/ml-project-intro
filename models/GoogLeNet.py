import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import json

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

# ------------------------------
# 4. Build GoogLeNet feature extractor
# ------------------------------
def build_googlenet_extractor(device='cuda'):
    weights = torchvision.models.GoogLeNet_Weights.DEFAULT
    model = torchvision.models.googlenet(weights=weights, aux_logits=True)
    model.aux1 = torch.nn.Identity()
    model.aux2 = torch.nn.Identity()
    model.fc = torch.nn.Identity()  # Output: 1024-d feature vector
    return model.to(device).eval()

# ------------------------------
# 5. Extract image embeddings
# ------------------------------
@torch.no_grad()
def extract_embeddings(loader, model, device='cuda'):
    embeddings = []
    for images, _ in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model(images)
        embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)

# ------------------------------
# 6. Compute top-k similar images
# ------------------------------
def retrieve_topk(query_embs, gallery_embs, query_paths, gallery_paths, k=10):
    query_embs = F.normalize(query_embs, dim=1)
    gallery_embs = F.normalize(gallery_embs, dim=1)
    sim_matrix = query_embs @ gallery_embs.T  # Cosine similarity
    topk_indices = sim_matrix.topk(k, dim=1, largest=True)[1]

    results = []
    for i, indices in enumerate(topk_indices):
        query_filename = os.path.basename(query_paths[i])
        top_filenames = [os.path.basename(gallery_paths[j]) for j in indices.tolist()]
        results.append({
            "filename": query_filename,
            "samples": top_filenames
        })
    return results

# ------------------------------
# 7. Main script execution
# ------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load datasets
    query_dataset = ImagePathDataset(query_folder, transform=transform)
    gallery_dataset = ImagePathDataset(gallery_folder, transform=transform)

    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)

    # Build the GoogLeNet feature extractor
    model = build_googlenet_extractor(device)

    # Extract features
    query_embs = extract_embeddings(query_loader, model, device)
    gallery_embs = extract_embeddings(gallery_loader, model, device)

    # Perform retrieval
    results = retrieve_topk(query_embs, gallery_embs, query_dataset.paths, gallery_dataset.paths, k=5)

    # Save output JSON to the submissions folder
    output_dir = os.path.join(project_root, 'submissions')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'sub_googlenet.json')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Submission saved to: {output_file}")