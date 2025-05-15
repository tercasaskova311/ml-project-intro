import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----- Config -----
K = 5  # top-k images to retrieve
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Image preprocessing -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# ----- Custom Dataset -----
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

# ----- Feature extractor -----
def load_model():
    model = efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()  # Remove classification head
    model.eval().to(DEVICE)
    return model

@torch.no_grad()
def extract_features(model, dataloader):
    features = []
    filenames = []
    for imgs, fnames in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        feats = model(imgs).cpu().numpy()
        features.append(feats)
        filenames.extend(fnames)
    features = np.vstack(features)
    return features, filenames

# ----- Main retrieval logic -----
def main():
    print("[1] Charging EfficientNet...")
    model = load_model()

    print("[2] Estracting features from gallery...")
    gallery_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/gallery"), transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32)
    gallery_feats, gallery_names = extract_features(model, gallery_loader)

    print("[3] Estracting features from query...")
    query_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/query"), transform)
    query_loader = DataLoader(query_dataset, batch_size=32)
    query_feats, query_names = extract_features(model, query_loader)

    print("[4] Calculating similarity and saving JSON...")
    result = []
    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    for i, qname in enumerate(query_names):
        topk_idx = np.argsort(sim_matrix[i])[::-1][:K]
        samples = [gallery_names[idx] for idx in topk_idx]
        result.append({
            "filename": qname,
            "samples": samples
        })

    with open("submission_EfficientNet-B0.json", "w") as f:
        json.dump(result, f, indent=2)

    print("✅ Done! File 'submission.json' ready!!!!!")

if __name__ == "__main__":
    main()
