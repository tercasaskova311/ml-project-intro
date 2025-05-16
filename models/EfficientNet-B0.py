import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----- Config -----
K = 5  # top-k images to retrieve
FINE_TUNE = True  # Toggle this to enable/disable fine-tuning
USE_GEM = False    # Toggle this to switch between GAP and GeM
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
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

# ----- GeM Pooling Layer ----- OPTIONAL!!!!! NOW WE ARE USING GAP BUT WE CAN TRY TO CHANGE IT WITH GEM IN THE MAIN PART
# GeM (Generalized Mean Pooling) is a pooling layer that generalizes the average and max pooling operations.    
# It is often used in image classification tasks to extract features from convolutional layers.
# GeM is useful for tasks where the spatial distribution of features is important, such as in image retrieval or object detection.
# GAP (Global Average Pooling) is a simpler pooling operation that computes the average of all spatial locations in the feature map.
# GAP is often used in image classification tasks to reduce the dimensionality of the feature map before passing it to a fully connected layer.
# GeM is more flexible than GAP, as it can adapt to different spatial distributions of features.
# GeM is more computationally expensive than GAP, as it requires computing the p-th power of the feature map.

class GeM(torch.nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1. / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.item():.4f})"

# ----- EfficientNet with GeM -----
class EfficientNetWithGeM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.gem_pool = GeM()
        self.flatten = torch.nn.Flatten()
        self.out_dim = base.classifier[1].in_features  # 1280 for B0

    def forward(self, x):
        x = self.features(x)
        x = self.gem_pool(x)
        x = self.flatten(x)
        return x

# ----- EfficientNet with GAP -----
def load_model():
    base_model = efficientnet_b0(pretrained=True)
    model = torch.nn.Sequential(*list(base_model.children())[:-1])  # Remove classifier
    model.eval().to(DEVICE)
    return model

def load_model_GEM():
    model = EfficientNetWithGeM()
    model.eval().to(DEVICE)
    return model

# ----- Fine-tuning (Optional) -----
def finetune_model(model, dataloader, epochs=1, lr=1e-4):
    print("🔧 Fine-tuning model on training data...")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  # Dummy self-supervised loss

    for epoch in range(epochs):
        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            loss = criterion(feats, feats.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    print("Fine-tuning completed.")

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

# ----- Feature Extraction -----
@torch.no_grad()
def extract_features(model, dataloader):
    features = []
    filenames = []
    for imgs, fnames in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        feats = model(imgs)

        # Flatten if needed
        if feats.dim() == 4:
            feats = feats.view(feats.size(0), -1)

        feats = feats.cpu().numpy()
        features.append(feats)
        filenames.extend(fnames)

    features = np.vstack(features)
    return features, filenames


# ----- Main Logic -----
def main():
    print("[1] Loading EfficientNet model...")
    model = load_model_GEM() if USE_GEM else load_model()

    if FINE_TUNE:
        print("[1.5] Fine-tuning is ENABLED.")
        train_dataset = ImageFolder(os.path.join(DATA_DIR, "training"), transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        finetune_model(model, train_loader)
    else:
        print("[1.5] Fine-tuning is DISABLED.")

    print("[2] Extracting features from gallery...")
    gallery_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/gallery"), transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32)
    gallery_feats, gallery_names = extract_features(model, gallery_loader)

    print("[3] Extracting features from query...")
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

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submissions"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sub_efficientnet.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done! File saved to: {output_path}")

if __name__ == "__main__":
    main()
