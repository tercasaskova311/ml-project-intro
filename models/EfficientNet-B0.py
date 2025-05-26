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
import time

# ----- Config -----
K = 10  # top-k images to retrieve
FINE_TUNE = False  # Toggle this to enable/disable fine-tuning
USE_GEM = True    # Toggle this to switch between GAP and GeM
batch_size=32
epochs=2


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

# effiecent net is used to extract features from images not for classification!!
# metric: The correct loss function: MSELoss. This applies to both pooling types:


 # ----- EfficientNet with GeM -----
class EfficientNetWithGeM(torch.nn.Module):
    def __init__(self, num_classes=None, fine_tune=False):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.gem_pool = GeM()
        self.flatten = torch.nn.Flatten()
        self.out_dim = base.classifier[1].in_features  # 1280 for B0
        # Classification head (optional)
        self.classifier = None
        if fine_tune and num_classes is not None:
            self.classifier = torch.nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gem_pool(x)
        x = self.flatten(x)
        if self.classifier:
            x = self.classifier(x)
        return x

# ----- EfficientNet with GAP -----
# -->use crossentropy loss


# def load_model(): #computed mse loss but got always 0.0
#     base_model = efficientnet_b0(pretrained=True)
#     model = torch.nn.Sequential(*list(base_model.children())[:-1])  # Remove classifier
#     model.eval().to(DEVICE)
#     return model

def load_model(num_classes=None, fine_tune=False): # supervised fine-tuning using CrossEntropyLoss
    base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if fine_tune and num_classes is not None:
        in_features = base_model.classifier[1].in_features
        base_model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    return base_model.to(DEVICE)


def load_model_GEM(num_classes=None, fine_tune=False):
    model = EfficientNetWithGeM(num_classes=num_classes, fine_tune=fine_tune)
    return model.to(DEVICE)

# ----- Fine-tuning (Optional) -----
def finetune_model(model, dataloader, epochs, lr=1e-4):
    print("Fine-tuning model on training data...")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  # Dummy self-supervised loss
    final_loss = None

    for epoch in range(epochs):
        running_loss = 0.0
        for img1, img2 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        # for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):  old version
        #     imgs = imgs.to(DEVICE) old version
            #feats = model(imgs) if only one
            feat1 = model(img1)
            feat2 = model(img2)
            loss = criterion(feat1, feat2)
            # loss = criterion(feats, feats.detach()) old version

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        final_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {final_loss:.4f}")

    model.eval()
    print("Fine-tuning completed.")
    return final_loss


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
    
class AugmentedImageFolder(ImageFolder):  # <-- called only  when FINE_TUNE = True, and only in the main() function, where the training dataset is defined for fine-tuning.
    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            img1 = self.transform(image)
            img2 = self.transform(image)
        return img1, img2

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


def calculate_top_k_accuracy(query_feats, gallery_feats, query_names, gallery_names, k=10):
    correct = 0
    total = len(query_names)

    def extract_class(name):
        return name.split("_")[0]  # modify as needed

    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    for i, qname in enumerate(query_names):
        qclass = extract_class(qname)
        topk_idx = np.argsort(sim_matrix[i])[::-1][:k]
        retrieved = [extract_class(gallery_names[j]) for j in topk_idx]
        if qclass in retrieved:
            correct += 1

    acc = correct / total
    print(f"Top-{k} Accuracy: {acc:.4f}")
    return acc


# ----- Main Logic -----
def main():

    start_time = time.time()
    print("[1] Loading EfficientNet model...")
    model = load_model_GEM() if USE_GEM else load_model()

    if FINE_TUNE:
        print("[1.5] Fine-tuning is ENABLED.")
        train_dataset = AugmentedImageFolder(os.path.join(DATA_DIR, "training"), transform)
        # train_dataset = ImageFolder(os.path.join(DATA_DIR, "training"), transform) old version
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        final_loss = finetune_model(model, train_loader, epochs, lr=1e-4)
    else:
        final_loss = None
        print("[1.5] Fine-tuning is DISABLED.")

    print("[2] Extracting features from gallery...")
    gallery_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/gallery"), transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size)
    gallery_feats, gallery_names = extract_features(model, gallery_loader)

    print("[3] Extracting features from query...")
    query_dataset = ImagePathDataset(os.path.join(DATA_DIR, "test/query"), transform)
    query_loader = DataLoader(query_dataset, batch_size)
    query_feats, query_names = extract_features(model, query_loader)

    print("[4] Calculating similarity and saving JSON...")
    result = {}
    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    for i, qname in enumerate(query_names):
        topk_idx = np.argsort(sim_matrix[i])[::-1][:K]
        samples = [gallery_names[idx] for idx in topk_idx]
        result[qname] = samples


    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submissions"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sub_efficientnet.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done! File saved to: {output_path}")

    print("[5] Evaluating Top-K accuracy...")
    topk_acc = calculate_top_k_accuracy(query_feats, gallery_feats, query_names, gallery_names, k=K)

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    print("[6] Saving metrics...")
    num_classes = len(train_dataset.classes) if FINE_TUNE else None
    
    pooling_type = "GeM" if USE_GEM else "GAP"

    save_metrics_json(
    model_name="efficientnet_b0",
    top_k_accuracy=topk_acc,
    batch_size=batch_size,
    is_finetuned=FINE_TUNE,
    num_classes=num_classes,
    runtime=total_time,
    loss_function="MSELoss" if FINE_TUNE else "NotApplicable (not fine tuned)",
    num_epochs=epochs if FINE_TUNE else 0,
    final_loss=final_loss,
    pooling_type=pooling_type
)


from datetime import datetime

def save_metrics_json(
    model_name,
    top_k_accuracy,
    batch_size,
    is_finetuned,
    num_classes=None,
    runtime=None,
    loss_function="MSELoss",
    num_epochs=None,
    final_loss=None,
    pooling_type=None 
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
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        "pooling_type": pooling_type
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
