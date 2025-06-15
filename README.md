
# From CNNs to Transformers: Top-k Image Retrieval in a Competition Setting

This repository contains our solution to the **Image Retrieval Competition** held in the **Intro to Machine Learning** course at the University of Trento. The objective was to develop a pipeline that retrieves the **Top-k most similar images** for each query image.

We implemented and evaluated multiple deep learning models, including CLIP, DINOv2, EfficientNet, ResNet, and GoogLeNet.
Each model extracts feature embeddings from query and gallery images using a pretrained or fine-tuned encoder. Retrieval is performed by ranking gallery images according to cosine similarity with the query embedding. We tested both frozen and fine-tuned variants, and evaluated the impact of different pooling strategies (GAP vs. GeM) on retrieval performance.

---

## Report

A full technical report is provided under `report/report_cvpr2025.pdf`, detailing all experimental settings, implementation choices, and results. The report follows the CVPR template and includes figures, tables, and references for all tested models.

---

## Project Structure

- `models/`: main model scripts (CLIP, DINOv2, EfficientNet, ResNet, GoogLeNet)
- `src/`: metric analysis and visualization tools
- `submissions/`: generated submission files
- `results/`: JSON logs of performance per model
- `report/`: CVPR-style report of the project
- `utils/`: utility functions for computing retrieval metrics (e.g., Top-k accuracy, Precision@K)

---

## Dataset Structure

The dataset is structured as follows:

```
data/
├── training/           # Labeled training images in class folders
└── test/
    ├── query/          # Unlabeled query images
    └── gallery/        # Unlabeled gallery images
```

Each image in `training/` is used to learn class-discriminative embeddings. Retrieval is evaluated based on how well query images retrieve same-class samples from the gallery.

---

## Models Implemented

| Model         | Type      | Variants         | Fine-tuning | Pooling     | Script                         |
|---------------|-----------|------------------|-------------|-------------|--------------------------------|
| CLIP          | ViT       | ViT-B/32, B/16, L/14 | Yes / No | Internal    | `Clip.py`, `Clip_test.py`                     |
| DINOv2        | ViT       | facebook/dinov2-base        | Yes / No     | GAP         | `dino2_retrieval_*.py`         |
| EfficientNet  | CNN       | B0, B3            | Yes / No     | GeM / GAP   | `EfficientNet.py`              |
| ResNet        | CNN       | 34, 50, 101, 152 | Yes / No   | GAP         | `ResNet.py`                    |
| GoogLeNet     | CNN       | base              | Yes / No     | GeM / GAP   | `GoogleNet_gem_gap.py`         |

*“Yes / No” indicates whether both frozen and fine-tuned variants were tested for that model.*
> **Note**:`Clip_test.py` contains the updated pipeline used in the post-competition phase, including proper layer unfreezing, contrastive loss, and memory optimizations.



Each script supports standalone execution and produces:
- Feature extraction
- Retrieval results
- Submission file
- Logged metrics

---

## Datasets Used for Testing

In addition to the competition dataset, given on competition day, we used the following public dataset to test and validate our models:
 
- [Animal Image Dataset (90 Categories)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

---

## Competition Task

> The goal of the competition was **image retrieval**. The task consisted of matching **real photos of celebrities** (queries) with **synthetic images** (gallery) of the same celebrities, generated in a different visual style.  
> For each query image, participants needed to retrieve the 10 most likely matching gallery images.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ml-project-intro.git
   cd ml-project-intro
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run a model script:  
   Example with CLIP:
   ```bash
   python Clip.py
   ```

4. Evaluate aggregated metrics:  
   ```bash
   python src/metrics_analysis.py
   ```
   This script prints the performance of all your runs (if you've saved the JSONs correctly in `results/`).

5. Visualize retrieval results:  
   ```bash
   python src/visualize_submission.py
   ```
   Ensure your dataset is in the correct directory format as expected by the scripts.

6. Submit results to the server:  
   ```bash
   python submit.py
   ```
   The submission script works **only with the dataset provided during the competition**, which is ignored in Git (see `.gitignore`).  
   Also, **you must be connected to the University of Trento Wi-Fi** for the server to be reachable.

---

## Metrics and Evaluation

### Competition Evaluation: Weighted Top-k Accuracy

The official metric used by the competition server assigns a weighted score based on the presence and rank of the correct identity among the retrieved results:

- **600 points** if the correct match is ranked first (Top-1)
- **300 points** if it appears in positions 2–5 (Top-5)
- **100 points** if it appears in positions 6–10 (Top-10)

The final score is the mean across all queries.

**Best result:**  
791.82 weighted Top-k accuracy  
Model: CLIP ViT-L/14, fully fine-tuned with contrastive + cross-entropy loss (post-competition)

---

### Pre-Competition Metric: Precision@K

For pre-competition testing on the Animals dataset, we used Precision@K, defined as:

Precision@K = (Number of correct matches in top-K) / K

This value was computed per query and averaged over all queries.

**Best result:**  
0.8513 Precision@K  
Model: EfficientNet-B3, fine-tuned with GAP pooling

---

### Metrics Format

Each script generates a JSON file in `results/` that looks like:

```json
{
  "model_name": "clip-vit-l14",
  "top_k": 10,
  "top_k_accuracy": 0.5104,
  "batch_size": 32,
  "is_finetuned": true,
  "num_classes": 30,
  "runtime_seconds": 42.1,
  "loss_function": "CrossEntropyLoss",
  "num_epochs": 10,
  "final_train_loss": 0.3471,
  "pooling_type": "GeM"
}
```

Run `python src/metrics_analysis.py` to see aggregated metrics across all runs.

---

## Submission Format

Each script saves a file like this in `submissions/`:

```json
{
  "query_001.jpg": ["gallery_123.jpg", "gallery_045.jpg", "gallery_084.jpg"],
  "query_002.jpg": ["gallery_067.jpg", "gallery_208.jpg", "gallery_301.jpg"]
}
```

You can submit your result by calling:

```python
submit(results, groupname="groupname")
```

---

## Visualization

To generate visual comparisons of retrievals:

```bash
python src/visualize_submission.py
```

Make sure that your test set is correctly located under `data/test/query/` and `data/test/gallery/`.

---

## Course Context

This project was developed as part of the **Introduction to Machine Learning** course in the Master degree in **Data Science** at the **University of Trento** (Academic Year 2024–2025).

---

## Authors

This project was developed by:

- Silvia Bortoluzzi  
- Diego Conti  
- Sara Lammouchi  
- Tereza Sásková