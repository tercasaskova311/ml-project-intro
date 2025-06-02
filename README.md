# Image Retrieval with Deep Learning

This repository contains our solution to the **Image Retrieval Competition** held in the *Intro to Machine Learning* course at the University of Trento (2025 edition). The objective was to develop a pipeline that retrieves the **Top-k most semantically similar gallery images** for each query image.

We implemented and evaluated multiple deep learning models, including CLIP, DINOv2, EfficientNet, ResNet, and GoogLeNet, exploring different pooling strategies (GeM vs GAP) and fine-tuning settings.

---

## Dataset Structure

The dataset is structured as follows:

data/
 ├── training/  -> Labeled training images in class folders
 └── test/
  ├── query/ -> Unlabeled query images
  └── gallery/ -> Unlabeled gallery images


Each image in `training/` is used to learn class-discriminative embeddings. Retrieval is evaluated based on how well query images retrieve same-class samples from the gallery.

---

## Models Implemented NEEDS TO BE CHANGED

| Model       | Variants       | Fine-tuning | Pooling     | Script File                    |
|-------------|----------------|-------------|-------------|--------------------------------|
| **CLIP**    | ViT-L/14       | ✅           | internal    | `Clip.py`                      |
| **DINOv2**  | ViT-B/14       | ✅/❌         | GAP         | `dino2_retrieval_*.py`         |
| **EffNet**  | B0, B3         | ✅/❌         | GeM or GAP  | `EfficientNet.py`              |
| **ResNet**  | 18–152         | ✅/❌         | GAP         | `ResNet.py`                    |
| **GoogLeNet**| base          | ✅/❌         | GeM or GAP  | `GoogleNet_gem_gap.py`         |

Each script supports standalone execution and produces:
- Feature extraction
- Retrieval results
- Submission file
- Logged metrics

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ml-project-intro.git
   cd ml-project-intro

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a model script**  
   For example, to run the CLIP-based retrieval pipeline:  
   ```bash
   python models/Clip.py
   ```

4. **Evaluate aggregated metrics**  
   ```bash
   python src/metrics_analysis.py
   ```

5. **Visualize retrieval results**  
   ```bash
   python src/visualize_submission.py
   ```
   This script works only  if you put your data in the correct directories, specified in the code. 

6. **Submit results to the evaluation server**  
   ```bash
   python submit.py
   ```
   The server only works if you are logged with the university wi-fi.

---

## Metrics and Evaluation

### Top-k Accuracy

Top-k Accuracy measures how often the correct class for a query image appears among the top-k retrieved gallery images. In formula form:

```
Top-k Accuracy = (Number of queries with at least one correct match in top-k) / (Total number of queries)
```

Only the presence of a matching class matters, not the rank.

---

### Metrics Output Format

Each model script writes one JSON file into the **results/** folder. An example entry:

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

To aggregate and compare all saved metrics, run:

```
python src/metrics_analysis.py
```

This script prints grouped average scores across runs for comparison.

---

## Submission Format

Each model generates a JSON file in **submissions/** with this format:

```json
{
  "query_001.jpg": ["gallery_123.jpg", "gallery_045.jpg", "gallery_084.jpg"],
  "query_002.jpg": ["gallery_067.jpg", "gallery_208.jpg", "gallery_301.jpg"]
}
```

To submit this file, the `submit.py` script sends a POST request:

```python
submit(results, groupname= groupname , url="http://tatooine.disi.unitn.it:3001/retrieval/")
```

The server responds with a “Top-k Accuracy” value computed against ground-truth labels.

---

## Visual Output

To generate a side-by-side comparison of query images and their top-k retrievals, run:

```
python src/visualize_submission.py
```

This creates PNG files under **visualizations/**. Each figure shows the query on the left and the top-k retrieved gallery images on the right. Just be sure you have set the dataset in the correct format and that the directories are right. 

---

## Requirements

Install all required Python packages in one step:

```
pip install -r requirements.txt
```

---

## Authors

**Team:** Stochastic_thr  
**Course:** Intro to Machine Learning 2024-2025  
**Institution:** University of Trento

---

## Future Work

- Add support for larger CLIP models (e.g., ViT-H/14).  
- Replace brute-force cosine search with FAISS for faster retrieval.  
- Experiment with metric learning losses (Triplet, ArcFace).  
- Ensemble multiple model embeddings or fuse scores.

---
