import os
import json
from collections import defaultdict
import numpy as np

# Define the absolute directory where JSON result files are stored
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "results_animals"))

# Dictionary to group result metrics by experiment configuration
groups = defaultdict(list)

print("\n Scanning results in:", RESULTS_DIR)

# Load all JSON metrics safely
for root, _, files in os.walk(RESULTS_DIR):
    for filename in files:
        if filename.endswith(".json"):
            path = os.path.join(root, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping invalid JSON: {filename} → {e}")
                continue
            except Exception as e:
                print(f"[Error] Could not read {filename}: {e}")
                continue

            key = (
                data.get("model_name"),
                data.get("batch_size"),
                data.get("is_finetuned"),
                data.get("num_classes"),
                data.get("num_epochs"),
                data.get("pooling_type", "Unknown")
            )

            metrics = {
                "top_k_accuracy": data.get("top_k_accuracy", 0) or 0,
                "precision_at_k": data.get("precision_at_k", 0) or 0,
                "final_train_loss": data.get("final_train_loss", 0) or 0,
                "runtime_seconds": data.get("runtime_seconds", 0) or 0,
            }

            groups[key].append(metrics)

# Aggregate and print results
print("\n Aggregated Metrics per Model Configuration:\n")
for key, runs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
    model_name, batch_size, is_finetuned, num_classes, num_epochs, pooling_type = key
    accs = [r["top_k_accuracy"] for r in runs]
    precs = [r["precision_at_k"] for r in runs]
    losses = [r["final_train_loss"] for r in runs]
    runtimes = [r["runtime_seconds"] for r in runs]

    print(f"Model: {model_name}, Batch: {batch_size}, Fine-tuned: {is_finetuned}, "
          f"Classes: {num_classes}, Epochs: {num_epochs}, Pooling: {pooling_type}")
    print(f"   Runs: {len(runs)}")
    print(f"   Avg Accuracy (at least 1 correct class):  {np.mean(accs):.4f}")
    print(f"   Avg Precision@K:   {np.mean(precs):.4f}")
    print(f"   Avg Final Loss:    {np.mean(losses):.4f}")
    print(f"   Avg Runtime (sec): {np.mean(runtimes):.2f}")
