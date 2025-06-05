import os
import json
from collections import defaultdict
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "results_animals"))

groups = defaultdict(list)

print("Scanning results in:", RESULTS_DIR)

# Load all JSON metrics safely
for filename in os.listdir(RESULTS_DIR):
    if filename.endswith(".json"):
        path = os.path.join(RESULTS_DIR, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON: {filename} → {e}")
            continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
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
print("\nAggregated Metrics per Model Configuration:\n")
for key, runs in groups.items():
    model_name, batch_size, is_finetuned, num_classes, num_epochs, pooling_type = key
    accs = [r["top_k_accuracy"] for r in runs]
    losses = [r["final_train_loss"] for r in runs]
    runtimes = [r["runtime_seconds"] for r in runs]

    print(f"Model: {model_name}, Batch: {batch_size}, Fine-tuned: {is_finetuned}, "
          f"Classes: {num_classes}, Epochs: {num_epochs}, Pooling: {pooling_type}")
    print(f"   Runs: {len(runs)}")
    print(f"   Avg Accuracy:      {np.mean(accs):.4f}")
    print(f"   Avg Final Loss:    {np.mean(losses):.4f}")
    print(f"   Avg Runtime (sec): {np.mean(runtimes):.2f}\n")
