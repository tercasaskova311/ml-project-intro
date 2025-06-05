import os

def extract_class(filename):
    """
    Extract class identifier from the filename.
    Assumes class is the prefix before the first underscore.
    Example: "dog_123.jpg" → "dog"
    """
    return filename.split("_")[0]

def top_k_accuracy(query_paths, retrievals, k=10):
    """
    Computes top-k accuracy: proportion of queries where at least one
    retrieved image shares the class with the query.
    """
    correct = 0
    total = 0
    for qname in query_paths:
        qfile = os.path.basename(qname)
        q_class = extract_class(qfile)
        retrieved_classes = [extract_class(name) for name in retrievals[qfile]]
        if q_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"[METRIC] Top-{k} Accuracy: {acc:.4f}")
    return acc

def precision_at_k(query_paths, retrievals, k=10):
    """
    Computes Precision@k: average proportion of top-k retrieved images
    that share the class with the query.
    """
    total_precision = 0
    for qname in query_paths:
        qfile = os.path.basename(qname)
        q_class = extract_class(qfile)
        retrieved_classes = [extract_class(name) for name in retrievals[qfile]]
        correct_retrieved = sum(1 for c in retrieved_classes if c == q_class)
        precision = correct_retrieved / k
        total_precision += precision
    avg_precision = total_precision / len(query_paths) if query_paths else 0.0
    print(f"[METRIC] Precision@{k}: {avg_precision:.4f}")
    return avg_precision
