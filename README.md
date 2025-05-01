# ML Image Recognition Project

This project aims to develop a neural network model for image recognition. The project includes scripts for feature extraction, similarity search, and submission creation.

## How to Run the Code

1. Clone the repository:

    ```bash
    git clone https://github.com/tercasaskova311/ml-project-intro.git
    cd ml-project-intro
    ```

2. Set up the environment:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the feature extraction script:

    ```bash
    python scripts/feature_extraction.py
    ```

4. Run the retrieval and similarity search script:

    ```bash
    python scripts/retrieval_similarity_search.py
    ```

5. Generate the final JSON submission:

    ```bash
    python scripts/final_submission.py
    ```

## Libraries Used

- PyTorch
- scikit-learn
- pandas
- numpy
- etc.

## Preparing the Submission JSON

The submission should be a JSON file that includes the following format:

```json
{
  "id": "image_1",
  "prediction": "label_1"
}

