import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from openai import OpenAI
import os
import json

# === CONFIG ===
MODEL_PATH = "embedding_transformer.pt"
LABEL_ENCODER_PATH = "label_encoder.npy"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

# === MODEL ===
class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = EmbeddingTransformer(1536, 128)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

# === EMBED NEW TEXT ===
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# === TRANSFORM EMBEDDING ===
def get_transformed_embedding(title):
    raw_embedding = get_embedding(title)
    x = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        transformed = model(x)
    return transformed.squeeze(0).numpy()

# === CLASSIFY BY SIMILARITY TO CATEGORY CENTROIDS ===
def classify_by_similarity(transformed_embedding, centroids):
    sims = [np.dot(transformed_embedding, c) / (np.linalg.norm(transformed_embedding) * np.linalg.norm(c)) for c in centroids]
    best_idx = np.argmax(sims)
    return label_classes[best_idx]

# === LOAD CENTROIDS ===
# Optionally, cache category centroids in transformed space
import numpy as np
X = np.load("embeddings.npy")
y = np.load("labels.npy")
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

with torch.no_grad():
    transformed_all = model(X)
    centroids = []
    for cls in range(len(label_classes)):
        cls_embeds = transformed_all[y == cls]
        centroid = cls_embeds.mean(dim=0).numpy()
        centroids.append(centroid)

# === RUN CLASSIFICATION ===
if __name__ == "__main__":
    # Load CSV file
    csv_path = "job_titles.csv"
    print(f"Loading job titles from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    if 'title' not in df.columns:
        raise ValueError("CSV must have a 'title' column")
    
    job_titles = df['title'].tolist()
    actual_categories = df['category'].tolist() if 'category' in df.columns else None
    
    print(f"Processing {len(job_titles)} job titles...\n")
    
    # Store results for JSON output
    results = []
    
    # Process each title
    for i, title in enumerate(job_titles):
        raw_embedding = get_embedding(title)
        transformed = get_transformed_embedding(title)
        predicted_category = classify_by_similarity(transformed, centroids)
        
        # Create result entry
        result_entry = {
            "index": i + 1,
            "title": title,
            "predicted_category": predicted_category,
            "raw_embedding": raw_embedding,
            "transformed_embedding": transformed.tolist(),
            "actual_category": actual_categories[i] if actual_categories else None
        }
        results.append(result_entry)
        
        # Print progress
        if actual_categories:
            print(f"{i+1}. Title: {title}")
            print(f"   Actual: {actual_categories[i]}")
            print(f"   Predicted: {predicted_category}")
            print()
        else:
            print(f"{i+1}. Title: {title} → Predicted Category: {predicted_category}")
    
    # Save results to JSON
    output_file = "inference_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nClassification complete!")
    print(f"Results saved to {output_file}")
    print(f"Total processed: {len(results)} job titles")
