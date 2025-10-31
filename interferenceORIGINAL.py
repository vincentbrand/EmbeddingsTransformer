import torch
import torch.nn as nn
import numpy as np
import openai
import os

# === CONFIG ===
MODEL_PATH = "embedding_transformer.pt"
LABEL_ENCODER_PATH = "label_encoder.npy"
EMBEDDING_MODEL = "text-embedding-3-small"

# Set OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
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
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response["data"][0]["embedding"]

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
    while True:
        title = input("\nEnter job title (or 'exit'): ").strip()
        if title.lower() == "exit":
            break
        transformed = get_transformed_embedding(title)
        category = classify_by_similarity(transformed, centroids)
        print(f"Predicted Category: {category}")
