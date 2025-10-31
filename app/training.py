import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# === CONFIG ===
EMBEDDING_PATH = "embeddings.npy"
LABELS_PATH = "labels.npy"
LABEL_ENCODER_PATH = "label_encoder.npy"
MODEL_SAVE_PATH = "embedding_transformer.pt"
EPOCHS = 200
LR = 0.001
BATCH_SIZE = 32
EMBEDDING_DIM = 1536

# === LOAD DATA ===
X = np.load(EMBEDDING_PATH)
y = np.load(LABELS_PATH)
label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
num_classes = len(label_classes)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

model = EmbeddingTransformer(EMBEDDING_DIM, 128)  # Transform to 128D task-specific space
optimizer = optim.Adam(model.parameters(), lr=LR)

# === TRAINING: Learn to move closer to class centroids ===
def compute_centroids(embeddings, labels, num_classes):
    centroids = []
    for cls in range(num_classes):
        cls_embeddings = embeddings[labels == cls]
        centroid = cls_embeddings.mean(dim=0)
        centroids.append(centroid)
    return torch.stack(centroids)

loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    total_loss = 0

    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()

        transformed = model(batch_x)  # Shape: (B, 128)

        # Get class centroids in current transformed space
        with torch.no_grad():
            transformed_all = model(X_train)
            centroids = compute_centroids(transformed_all, y_train, num_classes)

        target_centroids = centroids[batch_y]
        loss = loss_fn(transformed, target_centroids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === SAVE ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Saved embedding transformer model.")
