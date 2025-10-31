import pandas as pd
import numpy as np
from openai import OpenAI
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time

# === CONFIG ===
CSV_PATH = "job_titles.csv"
EMBEDDING_PATH = "embeddings.npy"
LABELS_PATH = "labels.npy"
LABEL_ENCODER_PATH = "label_encoder.npy"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

# === LOAD DATA ===
print("Loading job titles CSV...")
df = pd.read_csv(CSV_PATH)

# Assuming CSV has columns like 'title' and 'category'
if 'title' not in df.columns:
    raise ValueError("CSV must have a 'title' column")
if 'category' not in df.columns:
    raise ValueError("CSV must have a 'category' column")

job_titles = df['title'].tolist()
categories = df['category'].tolist()

print(f"Found {len(job_titles)} job titles in {len(set(categories))} categories")

# === ENCODE LABELS ===
print("Encoding labels...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(categories)

# Save label encoder classes
np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
print(f"Saved label encoder with {len(label_encoder.classes_)} classes")

# === GENERATE EMBEDDINGS ===
print("Generating embeddings with OpenAI API...")
embeddings = []

for i, title in enumerate(tqdm(job_titles, desc="Processing titles")):
    try:
        response = client.embeddings.create(
            input=title,
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
        
        # Rate limiting - sleep briefly between requests
        time.sleep(0.1)
        
    except Exception as e:
        print(f"Error processing title '{title}': {e}")
        # Use zero vector as fallback
        embeddings.append([0.0] * 1536)

# === SAVE DATA ===
embeddings_array = np.array(embeddings)
labels_array = np.array(encoded_labels)

print(f"Embeddings shape: {embeddings_array.shape}")
print(f"Labels shape: {labels_array.shape}")

np.save(EMBEDDING_PATH, embeddings_array)
np.save(LABELS_PATH, labels_array)

print("Preprocessing complete!")
print(f"Saved {len(embeddings)} embeddings to {EMBEDDING_PATH}")
print(f"Saved {len(labels_array)} labels to {LABELS_PATH}")
print(f"Categories: {list(label_encoder.classes_)}")