from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from openai import OpenAI
import os

app = FastAPI(title="Embedding Transformer API", version="1.0.0")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Model configuration
MODEL_PATH = "embedding_transformer.pt"
LABEL_ENCODER_PATH = "label_encoder.npy"
EMBEDDING_MODEL = "text-embedding-3-small"

# Model class
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

# Load model and data
model = EmbeddingTransformer(1536, 128)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

# Load centroids
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

# Pydantic models
class HelloWorldResponse(BaseModel):
    message: str

class ClassificationRequest(BaseModel):
    job_title: str

class ClassificationResponse(BaseModel):
    job_title: str
    predicted_category: str
    confidence_score: float

class TransformRequest(BaseModel):
    embedding: list[float]

class TransformResponse(BaseModel):
    original_embedding: list[float]
    transformed_embedding: list[float]

# Helper functions
def get_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def get_transformed_embedding(title: str):
    raw_embedding = get_embedding(title)
    x = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        transformed = model(x)
    return transformed.squeeze(0).numpy()

def classify_by_similarity(transformed_embedding, centroids):
    sims = [np.dot(transformed_embedding, c) / (np.linalg.norm(transformed_embedding) * np.linalg.norm(c)) for c in centroids]
    best_idx = np.argmax(sims)
    confidence = float(sims[best_idx])
    return label_classes[best_idx], confidence

def transform_embedding_only(embedding_list):
    """Transform an embedding using the trained model"""
    x = torch.tensor(embedding_list, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        transformed = model(x)
    return transformed.squeeze(0).numpy()

# Routes
@app.get("/", response_model=HelloWorldResponse)
async def hello_world():
    """Hello World endpoint"""
    return HelloWorldResponse(message="Hello World")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_job_title(request: ClassificationRequest):
    """Classify a job title using the trained embedding transformer model"""
    try:
        transformed = get_transformed_embedding(request.job_title)
        predicted_category, confidence = classify_by_similarity(transformed, centroids)
        
        return ClassificationResponse(
            job_title=request.job_title,
            predicted_category=predicted_category,
            confidence_score=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/transform", response_model=TransformResponse)
async def transform_embedding(request: TransformRequest):
    """Transform an embedding using the trained model"""
    try:
        # Validate embedding dimensions
        if len(request.embedding) != 1536:
            raise HTTPException(status_code=400, detail=f"Embedding must be 1536-dimensional, got {len(request.embedding)}")
        
        # Transform the embedding
        transformed = transform_embedding_only(request.embedding)
        
        return TransformResponse(
            original_embedding=request.embedding,
            transformed_embedding=transformed.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True, "categories": len(label_classes)}