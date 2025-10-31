# Embedding Transformer

A machine learning project that transforms OpenAI embeddings for job title classification using a neural network approach with centroid-based learning.

## What it does

This project takes job titles, converts them to embeddings using OpenAI's text-embedding-3-small model, and then transforms those embeddings into a task-specific vector space optimized for classification. The system learns to move embeddings closer to class centroids during training, enabling more accurate job title categorization.

### Key Components

- **Training (`training.py`)**: Trains a neural network to transform 1536-dimensional OpenAI embeddings into a 128-dimensional task-specific space using centroid-based loss
- **Inference (`interference.py`)**: Loads the trained model and classifies new job titles by comparing transformed embeddings to category centroids
- **Data**: Job titles dataset (`job_titles.csv`) used for training and evaluation

## How to Use

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key

### Configuration

1. Set your OpenAI API key in the `.env` file:
```bash
OPENAI_API_KEY=your-actual-api-key-here
```

### Local Development

1. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```

2. Set environment variable:
```bash
export OPENAI_API_KEY=your-actual-api-key-here
```

3. Generate embeddings from CSV data:
```bash
python preprocess.py
```

4. Train the model:
```bash
python training.py
```

5. Run inference:
```bash
python interference.py
```

### Docker Setup

The Docker setup supports three modes: preprocessing, training, and inference.

#### Complete Workflow

1. **Preprocessing**: Generate embeddings from your job titles CSV:
```bash
docker-compose up --build preprocess
```

2. **Training**: Train the model using the generated embeddings:
```bash
docker-compose up training
```

3. **Inference**: Run inference with the trained model:
```bash
docker-compose up inference
```

#### Manual Docker Build

1. Build the image:
```bash
docker build -f docker/Dockerfile -t embedding-transformer .
```

2. Run preprocessing:
```bash
docker run -v $(pwd)/app:/app --env-file .env -e MODE=preprocess embedding-transformer
```

3. Run training:
```bash
docker run -v $(pwd)/app:/app --env-file .env -e MODE=training embedding-transformer
```

4. Run inference:
```bash
docker run -v $(pwd)/app:/app --env-file .env -e MODE=inference -it embedding-transformer
```

## Data Requirements

The preprocessing step expects a `job_titles.csv` file in the `app/` directory with the following columns:
- `title`: Job title text
- `category`: Job category/classification

Example CSV format:
```
title,category
Software Engineer,Technology
Data Scientist,Technology
Marketing Manager,Marketing
Sales Representative,Sales
```

## Architecture

- **Input**: 1536-dimensional OpenAI embeddings
- **Network**: Linear layers (1536 → 512 → 128) with ReLU activation
- **Training**: MSE loss against class centroids in transformed space
- **Classification**: Cosine similarity to category centroids