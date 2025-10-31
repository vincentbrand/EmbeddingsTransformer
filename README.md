# Embedding Transformer

A comprehensive machine learning project that transforms OpenAI embeddings for job title classification using a neural network approach with centroid-based learning. The project includes a complete ML pipeline with preprocessing, training, inference, and a production-ready REST API.

## What it does

This project takes job titles, converts them to embeddings using OpenAI's text-embedding-3-small model, and then transforms those embeddings into a task-specific vector space optimized for classification. The system learns to move embeddings closer to class centroids during training, enabling more accurate job title categorization.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Job Titles    │───▶│  OpenAI API      │───▶│  Embeddings     │
│   (CSV)         │    │  (1536D)         │    │  (1536D)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Categories     │◀───│  Transformer     │◀───│  Neural Network │
│  (Classification)│    │  Model (128D)    │    │  Training       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌──────────────────┐
                    │    FastAPI       │
                    │  REST Service    │
                    └──────────────────┘
```

## Key Components

### Core ML Pipeline
- **Preprocessing (`preprocess.py`)**: Converts job titles CSV to embeddings using OpenAI API and prepares training data
- **Training (`training.py`)**: Trains a neural network to transform 1536-dimensional OpenAI embeddings into a 128-dimensional task-specific space using centroid-based loss
- **Inference (`interference.py`)**: Loads the trained model and classifies new job titles by comparing transformed embeddings to category centroids

### API Service
- **FastAPI Service (`api/main.py`)**: Production-ready REST API that exposes the trained model for real-time job title classification and embedding transformation
- **Bruno API Tests (`bruno/`)**: Complete API testing suite with sample requests for all endpoints

### Data & Configuration
- **Job Titles Dataset (`job_titles.csv`)**: Training data with job titles and categories
- **Environment Configuration (`.env`)**: Secure OpenAI API key management
- **Docker Setup**: Multi-service containerization for ML pipeline and API

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

The Docker setup supports four services: preprocessing, training, inference, and API.

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

4. **API Service**: Start the FastAPI web service:
```bash
docker-compose up api
```

#### Quick Start - API Only

If you already have trained model files and just want to run the API:
```bash
docker-compose up api-only
```

#### FastAPI Service

The API service provides a REST interface for real-time job title classification:

- **Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Health Check**: `GET /health`

**Available Endpoints:**

1. **Hello World**
   - `GET /` - Returns welcome message
   ```json
   {"message": "Hello World"}
   ```

2. **Job Title Classification**
   - `POST /classify` - Classify a job title using the trained model
   ```json
   // Request
   {"job_title": "Senior Software Engineer"}
   
   // Response
   {
     "job_title": "Senior Software Engineer",
     "predicted_category": "Technology",
     "confidence_score": 0.95
   }
   ```

3. **Embedding Transformation**
   - `POST /transform` - Transform a 1536D embedding to 128D task-specific space
   ```json
   // Request
   {"embedding": [0.1, 0.2, ..., 0.3]} // 1536 values
   
   // Response
   {
     "original_embedding": [0.1, 0.2, ..., 0.3],
     "transformed_embedding": [0.5, 0.8, ..., 0.1] // 128 values
   }
   ```

4. **Health Check**
   - `GET /health` - Service health and model status
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "categories": 5
   }
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

5. Run API service:
```bash
docker build -f api/docker/Dockerfile -t embedding-api ./api
docker run -v $(pwd)/api:/app --env-file api/.env -p 8000:8000 embedding-api
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

## API Testing with Bruno

The project includes a complete Bruno API testing suite in the `bruno/` folder:

- **`Hello World.bru`** - Test the welcome endpoint
- **`Classify Job Title.bru`** - Test job classification with "Senior Software Engineer"
- **`Transform Embedding.bru`** - Test embedding transformation with 1536D dummy data
- **`Health Check.bru`** - Test service health status

To use:
1. Install [Bruno API Client](https://usebruno.com/)
2. Open Bruno and import the collection from the `bruno/` folder
3. Start the API service (`docker-compose up api-only`)
4. Run the requests to test all endpoints

## Technical Architecture

- **Input**: 1536-dimensional OpenAI embeddings (text-embedding-3-small)
- **Network**: Linear layers (1536 → 512 → 128) with ReLU activation
- **Training**: MSE loss against class centroids in transformed space
- **Classification**: Cosine similarity to category centroids
- **API**: FastAPI with Pydantic validation and automatic OpenAPI docs

## Project Structure

```
├── app/                    # ML Pipeline
│   ├── preprocess.py      # Data preprocessing
│   ├── training.py        # Model training
│   ├── interference.py    # Batch inference
│   ├── job_titles.csv     # Training data
│   └── requirements.txt   # Python dependencies
├── api/                   # REST API Service
│   ├── main.py           # FastAPI application
│   ├── docker/           # API Docker config
│   ├── requirements.txt  # API dependencies
│   └── .env             # API environment variables
├── bruno/                # API Testing Suite
│   ├── *.bru            # Bruno request files
│   └── bruno.json       # Collection config
├── docker/               # Main Docker config
│   ├── Dockerfile       # ML pipeline container
│   └── entrypoint.sh    # Multi-mode entrypoint
├── docker-compose.yml    # Multi-service orchestration
└── .gitignore           # Git ignore rules
```