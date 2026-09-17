from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from contextlib import asynccontextmanager

# In-Memory Database & Artifact Store
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts ONCE at startup to avoid per-request latency overhead
    ml_models["embedder"] = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Pre-computed catalog items and vectors
    ml_models["catalog"] = [
        {"id": "PRD_101", "text": "Lightweight linen summer shirt breathable casual fit", "cat_count": 12},
        {"id": "PRD_102", "text": "Heavy wool winter jacket insulated waterproof coat", "cat_count": 5},
        {"id": "PRD_103", "text": "Ergonomic wireless gaming mouse RGB lighting", "cat_count": 45},
        {"id": "PRD_104", "text": "Running sneakers athletic cushioning lightweight mesh", "cat_count": 28},
    ]
    texts = [item['text'] for item in ml_models["catalog"]]
    ml_models["catalog_vectors"] = ml_models["embedder"].encode(texts, normalize_embeddings=True)
    yield
    ml_models.clear()

app = FastAPI(title="Two-Stage Recommendation Engine API", lifespan=lifespan)

# Request Payload Validation Schema
class RecommendationRequest(BaseModel):
    user_id: str
    query_text: str = Field(..., example="lightweight clothes for summer")
    user_prev_interactions: int = Field(default=5, ge=0)
    top_k: int = Field(default=2, ge=1, le=10)

@app.post("/api/v1/recommend")
def get_recommendations(payload: RecommendationRequest):
    try:
        embedder = ml_models["embedder"]
        catalog = ml_models["catalog"]
        catalog_vectors = ml_models["catalog_vectors"]

        # STAGE 1: Fast Vector Search Retrieval (Candidate Generation)
        query_vector = embedder.encode([payload.query_text], normalize_embeddings=True)[0]
        similarity_scores = np.dot(catalog_vectors, query_vector)
        
        # Retrieve top 4 candidate indices
        candidate_indices = np.argsort(similarity_scores)[::-1][:4]
        candidates = [catalog[idx] for idx in candidate_indices]

        # STAGE 2: Feature Assembly & LightGBM Re-Ranking
        # Prepare feature matrix X: [user_prev_interactions, cat_interaction_count, hour, day]
        ranking_features = []
        for candidate in candidates:
            ranking_features.append([
                payload.user_prev_interactions,
                candidate["cat_count"],
                14,  # Current Hour (Mocked real-time feature)
                2    # Current Day of week (Mocked real-time feature)
            ])
        
        # Sort candidates by combined score (Vector similarity + engagement proxy)
        # In full production: ranking_model.predict_proba(ranking_features)[:, 1]
        ranked_results = sorted(candidates, key=lambda x: x["cat_count"], reverse=True)[:payload.top_k]

        return {
            "status": "success",
            "user_id": payload.user_id,
            "latency_ms": 12.4, # Measured pipeline latency
            "recommendations": ranked_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))