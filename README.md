# Real-Time Multimodal Search & Two-Stage Re-Ranking Microservice

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Re--Ranking-2E8B57?style=flat)](https://lightgbm.readthedocs.io/)

A production-grade, low-latency recommendation microservice built on a **two-stage candidate retrieval and re-ranking architecture**. It pairs dense-vector semantic search with a gradient-boosted decision tree so personalization stays fast at large catalog scale.

| | |
|---|---|
| **Median latency** | ~12 ms end to end |
| **Catalog scale** | ~100k items reduced to 50 candidates before ranking |
| **Retrieval** | `all-MiniLM-L6-v2`, 384-d, L2-normalized dot product |
| **Ranking** | LightGBM, class-balanced, leakage-free temporal features |

---

## 🏛️ System Architecture

```text
                         ┌───────────────────────┐
                         │     Client Request    │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  FastAPI  /recommend  │
                         └───────────┬───────────┘
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │  STAGE 1  ·  Candidate Retrieval                                  │
   │                                                                   │
   │  Sentence-Transformers (all-MiniLM-L6-v2) -> 384-d dense vectors  │
   │  L2-normalized dot product across the catalog index               │
   │  ~100k catalog items narrowed to a 50-item candidate set          │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │  STAGE 2  ·  ML Re-Ranking                                        │
   │                                                                   │
   │  LightGBM gradient-boosted decision trees                         │
   │  Zero-leakage behavioral features (views, category affinity)      │
   │  Probability scores normalized to a comparable 0-1 range          │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  Top-K JSON response  │
                         │     (~12 ms p50)      │
                         └───────────────────────┘
```

---

## 🚀 Key Engineering Features

* **Zero-leakage data pipeline.** Temporal interaction features (`cumulative_user_views`, `category_affinity`) are computed strictly from events that precede the target event, so no future signal bleeds into training.
* **Vector semantic search.** Unstructured item descriptions are embedded into 384-dimensional dense vectors with `all-MiniLM-L6-v2` and L2-normalized, which turns cosine similarity into a plain dot product.
* **Tabular probability scoring.** LightGBM is trained with `class_weight='balanced'` to absorb the severe class imbalance of implicit feedback, where views vastly outnumber purchases.
* **Containerized microservice.** The whole service ships in a single Docker image. Models load once inside an async lifespan context, so no request pays the cold-start model load cost.

---

## 🛠️ Quickstart

### Option A: Docker (recommended)

```bash
# Build the image
docker build -t recommender-microservice .

# Run it on port 8000
docker run -p 8000:8000 recommender-microservice
```

### Option B: Local Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Once the service is up:

* Interactive docs: <http://localhost:8000/docs>
* OpenAPI schema: <http://localhost:8000/openapi.json>

---

## 📡 API Reference

### `POST /api/v1/recommend`

Returns the top-K re-ranked items for a user and a free-text query.

**Request body**

```json
{
  "user_id": "USR_1042",
  "query_text": "lightweight breathable summer wear",
  "user_prev_interactions": 14,
  "top_k": 3
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `user_id` | string | yes | Identifier used to look up behavioral features. |
| `query_text` | string | yes | Free-text intent, embedded at request time. |
| `user_prev_interactions` | integer | no | Prior interaction count fed to the ranker. Defaults to 0. |
| `top_k` | integer | no | Number of items to return. Defaults to 10. |

**Example request**

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR_1042",
    "query_text": "lightweight breathable summer wear",
    "user_prev_interactions": 14,
    "top_k": 3
  }'
```

**Example response**

```json
{
  "user_id": "USR_1042",
  "latency_ms": 12.4,
  "results": [
    {
      "item_id": "ITM_8831",
      "title": "Linen Short-Sleeve Shirt",
      "retrieval_score": 0.83,
      "rerank_score": 0.91
    },
    {
      "item_id": "ITM_4417",
      "title": "Perforated Running Tee",
      "retrieval_score": 0.79,
      "rerank_score": 0.86
    },
    {
      "item_id": "ITM_2290",
      "title": "Cotton Gauze Summer Dress",
      "retrieval_score": 0.81,
      "rerank_score": 0.74
    }
  ]
}
```

`retrieval_score` is the Stage 1 similarity, `rerank_score` is the Stage 2 probability. Results are ordered by `rerank_score`, which is why the raw similarity ordering does not always survive into the final list.

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.110, Uvicorn |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Ranking | LightGBM |
| Data | pandas, NumPy |
| Packaging | Docker, Python 3.10 |
