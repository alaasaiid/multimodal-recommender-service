# Real-Time Multimodal Search & Two-Stage Re-Ranking Microservice

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Re--Ranking-2E8B57?style=flat)](https://lightgbm.readthedocs.io/)

A production-grade, low-latency recommendation microservice built with a **Two-Stage Candidate Retrieval & Re-Ranking Architecture**. Designed to handle large catalog scale by pairing dense vector semantic search with a gradient-boosted decision tree for real-time personalization.

---

## 🏛️ System Architecture

```text
                                 [ Client Request ]
                                         │
                                         ▼
                         [ FastAPI REST Endpoint (/recommend) ]
                                         │
         ┌───────────────────────────────┴───────────────────────────────┐
         │                                                               │
         ▼                                                               ▼
[ Stage 1: Candidate Retrieval ]                             [ Stage 2: ML Re-Ranking ]
• Sentence-Transformers (384-d)                              • LightGBM Decision Trees
• Normalized Cosine / Dot Product Similarity                 • Zero-Leakage Behavioral Features
• Fast Catalog Filtering (100k -> 50 Items)                   • Probability Score Normalization
         │                                                               │
         └───────────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                             [ Top-K Ranked Response ]
                                 (Latency: ~12ms)
