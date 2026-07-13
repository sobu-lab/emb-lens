"""
emb-lens API サーバー
=====================
extract.py と同じロジック（複数モデルでエンコード + PCA）を、
任意の単語リストに対してその場で実行し、viewer_template.html が
期待する DATA 形式（{"words": [...], "models": {...}}）で返す。
"""
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---- 比較するモデル（表示名, HuggingFace ID）----
MODELS = [
    ("ruri-v3-130m", "cl-nagoya/ruri-v3-130m"),
    ("mE5-small", "intfloat/multilingual-e5-small"),
]

# モデル別のプレフィックス
# ruri-v3は1+3 prefixスキームを採用（https://huggingface.co/cl-nagoya/ruri-v3-130m）。
# 単語=トピックとして比較するため "トピック: "（分類・クラスタリング・トピック情報用）を使用
# mE5はsemantic similarityのような対称タスクでは両側に "query: " を使う仕様
# (https://huggingface.co/intfloat/multilingual-e5-small)
PREFIX = {
    "ruri-v3-130m": "トピック: ",
    "mE5-small": "query: ",
}

MAX_WORDS = 64

CORS_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o]

app = FastAPI(title="emb-lens API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_loaded_models: dict[str, SentenceTransformer] = {}


def get_model(name: str, repo: str) -> SentenceTransformer:
    if name not in _loaded_models:
        _loaded_models[name] = SentenceTransformer(repo)
    return _loaded_models[name]


def pca2d(X: np.ndarray) -> np.ndarray:
    """extract.py と同じ決定的な PCA。"""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P2 = Xc @ Vt[:2].T
    return P2 / np.abs(P2).max() * 0.9


class EncodeRequest(BaseModel):
    words: list[str] = Field(..., min_length=2, max_length=MAX_WORDS)


@app.on_event("startup")
def warm_up():
    for name, repo in MODELS:
        get_model(name, repo)


@app.get("/health")
def health():
    return {"status": "ok", "models": [name for name, _ in MODELS]}


@app.post("/encode")
def encode(req: EncodeRequest):
    words = [w.strip() for w in req.words if w.strip()]
    if len(words) < 2:
        raise HTTPException(status_code=400, detail="words には2語以上を指定してください")
    if len(set(words)) != len(words):
        raise HTTPException(status_code=400, detail="words に重複があります")

    out: dict = {"words": words, "models": {}}
    for name, repo in MODELS:
        model = get_model(name, repo)
        texts = [PREFIX.get(name, "") + w for w in words]
        X = model.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float64)
        P2 = pca2d(X)
        out["models"][name] = {
            "dims": int(X.shape[1]),
            "vectors": {w: [round(float(v), 4) for v in X[i]] for i, w in enumerate(words)},
            "pca": {w: [round(float(P2[i, 0]), 4), round(float(P2[i, 1]), 4)] for i, w in enumerate(words)},
            "maxabs": round(float(np.abs(X).max()), 4),
        }
    return out
