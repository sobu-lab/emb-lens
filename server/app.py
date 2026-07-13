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

# RAG用の非対称prefix（クエリ→文書検索）。上のPREFIX（/encode専用、対称比較用）とは別物。
# ruri-v3: 1+3 prefixスキームの検索クエリ/検索文書ペア
# (https://huggingface.co/cl-nagoya/ruri-v3-130m)
# mE5: 非対称タスク（passage retrieval）向けのquery/passageペア
# (https://huggingface.co/intfloat/multilingual-e5-small)
RAG_PREFIX = {
    "ruri-v3-130m": {"query": "検索クエリ: ", "passage": "検索文書: "},
    "mE5-small": {"query": "query: ", "passage": "passage: "},
}

MAX_WORDS = 64
MAX_PASSAGES = 20
MAX_CORPUS = 300

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


class RankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    passages: list[str] = Field(..., min_length=1, max_length=MAX_PASSAGES)
    correct_index: int | None = None


class CorpusRequest(BaseModel):
    chunks: list[str] = Field(..., min_length=2, max_length=MAX_CORPUS)
    model: str


@app.on_event("startup")
def warm_up():
    for name, repo in MODELS:
        get_model(name, repo)


@app.get("/health")
def health():
    return {"status": "ok", "models": [name for name, _ in MODELS]}


def _encode_words(words: list[str]) -> dict:
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


@app.post("/encode")
def encode(req: EncodeRequest):
    words = [w.strip() for w in req.words if w.strip()]
    if len(words) < 2:
        raise HTTPException(status_code=400, detail="words には2語以上を指定してください")
    if len(set(words)) != len(words):
        raise HTTPException(status_code=400, detail="words に重複があります")
    return _encode_words(words)


@app.post("/rank")
def rank(req: RankRequest):
    """RAG検索精度チェック: クエリに対して文書チャンクをランキングする（非対称prefix）。"""
    query = req.query.strip()
    passages = [p.strip() for p in req.passages if p.strip()]
    if not query:
        raise HTTPException(status_code=400, detail="query を指定してください")
    if not passages:
        raise HTTPException(status_code=400, detail="passages を1件以上指定してください")
    if req.correct_index is not None and not (0 <= req.correct_index < len(passages)):
        raise HTTPException(status_code=400, detail="correct_index が範囲外です")

    out: dict = {
        "query": query,
        "passages": passages,
        "correct_index": req.correct_index,
        "models": {},
    }
    for name, repo in MODELS:
        model = get_model(name, repo)
        px = RAG_PREFIX.get(name, {"query": "", "passage": ""})
        qvec = np.asarray(
            model.encode([px["query"] + query], normalize_embeddings=True)[0],
            dtype=np.float64,
        )
        P = np.asarray(
            model.encode([px["passage"] + p for p in passages], normalize_embeddings=True),
            dtype=np.float64,
        )
        scores = P @ qvec
        order = np.argsort(-scores)
        ranking = [{"index": int(i), "score": round(float(scores[i]), 4)} for i in order]
        correct_rank = None
        if req.correct_index is not None:
            correct_rank = int(np.where(order == req.correct_index)[0][0]) + 1
        out["models"][name] = {"ranking": ranking, "correct_rank": correct_rank}
    return out


@app.post("/corpus_map")
def corpus_map(req: CorpusRequest):
    """コーパス概観マップ: 多数の文書チャンクを1モデルでエンコード・PCAし、
    /encode と同じDATA形状（words/models）で返す（既存の描画コードを再利用するため）。"""
    chunks = [c.strip() for c in req.chunks if c.strip()]
    if len(chunks) < 2:
        raise HTTPException(status_code=400, detail="chunks を2件以上指定してください")
    valid_names = {n for n, _ in MODELS}
    if req.model not in valid_names:
        raise HTTPException(
            status_code=400, detail=f"model は {sorted(valid_names)} のいずれかを指定してください"
        )
    repo = dict(MODELS)[req.model]
    model = get_model(req.model, repo)
    px = RAG_PREFIX.get(req.model, {}).get("passage", "")
    ids = [f"#{i + 1}" for i in range(len(chunks))]

    texts = [px + c for c in chunks]
    X = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float64)
    P2 = pca2d(X)

    return {
        "words": ids,
        "preview": {ids[i]: chunks[i][:60] for i in range(len(chunks))},
        "models": {
            req.model: {
                "dims": int(X.shape[1]),
                "vectors": {ids[i]: [round(float(v), 4) for v in X[i]] for i in range(len(chunks))},
                "pca": {
                    ids[i]: [round(float(P2[i, 0]), 4), round(float(P2[i, 1]), 4)]
                    for i in range(len(chunks))
                },
                "maxabs": round(float(np.abs(X).max()), 4),
            }
        },
    }
