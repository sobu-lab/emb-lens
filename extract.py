#!/usr/bin/env python3
"""
埋め込みモデル比較ツール — 抽出スクリプト
========================================
複数の埋め込みモデルで同じ単語リストをエンコードし、
自己完結型のビューア (viewer.html) を生成します。

使い方:
    pip install sentence-transformers
    python extract.py
    → viewer.html をブラウザで開くだけ（サーバー不要）

モデルと単語は下の MODELS / WORDS を書き換えてください。
初回はモデルのダウンロードが走ります（HuggingFaceから）。
"""
import json
import numpy as np
from pathlib import Path

# ---- 比較したいモデル（表示名, HuggingFace ID）----
MODELS = [
    ("ruri-v3-130m", "cl-nagoya/ruri-v3-130m"),
    ("mE5-small",    "intfloat/multilingual-e5-small"),
    # ("ruri-v3-310m", "cl-nagoya/ruri-v3-310m"),
    # ("sbert-ja",     "sonoisa/sentence-bert-base-ja-mean-tokens-v2"),
]

# ---- 比較したい単語 ----
WORDS = [
    "猫", "犬", "鳥", "魚", "馬",
    "東京", "大阪", "名古屋", "京都", "パリ",
    "林檎", "蜜柑", "バナナ", "寿司", "ラーメン", "カレー",
    "車", "電車", "飛行機", "自転車",
    "愛", "怒り", "喜び", "悲しみ",
    "数学", "物理", "音楽", "絵画",
    "パソコン", "インターネット", "ロボット", "人工知能",
]

# mE5はsemantic similarityのような対称タスクでは両側に "query: " を使う仕様
# (https://huggingface.co/intfloat/multilingual-e5-small)
PREFIX = {
    "mE5-small": "query: ",
}


def pca2d(X: np.ndarray) -> np.ndarray:
    """300次元→2次元。決定的なPCA（proximaと同じ思想）。"""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P2 = Xc @ Vt[:2].T
    return P2 / np.abs(P2).max() * 0.9


def main():
    from sentence_transformers import SentenceTransformer

    out = {"words": WORDS, "models": {}}
    for name, repo in MODELS:
        print(f"[{name}] loading {repo} ...")
        model = SentenceTransformer(repo)
        texts = [PREFIX.get(name, "") + w for w in WORDS]
        X = model.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float64)
        P2 = pca2d(X)
        out["models"][name] = {
            "dims": int(X.shape[1]),
            "vectors": {w: [round(float(v), 4) for v in X[i]] for i, w in enumerate(WORDS)},
            "pca": {w: [round(float(P2[i, 0]), 4), round(float(P2[i, 1]), 4)] for i, w in enumerate(WORDS)},
            "maxabs": round(float(np.abs(X).max()), 4),
        }
        print(f"[{name}] dims={X.shape[1]}  例: 猫×犬 = {float(X[0] @ X[1]):.3f}")

    data = json.dumps(out, ensure_ascii=False)
    tpl = Path(__file__).with_name("viewer_template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", data)
    dst = Path(__file__).with_name("viewer.html")
    dst.write_text(html, encoding="utf-8")
    print(f"\n書き出しました → {dst}  ({len(html)//1024} KB)")
    print("ブラウザでそのまま開けます。")


if __name__ == "__main__":
    main()
