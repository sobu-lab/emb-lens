# emb-lens

埋め込みモデルの「意味の捉え方の癖」を目で確かめるローカルツール。

同じ単語リストを複数の埋め込みモデルでエンコードし、
2語のベクトルを次元単位で突き合わせて可視化します。
ベンチマークの総合スコアではなく、**自分の語彙・ドメインでモデルがどう振る舞うか**を
体感してモデル選定するのが目的です。

## 使い方

```bash
pip install -r requirements.txt
python extract.py
# → viewer.html が生成される。ブラウザでそのまま開く（サーバー不要）
```

- 比較するモデルは `extract.py` の `MODELS` を編集（HuggingFace ID を指定）
- 単語リストは `WORDS` を編集
- prefixを前提とするモデルは `PREFIX` に設定。モデルごとに適切なprefixは異なり、
  モデルカードで確認するのが確実（例: mE5系は対称タスク（意味的類似度）では
  `"query: "` を両側に使う仕様。ruri-v3は1+3 prefixスキームを持ち、
  単語単位の比較・分類・クラスタリング用途には `"トピック: "` が近い）

動作イメージは `viewer_sample.html` をブラウザで開くと確認できます
（spaCy ja 300次元と、その128次元切り詰め版の2モデル入りデモ）。

## ビュー

| ビュー | 内容 |
|---|---|
| 点列 | 1語のベクトルをそのまま走査（1次元=1点）。比較語を重ねて次元単位で突き合わせ |
| 一致度 | 各次元での2語の値を散布図に（x=語A, y=語B）。一致線に沿うほど近い語 |
| 地図 | 全語をPCAで2次元に配置。タップで比較語を選択 |

点列ビューの下段には各次元の一致度ストリップが列を揃えて表示され、
2語全体の類似度（コサイン類似度）をθに換算した参考線と見比べられます。

点列・一致度ビューの表示スケールは、外れ値次元（rogue dimension）に
引っ張られないよう上位1%の外れ値を除いた値を基準にしています。
その外れ値自体は▲▼（点列）や白い輪（一致度）で表示範囲外にあることを示します。

## 設計メモ

- **次元同士に連続性はない**（並び順は任意）ため、点を線でつなぐ表現は採用していません
- コサイン類似度が次元ごとに正確に分解できるのは「積 aₖ×bₖ」であり、
  一致度ストリップは読みやすさ優先の指標です（合計してもθにはなりません）
- 個々の次元の「意味」の解釈は目的にしていません。意味は複数次元にまたがる
  方向として分散しており、言語化を経由せずパターンとして見ることを狙っています
- これらの埋め込みモデルは基本的に「文」を単位に学習されており、単語1つを
  そのまま入れるのは学習分布から外れた特殊ケースです。そのため異方性
  （anisotropy: どの単語同士でもコサイン類似度が高めに出る）や
  rogue dimension（一部次元だけ全単語で異常に大きい値を持つ現象）の影響を受けやすく、
  類似度の絶対値が実際の意味的な近さより高く出がちです。絶対値ではなく
  レンジ（近いペアと遠いペアの差）で見る、という上記の指針はこれが理由です

## APIサーバー（任意の単語を動的にエンコード）

事前生成した静的ページではなく、ブラウザから任意の単語を送って
その場でエンコードしたい場合は `server/app.py`（FastAPI）を使います。

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --reload
# POST /encode  { "words": ["猫", "犬", ...] }
# → viewer_template.html の DATA と同じ形式のJSONを返す
```

### Cloud Runへのデプロイ

`Dockerfile` はビルド時にモデル重みをイメージへ焼き込むため、
起動時のダウンロードが発生しません。`main` への push で
`.github/workflows/deploy.yml` が Artifact Registry へのビルド・push と
Cloud Run へのデプロイを行います（Workload Identity Federationで認証）。

- リポジトリ変数 `CORS_ORIGINS`（Settings → Secrets and variables →
  Actions → Variables）に、下記のGitHub PagesのURLを設定してください
  （例: `https://sobu-lab.github.io`）。未設定の場合は全オリジンを許可します
- 必要なSecrets: `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`
  （Organization Secretとして他リポジトリと共有、またはこのリポジトリに個別登録）

## GitHub Pages（みんなで見られるビューア）

公開URL: https://sobu-lab.github.io/emb-lens/

`docs/index.html` が単語入力付きのビューアです。単語を送信すると
上記のCloud Run APIを呼び出してその場でエンコードします。

- 初回のみ Settings → Pages → Source を「GitHub Actions」に設定してください
- `main` への push（`docs/**` 変更時）で `.github/workflows/pages.yml` が自動デプロイします
- リポジトリ変数 `API_BASE_URL` に、Cloud RunデプロイでできたサービスURL
  （例: `https://emb-lens-xxxxx-an.a.run.app`）を設定してください。
  ビルド時に `docs/index.html` 内の `__API_BASE__` をこの値に置換します
- 公開後のURLは `https://<org>.github.io/emb-lens/` になります
  （このURLを `CORS_ORIGINS` に設定してください）

## 構成

```
extract.py             # モデルでエンコードして viewer.html を生成（静的・事前生成用）
viewer_template.html   # ビューアのテンプレート（次元数可変）
viewer_sample.html     # 事前生成済みデモ
server/app.py          # 任意の単語を動的にエンコードするFastAPI
Dockerfile             # server/app.py 用（Cloud Run向け、モデルをビルド時に焼き込み）
docs/index.html         # GitHub Pages公開用（単語入力→Cloud Run API呼び出し）
.github/workflows/deploy.yml  # main push で Cloud Run へ自動デプロイ
.github/workflows/pages.yml   # main push で GitHub Pages へ自動デプロイ
```

## License

MIT
