# emb-lens

埋め込みモデルの「意味の捉え方の癖」を目で確かめるツール。

同じ単語リストを複数の埋め込みモデルでエンコードし、
2語のベクトルを次元単位で突き合わせて可視化します。
ベンチマークの総合スコアではなく、**自分の語彙・ドメインでモデルがどう振る舞うか**を
体感してモデル選定するのが目的です。特に、**RAGで使う埋め込みモデルの選定**という
実用面を見据えており、単語同士の比較（本ツールの元々の可視化）に加えて、
自分の文書チャンクを使った検索精度チェック・概観マップ機能を用意しています
（下記「RAG検索精度チェック」「コーパス概観マップ」）。

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

## RAG検索精度チェック

RAGは「クエリ（質問文）と文書チャンクの非対称な検索」なので、単語同士の
対称比較用の`PREFIX`とは別に、非対称prefixマップ`RAG_PREFIX`を使います
（ruri-v3は`"検索クエリ: "`/`"検索文書: "`、mE5は`"query: "`/`"passage: "`）。

```bash
# POST /rank
{
  "query": "瑠璃色はどんな色？",
  "passages": ["瑠璃色（るりいろ）は...", "他のチャンク...", ...],
  "correct_index": 0   # 省略可。本来正解のチャンクのindex
}
# → モデルごとのランキング（index順・スコア）と、
#   correct_indexを指定した場合はその順位（correct_rank）を返す
```

`docs/index.html`の「RAG検索チェック」モードでは、チャンクを1件ずつ
「追加」フォームで入力し、どれが正解かをラジオボタンで指定できます。
検索結果はモデルごとに分けて表示され、正解チャンクが1位に来ていれば✓、
そうでなければ実際の順位と✗を表示します。少数（〜20件）のチャンクで
「このモデルは自分のドメインで正しく検索できるか」を素早く確認するのが目的です。

## コーパス概観マップ

多数（〜300件）の文書チャンクを一括貼り付けし、1モデルでエンコード・PCAして
全体のクラスター構造を地図で眺める機能です。実際のベクトルDBに接続するのでは
なく、チャンクをテキストとしてコピペする方式（認証・外部接続の実装は不要）。

```bash
# POST /corpus_map
{ "chunks": ["チャンク1...", "チャンク2...", ...], "model": "ruri-v3-130m" }
# → /encode と同じ words/models 形式で返す（words は "#1","#2",...という短いID、
#   本文の先頭60文字は追加フィールド preview に入る）
```

レスポンス形状を`/encode`と揃えているため、既存の点列・一致度・地図ビューが
そのまま使えます。件数が多いとチップボタン（語/比較の選択ボタン）が
使いにくくなるため、チップに表示するのは先頭`DISPLAY_LIMIT`（50）件のみに
絞っていますが、地図ビューは全件を描画し、タップで任意の点を選択できます
（チップに出ない分もタップで比較語に選べます）。

## 保留中の案

- **JMdict由来の単語プールでの近傍探索**（「猫」と入力したらその近傍語を
  辞書から自動抽出する機能）は検討したが、RAGモデル選定という本来の目的には
  直接応えないため保留。将来、汎用的な語彙探索用途で必要になったら再検討する

## Cloud Runへのデプロイ

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
