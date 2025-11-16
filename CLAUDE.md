# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、ブロードリスニング（パブリックコメントのクラスタリング分析）の研究プロジェクトです。kouchou-ai (https://github.com/digitaldemocracy2030/kouchou-ai) のクラスタリングアルゴリズムを改善するための実験環境を提供しています。

**重要な特徴**:
- kouchou-aiの実装を変更せずに、異なるクラスタリングパラメータで実験可能
- 実験結果を中間データとして保存し、後続の分析に利用可能
- 複数のデータセット（aipubcom, bikeshed, team-mirai）を使用した研究

## セットアップ

### 環境構築

```bash
# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# 依存パッケージのインストール
pip install numpy pandas scikit-learn hdbscan umap-learn matplotlib openai
```

### データファイルの準備

詳細は[SETUP.md](SETUP.md)を参照してください。

必要なファイル:
1. `embeddings.pkl` - パブコメの埋め込みベクトル（形状: 9883 x 3072）
2. `args.csv` - パブコメのメタデータ

データセットは `dataset/` ディレクトリに配置されます。

### Git LFSの設定

大規模ファイル（embeddings.pkl）を扱うため、Git LFSを使用しています:

```bash
# Git LFSのインストール（macOS）
brew install git-lfs

# Git LFSの初期化
git lfs install
```

## 開発コマンド

### クラスタリング実験の実行

#### 単一パラメータセットでの実験

```bash
cd experiments/2025-03/clustering_param_experiment

python run_clustering_experiment.py \
  --embeddings ~/path/to/embeddings.pkl \
  --args ~/path/to/args.csv \
  --cluster-nums 3 6 12 24 \
  --output experiment_results
```

#### 複数パラメータセットでの実験

```bash
python run_multi_experiment.py \
  --config experiment_config.json \
  --embeddings ~/path/to/embeddings.pkl \
  --args ~/path/to/args.csv \
  --output experiments \
  --run-all \
  --generate-labels
```

### クラスタラベル生成

```bash
cd experiments/2025-02

python generate_cluster_labels.py \
  --cluster-file clustered_arguments.csv \
  --output-file cluster_labels.json
```

**注意**: GPT-4oのAPIキーが環境変数 `OPENAI_API_KEY` に設定されている必要があります。

### 実験結果の比較

```bash
cd experiments/2025-03/clustering_param_experiment

python compare_cluster_labels.py \
  --experiment-dirs experiments/default_20250327_123456 experiments/fine_grained_20250327_123456 \
  --output-report comparison_report.md \
  --output-csv comparison_data.csv
```

## アーキテクチャ

### ディレクトリ構造

```
broadlistening-research/
├── dataset/               # データセット
│   ├── aipubcom/         # AIと著作権のパブコメデータ（9,883件）
│   ├── bikeshed/         # テスト用データセット（165件）
│   └── team-mirai/       # 新しいデータセット
├── experiments/
│   ├── 2025-02/          # 初期実験（スクリプトが散在）
│   │   ├── core/         # 共通ユーティリティ（data_processing.py等）
│   │   ├── clustering/   # クラスタリング実験スクリプト
│   │   ├── results/      # 実験結果（JSON）
│   │   └── generate_cluster_labels.py  # ラベル生成（2段階アプローチ）
│   └── 2025-03/          # 整理された実験環境
│       └── clustering_param_experiment/
│           ├── run_clustering_experiment.py
│           ├── run_multi_experiment.py
│           ├── generate_cluster_labels_cli.py
│           ├── compare_cluster_labels.py
│           └── experiment_config.json
├── notes/                # 研究ノート（日付-連番-内容の命名規則）
└── publish/              # 公開用レポート
```

### kouchou-aiとの連携

このプロジェクトは、外部リポジトリ `kouchou-ai` のモジュールをインポートして使用します:

```python
# run_clustering_experiment.py内
KOUCHOU_PATH = os.path.expanduser("~/kouchou-ai/server/broadlistening")
sys.path.append(KOUCHOU_PATH)

from hierarchical_clustering import hierarchical_clustering_embeddings
from embedding import extract_embeddings_from_pkl
```

**前提条件**:
- kouchou-aiリポジトリが `~/kouchou-ai` にクローンされていること
- もし異なる場所にある場合は、スクリプト内の `KOUCHOU_PATH` を修正すること

### 実験ワークフロー

1. **クラスタリング実験** (`run_clustering_experiment.py`):
   - UMAPで次元削減（2次元）
   - kouchou-aiの階層的クラスタリングを実行
   - 複数のクラスタ数（例: 3, 6, 12, 24）で結果を生成
   - 結果を `clustering_result.csv` に保存

2. **ラベル生成** (`generate_cluster_labels.py`):
   - GPT-4oを使用した2段階アプローチ
   - クラスタ内外の意見を比較して特徴を抽出
   - 生成されたラベルの品質を検証・改善
   - ドメイン固有キーワード（AI、著作権等）を必須とする

3. **結果の比較** (`compare_cluster_labels.py`):
   - 異なるパラメータでの実験結果を比較
   - センチメント分布の分析
   - Markdownレポートの生成

### データ処理パイプライン

```
embeddings.pkl (9883 x 3072)
    ↓
UMAP次元削減 (9883 x 2)
    ↓
階層的クラスタリング (複数のクラスタ数)
    ↓
clustering_result.csv
    ↓
GPT-4oによるラベル生成
    ↓
cluster_labels.json
```

### 重要な実装詳細

#### メモリ管理

- データ読み込み時に最低2GBの空きメモリが必要
- `experiments/2025-02/core/data_processing.py` でメモリチェックを実施
- 大規模データセット処理時は `psutil` でメモリ使用量を監視

#### クラスタラベル生成の品質保証

`experiments/2025-02/generate_cluster_labels.py` の実装:

1. **基本検証**:
   - ラベル長（5-30文字）
   - ドメイン固有キーワードの有無
   - 一般的な表現の除外

2. **LLM検証**:
   - GPT-4oで品質評価（temperature=0.4）
   - 最大3回のリトライロジック
   - 失敗時は改善案を採用

3. **必須フィールド**:
   ```json
   {
     "label": "具体的なラベル",
     "description": "説明",
     "keywords": ["キーワード1", "キーワード2"],
     "sentiment": "positive/negative/neutral",
     "cluster_id": 0,
     "size": 100
   }
   ```

#### クラスタリングパラメータ実験

`experiment_config.json` で複数の実験パターンを定義:

```json
{
  "experiments": [
    {
      "name": "default",
      "description": "デフォルトのクラスタ数設定",
      "cluster_nums": [3, 6, 12, 24]
    }
  ]
}
```

## データセット

### aipubcom

- **内容**: AIと著作権に関するパブリックコメント
- **サイズ**: 9,883件
- **用途**: メイン研究データ

### bikeshed

- **内容**: 架空の「市庁舎の自転車置き場」に関する意見（AI生成）
- **サイズ**: 165件（うち80件は多数派工作シミュレーション）
- **用途**: 多数派工作に対するロバスト性テスト
- **重要な発見**: GPT-4.5による洗練された多数派工作は、既存のクラスタリング手法では検出困難

### team-mirai

- **内容**: 新しいデータセット
- **用途**: 追加実験

## プロジェクト進化の経緯

### 2025-02 フェーズ

- 初期の実験段階
- スクリプトと中間データが散在
- HDBSCANとk-meansの比較実験
- クラスタラベル生成手法の開発

### 2025-03 フェーズ

- 大幅な整理と構造化
- `experiments/2025-03/clustering_param_experiment/` に実験環境を統合
- kouchou-aiのコードを変更せずに実験可能な設計
- 複数パラメータセットでの一括実験サポート

### 2025-10-28 重要な発見

**クラスタリングと埋め込み空間の関係に関する重要な知見を発見**

#### 発見1: HDBSCANとUMAPのスケールミスマッチ

HDBSCANで抽出したクラスタをUMAPで可視化すると、クラスタが引き裂かれたりオーバーラップする現象の原因を解明。

**主要な発見**:
- HDBSCANのクラスタは高次元空間でも**k=15近傍レベルでは純度71.6%**（約28%が他クラスタ）
- UMAPのスケール（n_neighbors=15）では既にクラスタが複雑に混在
- HDBSCANは多層的なスケールで動作：
  - 密度計算: min_samples=5（純度83.5%）
  - クラスタ抽出: min_cluster_size=30（純度59.7%）
- **UMAPは正しく動作している**：高次元の近傍構造を忠実に保存
- 問題は、その近傍構造自体が既に混在していること

詳細: `notes/2025-10-28-01-hdbscan-umap-scale-mismatch.md`

#### 発見2: 意味ベース分類 vs 密度ベースクラスタリング

YouTubeコメント（1,276件）でJigsaw SenseMakerの分類を検証。

**驚くべき結果**:

| 分類手法 | k=15純度 | 分離度 |
|---------|---------|--------|
| HDBSCAN（密度ベース） | 71.6% | △ 中程度 |
| SenseMaker Topic（意味ベース、5カテゴリ） | 57.6% | ✗ 混在 |
| SenseMaker Subtopic（意味ベース、24カテゴリ） | 22.9% | ✗✗ 極めて混在 |

**意味ベースの分類よりも、密度ベースのHDBSCANの方が高次元空間で分離されている！**

**重要な示唆**:
- カテゴリの抽象度と純度の逆相関：具体的（「入湯税」77.5%）vs 抽象的（「SNS」36.1%）
- 24カテゴリは細かすぎる（境界点100%）
- 使用するembeddingモデルによって分類の純度が変わる
- 意味的分類（トップダウン）と密度ベース分類（ボトムアップ）は異なる目的

詳細: `notes/2025-10-28-02-youtube-comments-topic-clustering-purity.md`

#### 実用的な示唆

**UMAP可視化の解釈**:
- トピック/カテゴリでグルーピングした可視化は、高次元空間では既に混在している可能性
- クラスタが引き裂かれたりオーバーラップするのは「UMAPの問題」ではなく「高次元構造の反映」
- Supervised UMAPで強制的に分離すると、高次元構造が歪む

**クラスタリング手法の選択**:
- 意味ベース分類：解釈可能だが、embedding空間と乖離する可能性
- 密度ベース：embedding空間の実態を反映するが、解釈は事後的
- ハイブリッドアプローチ推奨：データドリブンで発見 → 人間が解釈

## 注意事項

1. **kouchou-aiの依存性**: このプロジェクトは外部リポジトリに依存しているため、kouchou-aiが正しくセットアップされていることを確認すること

2. **APIキーの管理**: クラスタラベル生成にはOpenAI APIキーが必要。環境変数または `.env` ファイルで管理すること

3. **実験結果の保存**: 実験結果はタイムスタンプ付きのディレクトリに保存されるため、ディスク容量に注意すること

4. **研究ノートの命名規則**: `notes/` ディレクトリのファイルは「日付-連番-内容」の形式で命名されている

5. **2025-02の実験コード**: 初期段階のコードは参考として残されているが、新しい実験には `2025-03/clustering_param_experiment/` を使用すること

6. **matplotlibでの日本語表示**: グラフに日本語を含める場合は、必ず日本語フォントを設定すること。設定しないと文字化けが発生する。以下のコードをmatplotlibのimport直後に追加すること：

   ```python
   import matplotlib.pyplot as plt
   import matplotlib

   # 日本語フォントの設定（macOS）
   matplotlib.rcParams['font.family'] = 'Hiragino Sans'
   # または
   # matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']

   # マイナス記号の文字化け対策
   matplotlib.rcParams['axes.unicode_minus'] = False
   ```

   **環境別のフォント例**:
   - macOS: `'Hiragino Sans'`, `'Hiragino Kaku Gothic Pro'`, `'Yu Gothic'`
   - Windows: `'Yu Gothic'`, `'MS Gothic'`, `'Meiryo'`
   - Linux: `'Takao'`, `'IPAexGothic'`, `'Noto Sans CJK JP'`

## クラスタ純度分析（2025-10-28新手法）

### 目的

クラスタリング結果が高次元空間でどの程度分離しているかを定量的に評価する手法。

### 手法

**基本的なアイデア**:
- 各点のk近傍（例：k=15）を高次元空間で計算
- k近傍のうち、同一クラスタ内の点の割合を計算
- この割合が高い → 高次元で分離、低い → 混在

**境界点の定義**:
- k近傍の10%以上が他クラスタに属する点
- 境界点割合が高い → クラスタ境界が曖昧

### 実装スクリプト

#### 1. `analyze_neighborhood_sampled.py` / `analyze_hdbscan_scale.py`

HDBSCANクラスタの純度分析。

```bash
cd experiments/2025-10-28
/path/to/venv/bin/python analyze_hdbscan_scale.py
```

**機能**:
- 複数のk値（5, 10, 15, 30, 50, 100）で純度を分析
- HDBSCANパラメータ（min_samples, min_cluster_size）との対応を確認
- Core distance（密度計算に使う距離）の分布を可視化

#### 2. `analyze_youtube_comments_cluster_purity.py`

Topic/Subtopicなどの既存分類の純度分析。

```bash
cd experiments/2025-10-28
/path/to/venv/bin/python analyze_youtube_comments_cluster_purity.py
```

**前提**:
- `youtube_comments_embeddings.pkl`が存在すること
- 事前に`analyze_youtube_comments.py`を実行してembeddingsを生成

**機能**:
- Topic/Subtopicごとの近傍純度を分析
- クラスタサイズと境界点の関係を可視化
- 複数スケールでの比較

### 分析結果の解釈

**純度の目安**:
- **90%以上**: 明確に分離（例：「入湯税」77.5%でも良好な部類）
- **70-90%**: 部分的に混在（例：HDBSCAN 71.6%）
- **50-70%**: 大きく混在（例：Topic全体 57.6%）
- **50%未満**: 極めて混在（例：Subtopic 22.9%）

**境界点割合の目安**:
- **50%未満**: 明確な境界
- **50-80%**: 曖昧な境界
- **80%以上**: 非常に曖昧（ほとんどの点が境界付近）
- **100%**: 全ての点が境界（Subtopicの例）

### 実験設計への応用

1. **クラスタリングパラメータの調整**:
   - min_cluster_sizeを大きくする → 純度向上、但しノイズ増加
   - min_samplesを調整 → 密度計算のスケールを変更

2. **カテゴリ設計の改善**:
   - 純度の低いカテゴリを統合
   - 抽象的なカテゴリを避け、具体的なカテゴリを使用
   - カテゴリ数を減らす

3. **可視化の解釈**:
   - 純度が低い分類でのUMAP可視化は慎重に解釈
   - Supervised UMAPは高次元構造を歪める可能性
   - 通常のUMAPで引き裂かれ・オーバーラップは正常な現象の可能性
