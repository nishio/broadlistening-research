# HDBSCANとUMAPのスケールミスマッチ実験

**日付**: 2025-10-28

## 概要

HDBSCANで抽出したクラスタをUMAPで可視化すると、クラスタが引き裂かれたりオーバーラップする現象の原因を調査した実験。

**主要な発見**: UMAPとHDBSCANは異なるスケールの構造を見ており、UMAPのスケール（15近傍）では既にクラスタが複雑に混在している。

詳細なレポートは [`notes/2025-10-28-01-hdbscan-umap-scale-mismatch.md`](../../notes/2025-10-28-01-hdbscan-umap-scale-mismatch.md) を参照。

## スクリプト

### 1. `analyze_neighborhood_sampled.py`

高次元空間での近傍構造の純度を分析。

**機能**:
- 5,000点をサンプリングしてHDBSCAN実行
- 各点の15近傍を高次元空間で計算
- 15近傍のうち同一クラスタ内の点の割合を分析
- クラスタ間の近傍リンクを検出
- 境界点（10%以上が他クラスタ）を特定

**実行方法**:
```bash
cd broadlistening-research/experiments/2025-10-28
broadlistening-research/venv/bin/python analyze_neighborhood_sampled.py
```

**出力**:
- `neighborhood_analysis_sampled.png`: 3つのグラフ
  - 近傍構造の純度ヒストグラム
  - クラスタサイズと境界点の関係
  - クラスタごとの近傍純度

**主要な結果**:
- 平均同一クラスタ割合: **71.6%**
- 境界点割合: **72.9%**

### 2. `analyze_hdbscan_scale.py`

複数のスケール（k=5-100）で近傍構造を分析し、HDBSCANのパラメータとの対応を調査。

**機能**:
- 複数のk値（5, 10, 15, 30, 50, 100）で近傍構造を分析
- HDBSCANのmin_samples、min_cluster_sizeとの比較
- Core distance（密度計算に使う距離）の分布を可視化

**実行方法**:
```bash
cd broadlistening-research/experiments/2025-10-28
broadlistening-research/venv/bin/python analyze_hdbscan_scale.py
```

**出力**:
- `hdbscan_scale_analysis.png`: 4つのグラフ
  - スケールごとのクラスタ純度
  - スケールごとの境界点割合
  - Core distanceの分布
  - k=15での近傍純度分布

**主要な結果**:
| k | 純度 | 意味 |
|---|------|------|
| 5 | 83.5% | HDBSCANの密度計算スケール |
| 15 | 71.6% | UMAPのスケール |
| 30 | 59.7% | HDBSCANのクラスタサイズスケール |

## 生成ファイル

```
experiments/2025-10-28/
├── README.md                           # このファイル
├── analyze_neighborhood_sampled.py     # 近傍構造分析スクリプト
├── analyze_hdbscan_scale.py            # スケール分析スクリプト
├── neighborhood_analysis_sampled.png   # 近傍構造分析結果
├── hdbscan_scale_analysis.png          # スケール分析結果
└── hdbscan_labels_cache.pkl            # HDBSCANラベルのキャッシュ
```

## データセット

- **ソース**: `dataset/team-mirai/embeddings.pkl`
- **サイズ**: 21,348点
- **次元**: 1,536次元
- **実験では5,000点にサンプリング**（計算時間削減のため）

## パラメータ

### HDBSCAN
```python
min_cluster_size = 30  # クラスタの最小サイズ
min_samples = 5        # 密度計算に使う近傍点数
metric = 'euclidean'
```

### UMAP（2025-10実験で使用）
```python
n_neighbors = 15   # 局所構造の範囲
min_dist = 0.1     # 最小距離
n_components = 2   # 出力次元
```

## 主要な発見

### 1. HDBSCANクラスタの近傍純度

高次元空間で各点の15近傍を分析した結果：
- **平均71.6%**のみが同一クラスタ内
- **72.9%が境界点**（15近傍の10%以上が他クラスタ）
- すべてのクラスタで60%以上が境界点

### 2. スケール依存の純度低下

近傍点数kを増やすと純度が急激に低下：
- k=5: 83.5%（HDBSCANの密度計算スケール）
- k=15: 71.6%（UMAPのスケール）
- k=30: 59.7%（HDBSCANのクラスタサイズスケール）

### 3. スケールミスマッチの証明

HDBSCANは多層的なスケールで動作：
1. **密度計算**: min_samples=5（純度83.5%）
2. **クラスタ抽出**: min_cluster_size=30（純度59.7%）

UMAPのn_neighbors=15はこの中間にあり、既にクラスタが混在している領域。

## 理論的示唆

### なぜクラスタが引き裂かれるのか

1. HDBSCANは5近傍の密度から、30点以上の大きな構造を抽出
2. UMAPは15近傍の関係を保存しようとする
3. 15近傍レベルでは既に他クラスタの点が混入（約28%）
4. UMAPは混入した関係も保存 → 2次元でクラスタがオーバーラップ

### UMAPは正しく動作している

- UMAPは高次元空間の15近傍構造を忠実に保存している
- 問題は、その近傍構造自体が複雑に混在していること
- → 可視化の歪みは、アルゴリズムの欠陥ではなく、**スケールの選択の問題**

## 改善策

### 1. Supervised UMAP（推奨）
```python
umap.UMAP(n_neighbors=15, target_weight=0.9).fit_transform(X, y=labels)
```
- クラスタラベルを教師信号として使用
- 近傍構造よりもクラスタの分離を優先

### 2. n_neighborsの調整
```python
# HDBSCANの密度計算スケールに合わせる
umap.UMAP(n_neighbors=5)

# クラスタの大域構造を保存
umap.UMAP(n_neighbors=30-50)
```

### 3. HDBSCANパラメータの調整
```python
# より厳密なクラスタ（ノイズが増える）
hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
```

## 関連実験

- **experiments/2025-10/**: 元の可視化実験
  - k-means、HDBSCAN、Supervised UMAPの比較
  - `hdbscan_umap_convexhull.png`: 問題の可視化
  - `hdbscan_supervised_umap_convexhull.png`: 改善版

## 参考文献

- HDBSCAN: Campello et al. (2013) "Density-Based Clustering Based on Hierarchical Density Estimates"
- UMAP: McInnes et al. (2018) "UMAP: Uniform Manifold Approximation and Projection"

## 次のステップ

1. 元データ（21,348点）での完全な分析
2. 異なるmin_samplesパラメータでの比較
3. Supervised UMAPとの定量的比較（シルエットスコア等）
4. t-SNE、PaCMAPとの比較実験
