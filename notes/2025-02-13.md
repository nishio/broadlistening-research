# Broadlistening Research Notes 2025-02-13

## CSV Files Analysis

以下のCSVファイルが見つかりました：

### experiments/results/
1. kmeans_same_n_metrics_2d.csv
2. dataset_x_clusters.csv
3. hdbscan_cluster_labels.csv
4. kmeans_same_size_metrics_2d.csv
5. hdbscan_detailed_metrics_2d.csv

各ファイルについて詳細な分析を行います：

### 1. kmeans_same_n_metrics_2d.csv
- 生成日時: 2025-02-10
- 生成コミット: 49c9aee
- 生成元スクリプト: experiments/clustering_comparison.py
- データの意味: KMeansクラスタリングの評価指標（2次元データ）。HDBSCANと同じクラスタ数を使用した場合の比較実験結果。各クラスタの密度指標（平均距離、最大距離）とサイズを含む。シルエット係数による評価も実施。
- カラムの説明:
  - cluster_id: クラスタの一意な識別子
  - size: クラスタ内の要素数
  - avg_distance: クラスタ内の全ペア間の平均距離（密度の逆指標）
  - max_distance: クラスタ内の最大距離
  - density: 密度指標（平均距離の逆数として計算: 1.0 / (avg_distance + 1e-10)）
- 基本統計:
  - Shape: (676, 5) - 676クラスタ、5カラム
  - 平均クラスタサイズ: 14.62 (標準偏差: 5.83)
  - クラスタサイズ範囲: 1-42
  - 平均密度: 9.09 (標準偏差: 2.79)

### 2. dataset_x_clusters.csv
- 生成日時: 2025-02-10
- 生成コミット: 542be8f, f1e342c
- 生成元スクリプト: experiments/process_dataset_x.py
- データの意味: 密度上位66件のクラスタに関するデータ。サイズが5以上のクラスタから選出。各クラスタのID、サイズ、平均距離、最大距離、密度、データインデックス、および実際の議論（argument）を含む。
- カラムの説明:
  - cluster_id: クラスタの一意な識別子
  - size: クラスタ内の要素数
  - avg_distance: クラスタ内の平均距離
  - max_distance: クラスタ内の最大距離
  - density: クラスタの密度指標
  - data_index: 元データのインデックス（arg-idに対応）
  - argument: 実際の議論テキスト
- 基本統計:
  - Shape: (41, 7) - 41行、7カラム
  - クラスタ数: 10
  - 平均クラスタサイズ: 14.34 (標準偏差: 5.80)
  - クラスタサイズ範囲: 6-27
  - 平均密度: 1.60 (標準偏差: 0.27)

### 3. hdbscan_cluster_labels.csv
- 生成日時: 2025-02-10
- 生成コミット: 9699392
- 生成元スクリプト: experiments/process_hdbscan_clusters.py
- データの意味: HDBSCANアルゴリズムによって生成されたクラスタラベル。データインデックスと対応するクラスタIDのマッピング情報を含む。min_cluster_size=3, max_cluster_size=50の設定で生成。
- カラムの説明:
  - data_index: データポイントの一意な識別子
  - cluster: クラスタラベル（-1はノイズポイントを示す）
  - probability: データポイントがクラスタに属する確率
  - outlier_score: 外れ値スコア（値が大きいほど外れ値の可能性が高い）
- 基本統計:
  - Shape: (9883, 4) - 9883行、4カラム
  - クラスタ数: 67（-1のノイズクラスタを含む）
  - カラム: data_index, cluster, probability, outlier_score
  - 平均確率: 0.052 (標準偏差: 0.220)
  - 平均外れ値スコア: 0.029 (標準偏差: 0.054)

### 4. kmeans_same_size_metrics_2d.csv
- 生成日時: 2025-02-10
- 生成コミット: 49c9aee
- 生成元スクリプト: experiments/clustering_comparison.py
- データの意味: KMeansクラスタリングの評価指標（2次元データ）。クラスタサイズを同じにした場合の比較実験結果。各クラスタの密度指標（平均距離、最大距離）とサイズを含む。クラスタ間の比較のための基準データとして使用。
- カラムの説明:
  - cluster_id: クラスタの一意な識別子
  - size: クラスタ内の要素数
  - avg_distance: クラスタ内の全ペア間の平均距離（密度の逆指標）
  - max_distance: クラスタ内の最大距離
  - density: 密度指標（平均距離の逆数として計算）
- 基本統計:
  - Shape: (952, 5) - 952クラスタ、5カラム
  - 平均クラスタサイズ: 10.38 (標準偏差: 4.55)
  - クラスタサイズ範囲: 1-35
  - 平均密度: 11.00 (標準偏差: 3.27)

### 5. hdbscan_detailed_metrics_2d.csv
- 生成日時: 2025-02-10
- 生成コミット: 49c9aee
- 生成元スクリプト: experiments/clustering_comparison.py
- データの意味: HDBSCANクラスタリングの詳細な評価指標（2次元データ）。各クラスタの密度指標（平均距離、最大距離）、サイズ、およびシルエット係数を含む。ノイズポイント（-1のラベル）は除外して計算。
- カラムの説明:
  - cluster_id: クラスタの一意な識別子
  - size: クラスタ内の要素数（最小5）
  - avg_distance: クラスタ内の全ペア間の平均距離（密度の逆指標）
  - max_distance: クラスタ内の最大距離
  - density: 密度指標（平均距離の逆数として計算）
- 基本統計:
  - Shape: (676, 5) - 676クラスタ、5カラム
  - 平均クラスタサイズ: 10.37 (標準偏差: 5.39)
  - クラスタサイズ範囲: 5-30
  - 平均密度: 15.45 (標準偏差: 7.87)
