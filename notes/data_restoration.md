# データファイルの復元と再生成に関するノート

## 1. 削除されたデータファイルの復元
### kmeans_same_size_metrics.csv
- 場所: experiments/results/metrics/kmeans_same_size_metrics.csv
- 復元方法: Git履歴（コミット9699392）から復元
- 内容: k-meansクラスタリングのメトリクス（1226クラスタ）
- 重要性: process_dataset_x.pyの入力として必要

## 2. 再生成が必要だったデータ
### kmeans_cluster_labels.csv
- 場所: experiments/results/kmeans_cluster_labels.csv
- 生成方法: k-meansクラスタリング（k=1226）の実行
- 内容: 各データポイントのクラスタ割り当て
- 形式:
  - カラム: arg-id, comment-id, argument, cluster-id
  - 行数: 9883
  - クラスタID: 0-1225

### kmeans_same_size_metrics.csv（新規）
- 場所: experiments/results/kmeans_same_size_metrics.csv
- 生成方法: クラスタメトリクスの計算
- 内容: 各クラスタの統計情報
- 形式:
  - カラム: cluster_id, size, avg_distance, max_distance, density
  - 行数: 1226（ヘッダー除く）
  - メトリクス: 平均距離、最大距離、密度

## 3. データ形式の要件
### process_dataset_x.pyの入力要件
1. kmeans_same_size_metrics.csv:
   - 必須カラム: cluster_id, size, avg_distance, max_distance, density
   - cluster_idは0から始まる連番
   - sizeは各クラスタの要素数
   - densityは平均距離の逆数として計算

2. kmeans_cluster_labels.csv:
   - 必須カラム: arg-id, comment-id, argument, cluster-id
   - arg-idはargs.csvと一致する形式
   - cluster-idはkmeans_same_size_metrics.csvと一致

## 4. データ生成スクリプト
### kmeans_1226.py
- 目的: k-meansクラスタリングの実行
- 入力: embeddings.pkl, args.csv
- 出力: kmeans_cluster_labels.csv
- パラメータ:
  - n_clusters=1226
  - random_state=42

### generate_metrics.py
- 目的: クラスタメトリクスの計算
- 入力: embeddings.pkl, kmeans_cluster_labels.csv
- 出力: kmeans_same_size_metrics.csv
- 計算内容:
  - クラスタごとのサイズ
  - クラスタ内の平均距離
  - クラスタ内の最大距離
  - 密度（平均距離の逆数）

## 5. 検証方法
1. データファイルの存在確認
2. カラム名と形式の確認
3. 行数の確認（9883行のデータ、1226クラスタ）
4. process_dataset_x.pyでの動作確認
5. クラスタサイズの分布確認

## 6. 今後の改善点
1. データファイルのバックアップ体制の整備
2. 再生成手順のドキュメント化
3. データ形式の検証スクリプトの作成
4. Git LFSの活用検討（大きなデータファイル用）
