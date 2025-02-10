# 実験ファイルの対応関係

## コアファイル
1. data_processing.py
   - データの読み込みと前処理
   - embeddings.pklの読み込みと形状確認

2. original_params.py
   - オリジナルのHDBSCANパラメータでの実験
   - min_cluster_size=5, max_cluster_size=30, min_samples=2

3. large_kmeans_experiments.py
   - k-meansクラスタリング実験（k=50,100,200）
   - クラスタサイズの分布分析
   - 実行時間とメモリ使用量の記録

## 実験結果
1. results/large_kmeans_results.json
   - k-means実験の結果
   - 各kに対するクラスタサイズの分布

2. results/subset_test_results.json
   - サブセットでのテスト実行結果
   - 実験パラメータの検証用

## 研究ノート
1. notes/2024-02-10-research-notes-01.md
   - 本日の実験結果と考察
   - 今後の課題と学んだこと

2. notes/research_guidelines.md
   - 研究ノートの書き方ガイドライン
   - 実験の進め方の指針
