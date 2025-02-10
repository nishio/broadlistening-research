# 実験ファイルの構成と対応関係

## ディレクトリ構造
```
.
├── dataset/
│   └── aipubcom/      # パブコメデータセット
├── experiments/
│   ├── core/          # 基本的なユーティリティ
│   ├── clustering/    # クラスタリング実験
│   └── results/       # 実験結果
└── notes/             # 研究ノート
```

## データセット (dataset/aipubcom/)
- embeddings.pkl (261MB)
  * パブコメの埋め込みベクトル
  * 形状: (9883, 3072)
  * 個人および組織からのパブコメを含む
  * 注: GitHubの容量制限（100MB）を超えるため、別途管理

## コアファイル (experiments/core/)
1. data_processing.py
   - 目的: データの読み込みと前処理の共通機能
   - 機能:
     * embeddings.pklの読み込みと形状確認
     * メモリ使用量の監視
     * 複数のデータパスに対応
   - 使用方法:
     ```python
     from core.data_processing import load_embeddings
     embeddings_array, embeddings_df = load_embeddings()
     ```

## クラスタリング実験 (experiments/clustering/)
1. original_params.py
   - 目的: オリジナルのHDBSCANパラメータでの実験
   - 実験内容:
     * min_cluster_size=5
     * max_cluster_size=30
     * min_samples=2
   - 結果:
     * クラスタ数: 66
     * ノイズポイント: 94.6%
     * 実行時間: 約7.0分
     * メモリ使用量: 約1.6GB

2. large_kmeans_experiments.py
   - 目的: 大規模k-meansクラスタリング実験
   - 実験内容:
     * k=50: 平均198ポイント/クラスタ
     * k=100: 平均99ポイント/クラスタ
     * k=200: 平均49ポイント/クラスタ
   - 結果:
     * 実行時間: 約0.1分
     * メモリ使用量: 約1.6GB
     * クラスタサイズが均一に分布

## 実験結果 (experiments/results/)
1. large_kmeans_results.json
   - 目的: k-means実験結果の保存
   - 内容:
     * 各kに対するクラスタサイズの分布
     * 実行時間とメモリ使用量
     * タイムスタンプと環境情報

2. subset_test_results.json
   - 目的: 実験パラメータの検証
   - 内容:
     * サブセットでのテスト実行結果
     * パラメータの妥当性確認
     * 小規模データでの動作確認

## 研究ノート (notes/)
1. 2024-02-10-research-notes-01.md
   - 目的: 本日の実験記録
   - 内容:
     * HDBSCANとk-meansの比較実験
     * 実行時間とメモリ使用量の分析
     * 今後の課題と学んだこと

2. research_guidelines.md
   - 目的: 研究プロジェクトの指針
   - 内容:
     * 研究ノートの書き方
     * 実験の進め方
     * ファイル構造と命名規則
     * コーディング規約
