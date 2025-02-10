# Broad Listening Research

パブリックコメントのクラスタリング分析を行うための研究プロジェクト。

## セットアップ手順

### 1. 必要なデータファイルの取得
以下のファイルを`dataset/aipubcom/`に配置する必要があります：

1. embeddings.pkl（261MB）
   - 場所: anno-broadlistening/scatter/pipeline/outputs/aipubcom/
   - 内容: パブコメの埋め込みベクトル
   - 形状: (9883, 3072)
   - 注意: GitHubの容量制限（100MB）を超えるため、別途管理

2. args.csv
   - 場所: anno-broadlistening/scatter/pipeline/outputs/aipubcom/
   - 内容: パブコメのメタデータ

### 2. 必要なパッケージのインストール
```bash
pip install numpy pandas scikit-learn hdbscan umap-learn
```

### 3. ディレクトリ構造
```
.
├── dataset/
│   └── aipubcom/      # パブコメデータセット
├── experiments/
│   ├── core/          # 基本的なユーティリティ
│   ├── clustering/    # クラスタリング実験
│   └── results/       # 実験結果
├── reports/           # 実験レポート
└── notes/            # 研究ノート
```

## 実験の実行方法
1. HDBSCANクラスタリング
```bash
python experiments/clustering/original_params.py
```

2. k-meansクラスタリング
```bash
python experiments/clustering/large_kmeans_experiments.py
```

## 実験結果の確認
- 実験結果: `experiments/results/`
- 実験レポート: `reports/`
- 研究ノート: `notes/`

詳細な実験の対応関係は`EXPERIMENT_MAPPING.md`を参照してください。
