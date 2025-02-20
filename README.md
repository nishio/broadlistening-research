# Broad Listening Research

パブリックコメントのクラスタリング分析を行うための研究プロジェクト。

## プロジェクトの概要
AIと著作権に関するパブリックコメントのクラスタリング分析を行い、意見の傾向を把握することを目的としています。

## セットアップ手順

### 1. 必要なデータファイルの取得
以下のファイルを`dataset/aipubcom/`に配置する必要があります：

1. embeddings.pkl（275MB）
   - 内容: パブコメの埋め込みベクトル（9883件）
   - 形状: (9883, 3072)
   - 注意: GitHubの容量制限（100MB）を超えるため、Git LFSで管理
   - 取得方法: `git lfs pull`を実行してください

2. args.csv
   - 内容: パブコメのメタデータ

詳細なセットアップ手順は[SETUP.md](SETUP.md)を参照してください。

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
└── notes/            # 研究ノート（ファイル名: YYYY-MM-DD-連番-内容.md）
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

詳細な実験の対応関係は[EXPERIMENT_MAPPING.md](EXPERIMENT_MAPPING.md)を参照してください。
