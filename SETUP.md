# セットアップ手順

## Git LFSのセットアップ
大規模ファイルを扱うためにGit LFSを使用します。

1. Git LFSのインストール
```bash
# Ubuntuの場合
sudo apt-get install git-lfs

# Git LFSの初期化
git lfs install
```

2. LFSの設定
```bash
# embeddings.pklをLFSで管理
git lfs track "*.pkl"
git add .gitattributes
```

## データファイルの取得

### 必要なファイル
1. embeddings.pkl（261MB）
   - パブコメの埋め込みベクトル
   - 形状: (9883, 3072)
   - 個人および組織からのパブコメを含む
   - 注意: GitHubの容量制限（100MB）を超えるため、Git LFSを使っている、ファイルサイズが小さい時はLFSを使えていない

2. args.csv
   - パブコメのメタデータ
   - コメントIDと対応する情報を含む

## 環境のセットアップ
1. 必要なPythonパッケージのインストール
```bash
pip install numpy pandas scikit-learn hdbscan umap-learn
```

2. ディレクトリ構造の確認
```bash
tree -L 2
```

以下の構造になっていることを確認：
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

## トラブルシューティング

### メモリ不足エラーが発生する場合
- 最低16GBのRAMを推奨
- スワップ領域の確保を検討
- 必要に応じてサブセットでのテスト実行を検討
