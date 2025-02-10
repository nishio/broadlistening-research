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

## ブランチ管理
1. ブランチ命名規則
   - 形式: devin/{timestamp}-{descriptive-slug}
   - 例: devin/1739163427-update-data-setup
   - タイムスタンプ生成: `date +%s`

2. ブランチ作成
```bash
# 新しいブランチの作成
git checkout main
git pull
git checkout -b devin/$(date +%s)-your-feature
```

## データファイルの取得

### 必要なファイル
1. embeddings.pkl（261MB）
   - パブコメの埋め込みベクトル
   - 形状: (9883, 3072)
   - 個人および組織からのパブコメを含む
   - 注意: GitHubの容量制限（100MB）を超えるため、別途管理

2. args.csv
   - パブコメのメタデータ
   - コメントIDと対応する情報を含む

### ファイルの取得方法
1. anno-broadlisteningリポジトリをクローン
```bash
gh repo clone takahiroanno2024/anno-broadlistening
```

2. 必要なファイルをコピー
```bash
# このリポジトリのdatasetディレクトリを作成
mkdir -p dataset/aipubcom

# ファイルをコピー
cp anno-broadlistening/scatter/pipeline/outputs/aipubcom/embeddings.pkl dataset/aipubcom/
cp anno-broadlistening/scatter/pipeline/outputs/aipubcom/args.csv dataset/aipubcom/
```

### ファイル配置の確認
```bash
ls -lh dataset/aipubcom/
```

以下のファイルが存在することを確認：
- dataset/aipubcom/embeddings.pkl（約261MB）
- dataset/aipubcom/args.csv

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
### embeddings.pklが見つからない場合
1. anno-broadlisteningリポジトリが正しくクローンされているか確認
2. パス`anno-broadlistening/scatter/pipeline/outputs/aipubcom/`が存在するか確認
3. ファイルの権限とアクセス権を確認

### メモリ不足エラーが発生する場合
- 最低16GBのRAMを推奨
- スワップ領域の確保を検討
- 必要に応じてサブセットでのテスト実行を検討
