# 実験ファイルの構成と対応関係

## データセット要件
### 必須ファイル
1. embeddings.pkl（261MB）
   - 場所: anno-broadlistening/scatter/pipeline/outputs/aipubcom/
   - 形状: (9883, 3072)
   - 内容: パブコメの埋め込みベクトル
   - 注意: GitHubの容量制限（100MB）を超えるため、別途管理

2. args.csv
   - 場所: anno-broadlistening/scatter/pipeline/outputs/aipubcom/
   - 内容: パブコメのメタデータ
   - 依存: embeddings.pklのインデックスと対応

## 実験ファイルの対応
### コアモジュール (experiments/core/)
1. data_processing.py
   - 目的: データの読み込みと前処理
   - 依存: embeddings.pkl, args.csv
   - メモリ要件: 約2GB
   - エラー処理: ファイル不在、メモリ不足の検出

### クラスタリング実験 (experiments/clustering/)
1. original_params.py
   - 目的: オリジナルのHDBSCANパラメータでの実験
   - 依存: data_processing.pyを介してembeddings.pkl
   - メモリ要件: 約2GB
   - 実行時間: 約7分

2. large_kmeans_experiments.py
   - 目的: k-meansクラスタリング実験
   - 依存: data_processing.pyを介してembeddings.pkl
   - メモリ要件: 約2GB
   - 実行時間: 約0.1分

## 実験結果の保存
### 結果ファイル (experiments/results/)
- クラスタリング結果のJSON形式での保存
- メトリクスとパラメータの記録
- 実行時間とメモリ使用量の記録

## データの取り扱い
1. データセットの配置
   ```
   dataset/
   └── aipubcom/
       ├── embeddings.pkl  # 261MB
       └── args.csv
   ```

2. データ検証手順
   ```python
   # データの形状確認
   python experiments/core/data_processing.py
   ```

3. トラブルシューティング
   - ファイル不在時: SETUP.mdの手順に従う
   - メモリ不足時: 他のプロセスを終了
   - 読み込みエラー: ファイルの破損を確認
