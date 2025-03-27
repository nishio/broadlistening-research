# クラスタリングパラメータ実験

このディレクトリには、kouchou-aiのクラスタリングアルゴリズムを使用して異なるパラメータでの実験を行うためのスクリプト群が含まれています。元のkouchou-aiのコードを変更せずに、様々なクラスタリングパラメータでの実験を容易に行い、結果を比較するための環境を提供します。

## 目的

1. 異なるクラスタ数設定でのクラスタリング結果を生成する
2. クラスタリング結果のラベルを中間データとして保存する
3. 異なるパラメータでの実験結果を比較する
4. 評価関数を作るための準備データを提供する

## 前提条件

以下のリポジトリとデータが必要です：

1. kouchou-aiリポジトリ（`~/repos/kouchou-ai`にクローン済みであること）
2. 埋め込みデータ（`embeddings.pkl`）
3. 引数データ（`args.csv`）

また、以下のPythonパッケージが必要です：

```
numpy
pandas
scikit-learn
umap-learn
matplotlib
```

## スクリプトの説明

### 1. run_clustering_experiment.py

単一のパラメータセットでクラスタリング実験を実行するスクリプトです。

**使用方法**:
```bash
python run_clustering_experiment.py --embeddings /path/to/embeddings.pkl --args /path/to/args.csv --cluster-nums 3 6 12 24 --output experiment_results
```

**引数**:
- `--embeddings`: 埋め込みデータのパス（.pkl）
- `--args`: 引数データのパス（.csv）
- `--cluster-nums`: クラスタサイズのリスト（例: 3 6 12 24）
- `--output`: 出力ディレクトリ

**出力**:
- `clustering_result.csv`: クラスタリング結果（各行に対するクラスタIDを含む）
- `clustering_metadata.json`: 実験のメタデータ
- `cluster_X_for_labelling.csv`: 各クラスタサイズ（X）に対するラベリング用データ

### 2. generate_cluster_labels_cli.py

クラスタリング結果にラベルを付与するスクリプトです。

**使用方法**:
```bash
python generate_cluster_labels_cli.py --cluster-file /path/to/cluster_file.csv --output-file /path/to/output.json --report-file /path/to/report.md
```

**引数**:
- `--cluster-file`: クラスタリング結果のCSVファイル
- `--output-file`: 出力するJSONファイル名
- `--report-file`: 出力するレポートファイル名（オプション）

**出力**:
- JSONファイル: クラスタラベル情報
- Markdownレポート: ラベル情報のレポート

### 3. run_multi_experiment.py

設定ファイルに基づいて複数のパラメータセットで実験を実行するスクリプトです。

**使用方法**:
```bash
python run_multi_experiment.py --config experiment_config.json --embeddings /path/to/embeddings.pkl --args /path/to/args.csv --output experiments --run-all --generate-labels
```

**引数**:
- `--config`: 実験設定ファイル
- `--embeddings`: 埋め込みデータのパス（.pkl）
- `--args`: 引数データのパス（.csv）
- `--output`: 出力ディレクトリ
- `--run-all`: すべての実験を実行（フラグ）
- `--experiment`: 特定の実験名のみを実行
- `--generate-labels`: ラベルも生成する（フラグ）

**出力**:
- 各実験ごとのディレクトリ（タイムスタンプ付き）
- 各ディレクトリ内にクラスタリング結果とラベル情報

### 4. compare_cluster_labels.py

異なるパラメータで生成されたクラスタラベルを比較するスクリプトです。

**使用方法**:
```bash
python compare_cluster_labels.py --experiment-dirs experiments/default_20250327_123456 experiments/fine_grained_20250327_123456 --output-report comparison_report.md --output-csv comparison_data.csv
```

**引数**:
- `--experiment-dirs`: 比較する実験ディレクトリのリスト
- `--output-report`: 出力するレポートファイル名
- `--output-csv`: 出力するCSVファイル名

**出力**:
- Markdownレポート: 比較結果のレポート
- CSVファイル: 比較データ

### 5. experiment_config.json

実験設定を定義するJSONファイルです。

**例**:
```json
{
  "experiments": [
    {
      "name": "default",
      "description": "デフォルトのクラスタ数設定",
      "cluster_nums": [3, 6, 12, 24]
    },
    {
      "name": "fine_grained",
      "description": "より細かいクラスタ数の設定",
      "cluster_nums": [5, 10, 20, 40]
    }
  ]
}
```

## 実験手順

### 基本的な実験手順

1. 埋め込みデータと引数データを準備する
2. 単一のパラメータセットで実験を行う場合：
   ```bash
   python run_clustering_experiment.py --embeddings /path/to/embeddings.pkl --args /path/to/args.csv --cluster-nums 3 6 12 24 --output experiment_results
   ```

3. 生成されたクラスタリング結果にラベルを付与する：
   ```bash
   python generate_cluster_labels_cli.py --cluster-file experiment_results/cluster_3_for_labelling.csv --output-file experiment_results/cluster_3_labels.json
   ```

### 複数パラメータでの実験手順

1. 実験設定ファイル（experiment_config.json）を編集する
2. 複数のパラメータセットで実験を実行する：
   ```bash
   python run_multi_experiment.py --config experiment_config.json --embeddings /path/to/embeddings.pkl --args /path/to/args.csv --output experiments --run-all
   ```

3. ラベルも同時に生成する場合は `--generate-labels` フラグを追加する：
   ```bash
   python run_multi_experiment.py --config experiment_config.json --embeddings /path/to/embeddings.pkl --args /path/to/args.csv --output experiments --run-all --generate-labels
   ```

### 実験結果の比較手順

1. 複数の実験結果を比較する：
   ```bash
   python compare_cluster_labels.py --experiment-dirs experiments/default_20250327_123456 experiments/fine_grained_20250327_123456 --output-report comparison_report.md
   ```

## 出力ファイルの説明

### クラスタリング結果（clustering_result.csv）

各行（パブコメ）に対するクラスタIDを含むCSVファイルです。以下の列が含まれます：

- `arg-id`: 引数ID
- `argument`: 引数テキスト
- `cluster_X`: 各クラスタサイズ（X）に対するクラスタID
- `umap_x`, `umap_y`: UMAP座標

### ラベリング用データ（cluster_X_for_labelling.csv）

ラベル生成のための入力ファイルです。以下の列が含まれます：

- `arg-id`: 引数ID
- `argument`: 引数テキスト
- `cluster_id`: クラスタID

### クラスタラベル（cluster_X_labels.json）

クラスタに対するラベル情報を含むJSONファイルです。各クラスタに対して以下の情報が含まれます：

- `cluster_id`: クラスタID
- `label`: クラスタのラベル
- `sentiment`: センチメント（ポジティブ/ネガティブ/ニュートラル）
- `size`: クラスタのサイズ（含まれる引数の数）
- `keywords`: キーワードリスト

### 比較レポート（comparison_report.md）

異なる実験結果の比較レポートです。以下の情報が含まれます：

- クラスタサイズごとのラベル数
- センチメント分布
- クラスタラベル一覧

## 注意事項

1. kouchou-aiリポジトリのパスは `~/repos/kouchou-ai` を想定しています。異なる場合は `run_clustering_experiment.py` の `KOUCHOU_PATH` 変数を修正してください。
2. 大きなデータセットでは処理に時間がかかる場合があります。
3. 実験結果は出力ディレクトリに保存されます。十分なディスク容量があることを確認してください。

## トラブルシューティング

### モジュールのインポートエラー

kouchou-aiのモジュールがインポートできない場合は、以下を確認してください：

1. kouchou-aiリポジトリが正しくクローンされているか
2. `KOUCHOU_PATH` が正しく設定されているか

### メモリエラー

大きなデータセットを処理する際にメモリエラーが発生する場合は、以下を試してください：

1. より小さなデータセットでテストする
2. クラスタ数を減らす
3. より多くのメモリを持つマシンで実行する
