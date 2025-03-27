"""
複数パラメータでのクラスタリング実験

設定ファイルに基づいて複数のパラメータセットで実験を実行します。
"""

import argparse
import json
import os
import subprocess
from datetime import datetime

def run_experiment(experiment_config, embeddings_path, args_path, base_output_dir):
    """実験設定に基づいて実験を実行"""
    experiment_name = experiment_config["name"]
    cluster_nums = experiment_config["cluster_nums"]
    
    print(f"\n=== 実験 '{experiment_name}' を開始します ===")
    print(f"説明: {experiment_config.get('description', 'なし')}")
    print(f"クラスタ数: {cluster_nums}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=2)
    
    cmd = [
        "python", "run_clustering_experiment.py",
        "--embeddings", embeddings_path,
        "--args", args_path,
        "--cluster-nums"
    ] + [str(n) for n in cluster_nums] + ["--output", output_dir]
    
    print("実行コマンド:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"実験 '{experiment_name}' が正常に完了しました。")
        print(f"結果は {output_dir} に保存されました。")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"実験 '{experiment_name}' の実行中にエラーが発生しました: {e}")
        return None

def generate_labels_for_experiment(experiment_dir):
    """実験結果にラベルを生成"""
    cluster_files = [f for f in os.listdir(experiment_dir) if f.startswith("cluster_") and f.endswith("_for_labelling.csv")]
    
    if not cluster_files:
        print(f"警告: {experiment_dir} にラベリング用クラスタファイルが見つかりません。")
        return
    
    print(f"\n=== {len(cluster_files)}個のクラスタファイルにラベルを生成します ===")
    
    for cluster_file in cluster_files:
        cluster_size = cluster_file.split("_")[1]
        
        output_file = os.path.join(experiment_dir, f"cluster_{cluster_size}_labels.json")
        report_file = os.path.join(experiment_dir, f"cluster_{cluster_size}_report.md")
        
        cmd = [
            "python", "generate_cluster_labels_cli.py",
            "--cluster-file", os.path.join(experiment_dir, cluster_file),
            "--output-file", output_file,
            "--report-file", report_file
        ]
        
        print(f"クラスタサイズ {cluster_size} のラベルを生成中...")
        print("実行コマンド:", " ".join(cmd))
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"クラスタサイズ {cluster_size} のラベル生成が完了しました。")
        except subprocess.CalledProcessError as e:
            print(f"クラスタサイズ {cluster_size} のラベル生成中にエラーが発生しました: {e}")

def main():
    parser = argparse.ArgumentParser(description="複数パラメータでのクラスタリング実験")
    parser.add_argument("--config", default="experiment_config.json", 
                      help="実験設定ファイル")
    parser.add_argument("--embeddings", required=True, 
                      help="埋め込みデータのパス（.pkl）")
    parser.add_argument("--args", required=True, 
                      help="引数データのパス（.csv）")
    parser.add_argument("--output", default="experiments", 
                      help="出力ディレクトリ")
    parser.add_argument("--run-all", action="store_true", 
                      help="すべての実験を実行")
    parser.add_argument("--experiment", 
                      help="特定の実験名のみを実行")
    parser.add_argument("--generate-labels", action="store_true", 
                      help="ラベルも生成する")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.experiment:
        selected_experiments = [exp for exp in config["experiments"] if exp["name"] == args.experiment]
        if not selected_experiments:
            print(f"指定された実験名 '{args.experiment}' は設定ファイルに存在しません。")
            return
    elif args.run_all:
        selected_experiments = config["experiments"]
    else:
        selected_experiments = [config["experiments"][0]]
        print(f"注意: デフォルトでは最初の実験のみ実行されます。すべての実験を実行するには --run-all を指定してください。")
    
    completed_experiments = []
    for experiment in selected_experiments:
        output_dir = run_experiment(experiment, args.embeddings, args.args, args.output)
        if output_dir:
            completed_experiments.append(output_dir)
    
    if args.generate_labels and completed_experiments:
        for experiment_dir in completed_experiments:
            generate_labels_for_experiment(experiment_dir)
    
    print("\n=== すべての処理が完了しました ===")
    for experiment_dir in completed_experiments:
        print(f"実験結果: {experiment_dir}")

if __name__ == "__main__":
    main()
