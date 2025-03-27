"""
クラスタラベル比較スクリプト

異なるパラメータで生成されたクラスタラベルを比較します。
"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_cluster_labels(label_file):
    """クラスタラベルファイルを読み込む"""
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        print(f"ラベルファイルの読み込みに失敗しました: {e}")
        return None

def extract_metadata(experiment_dir):
    """実験のメタデータを抽出"""
    metadata_file = os.path.join(experiment_dir, "clustering_metadata.json")
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"メタデータファイルの読み込みに失敗しました: {e}")
        return {}

def compare_labels(experiment_dirs):
    """複数の実験結果からラベルを比較"""
    all_results = []
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        metadata = extract_metadata(exp_dir)
        
        label_files = [f for f in os.listdir(exp_dir) if f.endswith("_labels.json")]
        
        for label_file in label_files:
            if "cluster_" in label_file:
                cluster_size = label_file.split("_")[1]
            else:
                cluster_size = "unknown"
            
            labels = load_cluster_labels(os.path.join(exp_dir, label_file))
            if not labels:
                continue
            
            for label_info in labels:
                all_results.append({
                    "experiment": exp_name,
                    "cluster_size": cluster_size,
                    "cluster_id": label_info.get("cluster_id", -1),
                    "label": label_info.get("label", ""),
                    "sentiment": label_info.get("sentiment", ""),
                    "size": label_info.get("size", 0),
                    "keywords": ", ".join(label_info.get("keywords", []))
                })
    
    if not all_results:
        print("比較するラベルが見つかりませんでした。")
        return None
    
    return pd.DataFrame(all_results)

def generate_comparison_report(results_df, output_file):
    """ラベル比較レポートを生成"""
    if results_df is None:
        return
    
    pivot = results_df.pivot_table(
        index="experiment", 
        columns="cluster_size", 
        values="cluster_id", 
        aggfunc="count"
    )
    
    sentiment_counts = results_df.groupby(["experiment", "sentiment"]).size().unstack()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# クラスタラベル比較レポート\n\n")
        
        f.write("## クラスタサイズごとのラベル数\n\n")
        f.write(pivot.to_markdown() + "\n\n")
        
        f.write("## センチメント分布\n\n")
        if sentiment_counts is not None and not sentiment_counts.empty:
            f.write(sentiment_counts.to_markdown() + "\n\n")
        else:
            f.write("センチメント情報がありません。\n\n")
        
        f.write("## クラスタラベル一覧\n\n")
        
        for experiment in results_df["experiment"].unique():
            f.write(f"### 実験: {experiment}\n\n")
            
            for cluster_size in sorted(results_df[results_df["experiment"] == experiment]["cluster_size"].unique()):
                f.write(f"#### クラスタサイズ: {cluster_size}\n\n")
                
                cluster_labels = results_df[(results_df["experiment"] == experiment) & 
                                           (results_df["cluster_size"] == cluster_size)]
                
                for _, row in cluster_labels.sort_values("cluster_id").iterrows():
                    f.write(f"- **クラスタID {row['cluster_id']}** ({row['size']}件): {row['label']}\n")
                    if row['keywords']:
                        f.write(f"  - キーワード: {row['keywords']}\n")
                
                f.write("\n")
    
    print(f"比較レポートを生成しました: {output_file}")
    return output_file

def export_comparison_data(results_df, output_file):
    """比較データをCSVファイルとして出力"""
    if results_df is None:
        return
    
    results_df.to_csv(output_file, index=False)
    print(f"比較データをCSVファイルとして出力しました: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="クラスタラベル比較ツール")
    parser.add_argument("--experiment-dirs", nargs="+", required=True,
                      help="比較する実験ディレクトリのリスト")
    parser.add_argument("--output-report", default="comparison_report.md",
                      help="出力するレポートファイル名")
    parser.add_argument("--output-csv", default="comparison_data.csv",
                      help="出力するCSVファイル名")
    
    args = parser.parse_args()
    
    valid_dirs = []
    for exp_dir in args.experiment_dirs:
        if os.path.isdir(exp_dir):
            valid_dirs.append(exp_dir)
        else:
            print(f"警告: 実験ディレクトリ {exp_dir} が存在しません。")
    
    if not valid_dirs:
        print("有効な実験ディレクトリがありません。")
        return
    
    results_df = compare_labels(valid_dirs)
    
    if results_df is not None:
        generate_comparison_report(results_df, args.output_report)
        export_comparison_data(results_df, args.output_csv)
    
    print("比較処理が完了しました。")

if __name__ == "__main__":
    main()
