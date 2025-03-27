"""
クラスタラベル生成スクリプト（CLI版）

コマンドライン引数をサポートするようにgenerate_cluster_labelsを拡張したバージョン
"""

import sys
import os
import argparse

sys.path.append(os.path.expanduser("~/repos/broadlistening-research/experiments/2025-02"))
from generate_cluster_labels import generate_cluster_labels, format_labels_report

def main():
    parser = argparse.ArgumentParser(description="クラスタラベル生成ツール")
    parser.add_argument("--cluster-file", required=True, 
                      help="クラスタリング結果のCSVファイル")
    parser.add_argument("--output-file", required=True,
                      help="出力するJSONファイル名")
    parser.add_argument("--report-file", 
                      help="出力するレポートファイル名（デフォルト: <output_file>_report.md）")
    
    args = parser.parse_args()
    
    print(f"クラスタファイル {args.cluster_file} からラベルを生成します...")
    labels = generate_cluster_labels(args.cluster_file, args.output_file)
    
    report_file = args.report_file or f"{os.path.splitext(args.output_file)[0]}_report.md"
    format_labels_report(labels, report_file)
    
    print("ラベル生成が完了しました。")
    print(f"- ラベル: {args.output_file}")
    print(f"- レポート: {report_file}")

if __name__ == "__main__":
    main()
