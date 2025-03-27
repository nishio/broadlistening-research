"""
クラスタリングパラメータ実験スクリプト

kouchou-aiのクラスタリング実装を使用して、異なるパラメータでの実験を行い、
クラスタリング結果とラベルを中間データとして保存します。
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

KOUCHOU_PATH = os.path.expanduser("~/repos/kouchou-ai/server/broadlistening")
sys.path.append(KOUCHOU_PATH)
sys.path.append(os.path.join(KOUCHOU_PATH, "pipeline/steps"))

try:
    from hierarchical_clustering import hierarchical_clustering_embeddings
    from embedding import extract_embeddings_from_pkl
except ImportError as e:
    print(f"kouchou-aiのモジュールのインポートに失敗しました: {e}")
    print(f"KOUCHOU_PATH: {KOUCHOU_PATH}")
    sys.exit(1)

def load_embeddings(embedding_path):
    """埋め込みデータを読み込む"""
    try:
        print(f"埋め込みデータを読み込み中: {embedding_path}")
        embeddings_df = pd.read_pickle(embedding_path)
        embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
        print(f"埋め込みデータを読み込みました: shape={embeddings_array.shape}")
        return embeddings_array
    except Exception as e:
        print(f"埋め込みデータの読み込みに失敗しました: {e}")
        sys.exit(1)

def load_args(args_path):
    """引数データを読み込む"""
    try:
        print(f"引数データを読み込み中: {args_path}")
        args_df = pd.read_csv(args_path)
        print(f"引数データを読み込みました: {len(args_df)}件")
        return args_df
    except Exception as e:
        print(f"引数データの読み込みに失敗しました: {e}")
        sys.exit(1)

def run_clustering_experiment(embeddings_array, args_df, cluster_nums, output_dir):
    """クラスタリング実験を実行"""
    print(f"クラスタリング実験を開始: cluster_nums={cluster_nums}")
    
    try:
        from umap import UMAP
        umap_model = UMAP(random_state=42, n_components=2)
        umap_embeds = umap_model.fit_transform(embeddings_array)
        
        cluster_series = hierarchical_clustering_embeddings(umap_embeds, cluster_nums)
        
        result_df = args_df.copy()
        for n_clusters, labels in cluster_series.items():
            result_df[f"cluster_{n_clusters}"] = labels
        
        result_df["umap_x"] = umap_embeds[:, 0]
        result_df["umap_y"] = umap_embeds[:, 1]
        
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "clustering_result.csv")
        result_df.to_csv(result_path, index=False)
        
        metadata = {
            "cluster_nums": cluster_nums,
            "timestamp": datetime.now().isoformat(),
            "umap_shape": umap_embeds.shape,
            "embeddings_shape": embeddings_array.shape,
            "args_count": len(args_df)
        }
        
        metadata_path = os.path.join(output_dir, "clustering_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"クラスタリング結果を保存しました: {result_path}")
        print(f"メタデータを保存しました: {metadata_path}")
        
        return result_df
    except Exception as e:
        print(f"クラスタリング実験の実行に失敗しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def extract_cluster_for_labelling(result_df, n_clusters, output_dir):
    """特定のクラスタサイズのデータを抽出してラベリング用に保存"""
    print(f"クラスタサイズ {n_clusters} のデータを抽出中...")
    
    labelling_df = result_df[["arg-id", "argument", f"cluster_{n_clusters}"]].copy()
    labelling_df.rename(columns={f"cluster_{n_clusters}": "cluster_id"}, inplace=True)
    
    labelling_path = os.path.join(output_dir, f"cluster_{n_clusters}_for_labelling.csv")
    labelling_df.to_csv(labelling_path, index=False)
    
    print(f"ラベリング用データを保存しました: {labelling_path}")
    return labelling_path

def main():
    parser = argparse.ArgumentParser(description="kouchou-aiのクラスタリングパラメータ実験")
    parser.add_argument("--embeddings", required=True, help="埋め込みデータのパス（.pkl）")
    parser.add_argument("--args", required=True, help="引数データのパス（.csv）")
    parser.add_argument("--cluster-nums", type=int, nargs="+", default=[3, 6, 12, 24], 
                      help="クラスタサイズのリスト")
    parser.add_argument("--output", default="experiment_results", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    embeddings_path = os.path.abspath(os.path.expanduser(args.embeddings))
    args_path = os.path.abspath(os.path.expanduser(args.args))
    output_dir = os.path.abspath(args.output)
    
    embeddings_array = load_embeddings(embeddings_path)
    args_df = load_args(args_path)
    
    result_df = run_clustering_experiment(embeddings_array, args_df, args.cluster_nums, output_dir)
    
    for n_clusters in args.cluster_nums:
        extract_cluster_for_labelling(result_df, n_clusters, output_dir)
    
    print("\n実験が完了しました。以下のコマンドでラベル生成を行うことができます：")
    for n_clusters in args.cluster_nums:
        print(f"python ../../../experiments/2025-02/generate_cluster_labels.py "
              f"--cluster-file {output_dir}/cluster_{n_clusters}_for_labelling.csv "
              f"--output-file {output_dir}/cluster_{n_clusters}_labels.json")

if __name__ == "__main__":
    main()
