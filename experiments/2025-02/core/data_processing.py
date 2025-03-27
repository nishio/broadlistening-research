import os
import psutil
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def check_memory_usage() -> Tuple[float, float]:
    """メモリ使用量を確認"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    available_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
    return memory_gb, available_gb

def validate_embeddings(embeddings_array: np.ndarray) -> None:
    """埋め込みベクトルの検証"""
    expected_shape = (9883, 3072)
    if embeddings_array.shape != expected_shape:
        raise ValueError(
            f"埋め込みベクトルの形状が不正です。"
            f"期待値: {expected_shape}, 実際: {embeddings_array.shape}"
        )

def find_data_file(filename: str, data_dir: Optional[str] = None) -> str:
    """データファイルのパスを探索"""
    possible_paths = [
        os.path.join("dataset/aipubcom", filename),
        os.path.join("../dataset/aipubcom", filename),
        os.path.join("../../dataset/aipubcom", filename),
        os.path.join("../../../dataset/aipubcom", filename)
    ]
    
    if data_dir:
        possible_paths.insert(0, os.path.join(data_dir, filename))
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    setup_guide = """
    データファイルが見つかりません。以下の手順で取得してください：
    1. anno-broadlistening/scatter/pipeline/outputs/aipubcom/から取得
    2. dataset/aipubcomディレクトリに配置
    詳細は SETUP.md を参照してください。
    """
    raise FileNotFoundError(f"{filename}が見つかりません。\n{setup_guide}")

def load_embeddings(data_dir: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """パブコメの埋め込みベクトルを読み込み"""
    print("=== データの読み込み開始 ===")
    
    # メモリ使用量の確認
    memory_gb, available_gb = check_memory_usage()
    print(f"現在のメモリ使用量: {memory_gb:.1f}GB")
    print(f"利用可能なメモリ: {available_gb:.1f}GB")
    
    if available_gb < 2.0:
        raise MemoryError(
            f"メモリ不足の可能性があります。"
            f"少なくとも2GB以上の空きメモリが必要です。"
            f"現在の空きメモリ: {available_gb:.1f}GB"
        )
    
    # embeddings.pklの読み込み
    embeddings_path = find_data_file("embeddings.pkl", data_dir)
    print(f"embeddings.pklを読み込み中: {embeddings_path}")
    embeddings_df = pd.read_pickle(embeddings_path)
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    
    # データの検証
    validate_embeddings(embeddings_array)
    print(f"データの形状: {embeddings_array.shape}")
    
    # args.csvの存在確認
    args_path = find_data_file("args.csv", data_dir)
    print(f"args.csvの場所を確認: {args_path}")
    
    return embeddings_array, embeddings_df

if __name__ == "__main__":
    # データの読み込みテスト
    try:
        embeddings_array, embeddings_df = load_embeddings()
        print("\nデータの詳細:")
        print(f"データ型: {embeddings_array.dtype}")
        print(f"最小値: {embeddings_array.min():.3f}")
        print(f"最大値: {embeddings_array.max():.3f}")
        print(f"メモリ使用量: {check_memory_usage()[0]:.1f}GB")
    except (FileNotFoundError, ValueError, MemoryError) as e:
        print(f"エラー: {str(e)}")
