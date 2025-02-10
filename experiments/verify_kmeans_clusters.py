import pandas as pd

def analyze_kmeans_clusters():
    # データの読み込み
    df = pd.read_csv("experiments/results/kmeans_same_size_metrics.csv")
    
    # サイズ5以上のクラスタをフィルタリング
    filtered_df = df[df["size"] >= 5]
    
    # 密度順でソートして上位66件を取得
    top_66_df = filtered_df.sort_values("density", ascending=False).head(66)
    
    # 結果の出力
    print(f"サイズ5以上のクラスタ総数: {len(filtered_df)}")
    print(f"上位66件の密度範囲: {top_66_df['density'].min():.3f} - {top_66_df['density'].max():.3f}")
    print(f"上位66件のサイズ範囲: {top_66_df['size'].min()} - {top_66_df['size'].max()}")

if __name__ == "__main__":
    analyze_kmeans_clusters()
