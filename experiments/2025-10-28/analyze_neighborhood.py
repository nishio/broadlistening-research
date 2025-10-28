"""
HDBSCANクラスタの高次元近傍構造を分析

疑問: HDBSCANで抽出されたクラスタは高次元空間で密集しているはずなのに、
      UMAPで2次元可視化すると引き裂かれたりオーバーラップするのはなぜか？

仮説: UMAPの「局所構造」(n_neighbors=15)とHDBSCANの「クラスタ」は
      異なるスケールを見ているため、クラスタの境界付近で異なるクラスタへの
      近傍リンクが発生し、2次元投影で構造が歪む。

検証: 各点の15近傍を高次元空間で計算し、同一クラスタ内の点の割合を調べる。
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import hdbscan
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# データ読み込み
print('=== データ読み込み ===')
with open('../../dataset/team-mirai/embeddings.pkl', 'rb') as f:
    df_embeddings = pickle.load(f)
    embeddings_array = np.array(df_embeddings['embedding'].tolist())
    print(f'Embeddings: {embeddings_array.shape}')

# HDBSCANクラスタリング（キャッシュを使用）
import os
cache_file = 'hdbscan_labels_cache.pkl'
if os.path.exists(cache_file):
    print('\n=== キャッシュからHDBSCANラベルを読み込み ===')
    with open(cache_file, 'rb') as f:
        hdbscan_labels = pickle.load(f)
else:
    print('\n=== HDBSCANクラスタリング ===')
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
    hdbscan_labels = hdbscan_clusterer.fit_predict(embeddings_array)
    with open(cache_file, 'wb') as f:
        pickle.dump(hdbscan_labels, f)
    print(f'ラベルを {cache_file} に保存しました')

n_clusters = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
n_noise = np.sum(hdbscan_labels == -1)
print(f'クラスタ数: {n_clusters}, ノイズ点数: {n_noise} ({n_noise/len(hdbscan_labels)*100:.1f}%)')

# 高次元空間での15近傍を計算
print('\n=== 高次元空間での近傍構造分析 ===')
k = 15
nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1は自分自身を含むため
nn.fit(embeddings_array)
distances, indices = nn.kneighbors(embeddings_array)

# 各点について、15近傍のうち同じクラスタに属する点の割合を計算
same_cluster_ratios = []
for i in range(len(hdbscan_labels)):
    my_label = hdbscan_labels[i]
    neighbor_labels = hdbscan_labels[indices[i, 1:]]  # 自分自身を除く

    if my_label == -1:  # ノイズ点
        same_cluster_ratios.append(np.nan)
    else:
        same_cluster_ratio = np.sum(neighbor_labels == my_label) / k
        same_cluster_ratios.append(same_cluster_ratio)

same_cluster_ratios = np.array(same_cluster_ratios)

# 結果の統計
cluster_points_mask = hdbscan_labels >= 0
cluster_ratios = same_cluster_ratios[cluster_points_mask]

print(f'\nクラスタ内点の近傍分析 (n_neighbors={k}):')
print(f'  平均: {np.nanmean(cluster_ratios):.3f} (同一クラスタ内の割合)')
print(f'  中央値: {np.nanmedian(cluster_ratios):.3f}')
print(f'  最小: {np.nanmin(cluster_ratios):.3f}')
print(f'  最大: {np.nanmax(cluster_ratios):.3f}')

# 割合ごとのヒストグラム
print(f'\n同一クラスタ内の近傍割合の分布:')
bins = [0, 0.5, 0.7, 0.9, 0.95, 1.0]
for i in range(len(bins)-1):
    count = np.sum((cluster_ratios >= bins[i]) & (cluster_ratios < bins[i+1]))
    pct = count / len(cluster_ratios) * 100
    print(f'  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:5d}点 ({pct:5.1f}%)')

# 0.9未満（10%以上が異なるクラスタ）の点を詳細分析
boundary_mask = cluster_ratios < 0.9
n_boundary = np.sum(boundary_mask)
print(f'\n境界点（同一クラスタ割合<0.9）: {n_boundary}点 ({n_boundary/len(cluster_ratios)*100:.1f}%)')

if n_boundary > 0:
    print('\n境界点の例（最初の10点）:')
    boundary_indices = np.where(cluster_points_mask)[0][boundary_mask][:10]
    for idx in boundary_indices:
        my_label = hdbscan_labels[idx]
        neighbor_labels = hdbscan_labels[indices[idx, 1:]]
        neighbor_unique, neighbor_counts = np.unique(neighbor_labels, return_counts=True)
        print(f'  点{idx}: クラスタ{my_label}, 近傍→', end='')
        for label, count in zip(neighbor_unique, neighbor_counts):
            if label == -1:
                print(f'ノイズ:{count}点', end=' ')
            else:
                print(f'C{label}:{count}点', end=' ')
        print(f'(同一クラスタ割合={same_cluster_ratios[idx]:.2f})')

# クラスタごとの分析
print('\n=== クラスタごとの境界点分析 ===')
cluster_ids = np.unique(hdbscan_labels[hdbscan_labels >= 0])
cluster_boundary_stats = []

for cluster_id in cluster_ids:
    cluster_mask = hdbscan_labels == cluster_id
    cluster_size = np.sum(cluster_mask)
    cluster_ratios_this = same_cluster_ratios[cluster_mask]

    boundary_ratio = np.sum(cluster_ratios_this < 0.9) / cluster_size
    avg_same_ratio = np.nanmean(cluster_ratios_this)

    cluster_boundary_stats.append({
        'cluster_id': cluster_id,
        'size': cluster_size,
        'avg_same_cluster_ratio': avg_same_ratio,
        'boundary_point_ratio': boundary_ratio
    })

df_stats = pd.DataFrame(cluster_boundary_stats).sort_values('boundary_point_ratio', ascending=False)
print(df_stats.to_string(index=False))

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ヒストグラム
ax = axes[0]
ax.hist(cluster_ratios, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(np.nanmean(cluster_ratios), color='red', linestyle='--',
           label=f'平均={np.nanmean(cluster_ratios):.3f}')
ax.set_xlabel('15近傍のうち同一クラスタ内の点の割合')
ax.set_ylabel('点数')
ax.set_title('高次元空間での近傍構造の純度')
ax.legend()
ax.grid(alpha=0.3)

# クラスタごとの境界点割合
ax = axes[1]
x = df_stats['size']
y = df_stats['boundary_point_ratio'] * 100
colors = plt.cm.rainbow(np.linspace(0, 1, len(df_stats)))
ax.scatter(x, y, c=colors, s=100, alpha=0.7)
for i, row in df_stats.iterrows():
    ax.annotate(f"C{row['cluster_id']}", (row['size'], row['boundary_point_ratio']*100),
                fontsize=8, ha='center')
ax.set_xlabel('クラスタサイズ')
ax.set_ylabel('境界点の割合 (%)')
ax.set_title('クラスタサイズと境界点の関係')
ax.set_xscale('log')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('neighborhood_analysis.png', dpi=150, bbox_inches='tight')
print('\n図を neighborhood_analysis.png に保存しました')

# 結論
print('\n=== 結論 ===')
avg_ratio = np.nanmean(cluster_ratios)
if avg_ratio > 0.95:
    print('✓ ほとんどの点の近傍は同一クラスタ内に収まっている')
    print('  → UMAPの局所構造保存とHDBSCANクラスタは整合的')
    print('  → 2次元での引き裂かれは別の原因（大域構造の歪み等）')
elif avg_ratio > 0.80:
    print('△ 多くの点で近傍に他クラスタの点が混入している')
    print('  → UMAPは混入した近傍関係も保存しようとする')
    print('  → 2次元でクラスタがオーバーラップする原因')
else:
    print('✗ 近傍構造が大きく混在している')
    print('  → HDBSCANのクラスタ自体が高次元で複雑な形状の可能性')
