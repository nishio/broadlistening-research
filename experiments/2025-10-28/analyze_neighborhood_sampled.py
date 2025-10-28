"""
HDBSCANクラスタの高次元近傍構造を分析（サンプリング版）

計算時間削減のため、データをサンプリングして分析します。
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
    print(f'元データ: {embeddings_array.shape}')

# サンプリング（5000点）
np.random.seed(42)
n_samples = 5000
sample_indices = np.random.choice(len(embeddings_array), n_samples, replace=False)
embeddings_sampled = embeddings_array[sample_indices]
print(f'サンプル: {embeddings_sampled.shape}')

# HDBSCANクラスタリング
print('\n=== HDBSCANクラスタリング ===')
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, metric='euclidean')
hdbscan_labels = hdbscan_clusterer.fit_predict(embeddings_sampled)
n_clusters = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
n_noise = np.sum(hdbscan_labels == -1)
print(f'クラスタ数: {n_clusters}, ノイズ点数: {n_noise} ({n_noise/len(hdbscan_labels)*100:.1f}%)')

# 高次元空間での15近傍を計算
print('\n=== 高次元空間での近傍構造分析 ===')
k = 15
nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
nn.fit(embeddings_sampled)
distances, indices = nn.kneighbors(embeddings_sampled)

# 各点について、15近傍のうち同じクラスタに属する点の割合を計算
same_cluster_ratios = []
cross_cluster_neighbors = []  # 他クラスタへの近傍リンク情報

for i in range(len(hdbscan_labels)):
    my_label = hdbscan_labels[i]
    neighbor_labels = hdbscan_labels[indices[i, 1:]]

    if my_label == -1:  # ノイズ点
        same_cluster_ratios.append(np.nan)
    else:
        same_cluster_ratio = np.sum(neighbor_labels == my_label) / k
        same_cluster_ratios.append(same_cluster_ratio)

        # 他クラスタへのリンク数
        other_cluster_links = neighbor_labels[(neighbor_labels != my_label) & (neighbor_labels != -1)]
        if len(other_cluster_links) > 0:
            for other_label in other_cluster_links:
                cross_cluster_neighbors.append((my_label, other_label))

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

# 境界点の分析
boundary_mask = cluster_ratios < 0.9
n_boundary = np.sum(boundary_mask)
print(f'\n境界点（同一クラスタ割合<0.9）: {n_boundary}点 ({n_boundary/len(cluster_ratios)*100:.1f}%)')

# クラスタ間の近傍リンク分析
if len(cross_cluster_neighbors) > 0:
    print(f'\n=== クラスタ間の近傍リンク ===')
    from collections import Counter
    link_counts = Counter(cross_cluster_neighbors)
    print(f'総リンク数: {len(cross_cluster_neighbors)}')
    print(f'最も多いリンク（上位10）:')
    for (from_c, to_c), count in link_counts.most_common(10):
        print(f'  クラスタ{from_c} → クラスタ{to_c}: {count}本')

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
        'avg_same_ratio': avg_same_ratio,
        'boundary_pct': boundary_ratio * 100
    })

df_stats = pd.DataFrame(cluster_boundary_stats).sort_values('boundary_pct', ascending=False)
print(df_stats.to_string(index=False))

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ヒストグラム
ax = axes[0]
ax.hist(cluster_ratios, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(np.nanmean(cluster_ratios), color='red', linestyle='--', linewidth=2,
           label=f'平均={np.nanmean(cluster_ratios):.3f}')
ax.set_xlabel('15近傍のうち同一クラスタ内の点の割合', fontsize=11)
ax.set_ylabel('点数', fontsize=11)
ax.set_title('高次元空間での近傍構造の純度\n(1.0 = 15近傍すべて同一クラスタ)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# クラスタごとの境界点割合
ax = axes[1]
x = df_stats['size']
y = df_stats['boundary_pct']
colors = plt.cm.rainbow(np.linspace(0, 1, len(df_stats)))
ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
for i, row in df_stats.iterrows():
    ax.annotate(f"C{int(row['cluster_id'])}", (row['size'], row['boundary_pct']),
                fontsize=9, ha='center', va='bottom')
ax.set_xlabel('クラスタサイズ', fontsize=11)
ax.set_ylabel('境界点の割合 (%)', fontsize=11)
ax.set_title('クラスタサイズと境界点の関係\n(境界点 = 15近傍の10%以上が他クラスタ)', fontsize=12)
ax.grid(alpha=0.3)

# クラスタごとの平均同一クラスタ割合
ax = axes[2]
x = df_stats['cluster_id']
y = df_stats['avg_same_ratio']
colors = plt.cm.rainbow(np.linspace(0, 1, len(df_stats)))
bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax.axhline(0.9, color='red', linestyle='--', linewidth=2, label='閾値 0.9')
ax.set_xlabel('クラスタID', fontsize=11)
ax.set_ylabel('平均同一クラスタ割合', fontsize=11)
ax.set_title('クラスタごとの近傍純度', fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('neighborhood_analysis_sampled.png', dpi=150, bbox_inches='tight')
print('\n図を neighborhood_analysis_sampled.png に保存しました')

# 結論
print('\n=== 結論 ===')
avg_ratio = np.nanmean(cluster_ratios)
boundary_pct = n_boundary / len(cluster_ratios) * 100

if avg_ratio > 0.95:
    print(f'✓ 平均 {avg_ratio:.1%}: ほとんどの点の近傍は同一クラスタ内')
    print('  → UMAPの局所構造保存とHDBSCANクラスタは整合的')
    print('  → 2次元での引き裂かれは大域構造の歪みが原因')
elif avg_ratio > 0.80:
    print(f'△ 平均 {avg_ratio:.1%}: 多くの点で近傍に他クラスタが混入')
    print(f'  → 境界点 {boundary_pct:.1f}%')
    print('  → UMAPは混入した近傍関係も保存しようとする')
    print('  → 2次元でクラスタがオーバーラップする原因')
else:
    print(f'✗ 平均 {avg_ratio:.1%}: 近傍構造が大きく混在')
    print('  → HDBSCANのクラスタ自体が高次元で複雑な形状')
