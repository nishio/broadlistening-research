"""
HDBSCANが見ている構造のスケールを分析

HDBSCANのパラメータ：
- min_samples=10: 各点の密度計算に使う近傍点数
- min_cluster_size=50: クラスタとして認識される最小サイズ

これらが実際にどのようなスケールの構造を抽出しているか検証する。
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

# データ読み込み（サンプリング版）
print('=== データ読み込み ===')
with open('../../dataset/team-mirai/embeddings.pkl', 'rb') as f:
    df_embeddings = pickle.load(f)
    embeddings_array = np.array(df_embeddings['embedding'].tolist())
    print(f'元データ: {embeddings_array.shape}')

np.random.seed(42)
n_samples = 5000
sample_indices = np.random.choice(len(embeddings_array), n_samples, replace=False)
embeddings_sampled = embeddings_array[sample_indices]
print(f'サンプル: {embeddings_sampled.shape}')

# HDBSCANクラスタリング
print('\n=== HDBSCANクラスタリング ===')
min_cluster_size = 30
min_samples = 5
hdbscan_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric='euclidean'
)
hdbscan_labels = hdbscan_clusterer.fit_predict(embeddings_sampled)
n_clusters = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
n_noise = np.sum(hdbscan_labels == -1)
print(f'パラメータ: min_samples={min_samples}, min_cluster_size={min_cluster_size}')
print(f'クラスタ数: {n_clusters}, ノイズ点数: {n_noise} ({n_noise/len(hdbscan_labels)*100:.1f}%)')

# 各スケールでの近傍構造を分析
print('\n=== 複数スケールでの近傍構造分析 ===')
k_values = [5, 10, 15, 30, 50, 100]  # HDBSCANのmin_samplesやmin_cluster_sizeと比較

results = []
for k in k_values:
    if k > len(embeddings_sampled) - 1:
        continue

    print(f'\nk={k}近傍を分析中...')
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(embeddings_sampled)
    distances, indices = nn.kneighbors(embeddings_sampled)

    # クラスタ内点のみ分析
    cluster_points_mask = hdbscan_labels >= 0
    same_cluster_ratios = []

    for i in range(len(hdbscan_labels)):
        if hdbscan_labels[i] == -1:
            continue

        my_label = hdbscan_labels[i]
        neighbor_labels = hdbscan_labels[indices[i, 1:]]
        same_cluster_ratio = np.sum(neighbor_labels == my_label) / k
        same_cluster_ratios.append(same_cluster_ratio)

    same_cluster_ratios = np.array(same_cluster_ratios)

    results.append({
        'k': k,
        'mean': np.mean(same_cluster_ratios),
        'median': np.median(same_cluster_ratios),
        'min': np.min(same_cluster_ratios),
        'max': np.max(same_cluster_ratios),
        'boundary_pct': np.sum(same_cluster_ratios < 0.9) / len(same_cluster_ratios) * 100
    })

    print(f'  平均同一クラスタ割合: {np.mean(same_cluster_ratios):.3f}')
    print(f'  境界点割合: {np.sum(same_cluster_ratios < 0.9) / len(same_cluster_ratios) * 100:.1f}%')

df_results = pd.DataFrame(results)
print('\n=== スケールごとの純度 ===')
print(df_results.to_string(index=False))

# HDBSCANのcore distanceを計算
print('\n=== HDBSCANのcore distance分析 ===')
# core distance = min_samples番目の最近傍点までの距離
nn_core = NearestNeighbors(n_neighbors=min_samples+1, metric='euclidean')
nn_core.fit(embeddings_sampled)
distances_core, _ = nn_core.kneighbors(embeddings_sampled)
core_distances = distances_core[:, min_samples]  # min_samples番目の距離

# クラスタごとのcore distance統計
cluster_ids = np.unique(hdbscan_labels[hdbscan_labels >= 0])
print(f'\nクラスタごとのcore distance統計:')
for cluster_id in cluster_ids:
    cluster_mask = hdbscan_labels == cluster_id
    cluster_core_distances = core_distances[cluster_mask]
    print(f'  クラスタ{cluster_id} (n={np.sum(cluster_mask)}):')
    print(f'    平均={np.mean(cluster_core_distances):.4f}, '
          f'中央値={np.median(cluster_core_distances):.4f}, '
          f'範囲=[{np.min(cluster_core_distances):.4f}, {np.max(cluster_core_distances):.4f}]')

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. スケールごとの同一クラスタ割合
ax = axes[0, 0]
ax.plot(df_results['k'], df_results['mean'], 'o-', linewidth=2, markersize=8, label='平均')
ax.plot(df_results['k'], df_results['median'], 's-', linewidth=2, markersize=8, label='中央値')
ax.axvline(min_samples, color='red', linestyle='--', linewidth=2,
           label=f'min_samples={min_samples}')
ax.axvline(min_cluster_size, color='orange', linestyle='--', linewidth=2,
           label=f'min_cluster_size={min_cluster_size}')
ax.set_xlabel('近傍点数 k', fontsize=11)
ax.set_ylabel('同一クラスタ内の割合', fontsize=11)
ax.set_title('スケールごとのクラスタ純度\n（k近傍のうち同一クラスタの点の割合）', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# 2. スケールごとの境界点割合
ax = axes[0, 1]
ax.plot(df_results['k'], df_results['boundary_pct'], 'o-', linewidth=2, markersize=8, color='purple')
ax.axvline(min_samples, color='red', linestyle='--', linewidth=2,
           label=f'min_samples={min_samples}')
ax.axvline(min_cluster_size, color='orange', linestyle='--', linewidth=2,
           label=f'min_cluster_size={min_cluster_size}')
ax.set_xlabel('近傍点数 k', fontsize=11)
ax.set_ylabel('境界点の割合 (%)', fontsize=11)
ax.set_title('スケールごとの境界点割合\n（k近傍の10%以上が他クラスタ）', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# 3. Core distanceの分布（クラスタごと）
ax = axes[1, 0]
for cluster_id in cluster_ids:
    cluster_mask = hdbscan_labels == cluster_id
    cluster_core_distances = core_distances[cluster_mask]
    ax.hist(cluster_core_distances, bins=30, alpha=0.5, label=f'C{cluster_id}')
ax.set_xlabel(f'Core Distance (min_samples={min_samples})', fontsize=11)
ax.set_ylabel('点数', fontsize=11)
ax.set_title(f'クラスタごとのCore Distance分布\n（各点の{min_samples}番目近傍までの距離）', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 4. k近傍での純度の分布（k=15の場合）
ax = axes[1, 1]
k_display = 15
nn_display = NearestNeighbors(n_neighbors=k_display+1, metric='euclidean')
nn_display.fit(embeddings_sampled)
distances_display, indices_display = nn_display.kneighbors(embeddings_sampled)

for cluster_id in cluster_ids:
    cluster_mask = hdbscan_labels == cluster_id
    cluster_ratios = []

    for i in np.where(cluster_mask)[0]:
        neighbor_labels = hdbscan_labels[indices_display[i, 1:]]
        same_ratio = np.sum(neighbor_labels == cluster_id) / k_display
        cluster_ratios.append(same_ratio)

    ax.hist(cluster_ratios, bins=20, alpha=0.5, label=f'C{cluster_id}')

ax.set_xlabel(f'{k_display}近傍での同一クラスタ割合', fontsize=11)
ax.set_ylabel('点数', fontsize=11)
ax.set_title(f'クラスタごとの近傍純度分布 (k={k_display})', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hdbscan_scale_analysis.png', dpi=150, bbox_inches='tight')
print('\n図を hdbscan_scale_analysis.png に保存しました')

# 結論
print('\n=== HDBSCANが見ているスケール ===')
print(f'1. 密度計算のスケール: min_samples={min_samples}')
print(f'   → 各点の{min_samples}番目近傍までの距離でlocal densityを計算')
print(f'   → {min_samples}近傍での平均純度: {df_results[df_results["k"]==min_samples]["mean"].values[0]:.3f}')

if min_cluster_size in df_results['k'].values:
    print(f'\n2. クラスタサイズのスケール: min_cluster_size={min_cluster_size}')
    print(f'   → {min_cluster_size}点以上のまとまりをクラスタとして認識')
    print(f'   → {min_cluster_size}近傍での平均純度: {df_results[df_results["k"]==min_cluster_size]["mean"].values[0]:.3f}')

print(f'\n3. UMAPのスケール: n_neighbors=15 (典型的な設定)')
print(f'   → 15近傍での平均純度: {df_results[df_results["k"]==15]["mean"].values[0]:.3f}')

print('\n【重要な発見】')
print('- HDBSCANは局所的に密度を計算（min_samples規模）')
print('- しかし、クラスタとしてはより大きな構造（min_cluster_size規模）を抽出')
print('- その間のスケール（UMAPのn_neighbors=15程度）では純度が低い')
print('- → UMAPとHDBSCANは異なるスケールの構造を見ている')
