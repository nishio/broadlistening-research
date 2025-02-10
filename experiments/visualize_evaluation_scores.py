import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'

def plot_evaluation_comparison():
    """HDBSCANとDataset Xの評価結果を比較"""
    try:
        # 評価結果の読み込み
        with open("experiments/results/hdbscan_evaluation.json", "r", encoding="utf-8") as f:
            hdbscan_eval = json.load(f)
        with open("experiments/results/dataset_x_evaluation.json", "r", encoding="utf-8") as f:
            dataset_x_eval = json.load(f)
    except FileNotFoundError as e:
        print(f"評価結果のJSONファイルが見つかりません: {e}")
        return
        
    metrics = {
        'total_score': '総合評価スコア',
        'consistency_score': '一貫性スコア',
        'specificity_score': '具体性スコア',
        'coverage_score': '網羅性スコア',
        'keyword_score': 'キーワードスコア'
    }
    
    # 評価結果の可視化
    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10, 6))
        sns.histplot(data=pd.DataFrame(hdbscan_eval), x=metric_key, 
                    label='HDBSCAN', alpha=0.5, bins=10)
        sns.histplot(data=pd.DataFrame(dataset_x_eval), x=metric_key, 
                    label='Dataset X', alpha=0.5, bins=10)
        plt.title(f'{metric_name}の分布比較')
        plt.xlabel('スコア')
        plt.ylabel('頻度')
        plt.legend()
        plt.savefig(f'experiments/results/comparison_{metric_key}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_evaluation_comparison()
    print("評価結果の可視化が完了しました。")
    print("出力ファイル:")
    print("- experiments/results/comparison_*.png")
