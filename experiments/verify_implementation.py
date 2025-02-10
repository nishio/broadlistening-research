import pandas as pd
import json
from generate_cluster_labels import generate_cluster_labels
from evaluate_cluster_labels import evaluate_cluster_labels
from process_dataset_x import process_dataset_x
from visualize_evaluation_scores import plot_evaluation_comparison
from generate_comparison_report import generate_comparison_report

def verify_with_subset():
    """実装の検証（小規模なサブセットを使用）"""
    print("1. Dataset Xの準備（上位5件）...")
    kmeans_df = pd.read_csv('experiments/results/kmeans_same_size_metrics.csv')
    filtered_df = kmeans_df[kmeans_df['size'] >= 5]
    dataset_x = filtered_df.nlargest(5, 'density')  # 上位5件のみ
    dataset_x.to_csv('experiments/results/test_dataset_x.csv', index=False)
    
    print("\n2. クラスタラベル生成のテスト...")
    labels = generate_cluster_labels(
        cluster_file="experiments/results/test_dataset_x.csv",
        output_file="experiments/results/test_labels.json"
    )
    
    # JSONフォーマットの確認
    print("\n3. JSONフォーマットの確認...")
    with open("experiments/results/test_labels.json", "r", encoding="utf-8") as f:
        test_labels = json.load(f)
        print("- label形式:", "label" in test_labels[0])
        print("- description形式:", "description" in test_labels[0])
        print("- keywords形式:", "keywords" in test_labels[0])
        print("- sentiment形式:", "sentiment" in test_labels[0])
    
    print("\n4. GPT-4oの使用確認...")
    with open("experiments/generate_cluster_labels.py", "r", encoding="utf-8") as f:
        content = f.read()
        if 'model="gpt-4o"' in content:
            print("- モデル名の確認: OK")
        else:
            print("- モデル名の確認: 要修正")
    
    print("\n5. 評価スコアの確認...")
    evaluations = evaluate_cluster_labels(
        labels_file="experiments/results/test_labels.json",
        output_file="experiments/results/test_evaluation.json"
    )
    
    # 評価スコアの範囲確認
    with open("experiments/results/test_evaluation.json", "r", encoding="utf-8") as f:
        test_eval = json.load(f)
        for eval in test_eval:
            print(f"- クラスタ {eval['cluster_id']} の総合スコア: {eval['total_score']}")
            if not (0 <= eval['total_score'] <= 100):
                print("  警告: スコアが範囲外です")
    
    print("\nテスト完了")

if __name__ == "__main__":
    verify_with_subset()
