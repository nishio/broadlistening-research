import pandas as pd
import json
from generate_cluster_labels import generate_cluster_labels, format_labels_report
from evaluate_cluster_labels import evaluate_cluster_labels, format_evaluation_report

def compare_clustering_methods():
    """HDBSCANとk-meansのクラスタラベルを生成し評価する"""
    
    # HDBSCANの結果に対する処理
    print("HDBSCANのクラスタラベル生成を開始...")
    hdbscan_labels = generate_cluster_labels(
        cluster_file="hdbscan_clustered_arguments.csv",
        output_file="hdbscan_cluster_labels.json"
    )
    format_labels_report(hdbscan_labels, "hdbscan_cluster_labels_report.md")
    
    print("HDBSCANのクラスタラベル評価を開始...")
    hdbscan_evaluations = evaluate_cluster_labels(
        labels_file="hdbscan_cluster_labels.json",
        output_file="hdbscan_label_evaluation.json"
    )
    format_evaluation_report(hdbscan_evaluations, "hdbscan_label_evaluation_report.md")
    
    # k-meansの結果に対する処理
    print("k-meansのクラスタラベル生成を開始...")
    kmeans_labels = generate_cluster_labels(
        cluster_file="kmeans_clustered_arguments.csv",
        output_file="kmeans_cluster_labels.json"
    )
    format_labels_report(kmeans_labels, "kmeans_cluster_labels_report.md")
    
    print("k-meansのクラスタラベル評価を開始...")
    kmeans_evaluations = evaluate_cluster_labels(
        labels_file="kmeans_cluster_labels.json",
        output_file="kmeans_label_evaluation.json"
    )
    format_evaluation_report(kmeans_evaluations, "kmeans_label_evaluation_report.md")
    
    # 比較結果のサマリーを生成
    generate_comparison_summary(hdbscan_evaluations, kmeans_evaluations)

def generate_comparison_summary(hdbscan_evals, kmeans_evals):
    """両手法の評価結果を比較するサマリーを生成"""
    
    def calculate_stats(evals):
        scores = [e["total_score"] for e in evals]
        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "cluster_count": len(evals),
            "avg_consistency": sum(e["consistency_score"] for e in evals) / len(evals),
            "avg_specificity": sum(e["specificity_score"] for e in evals) / len(evals),
            "avg_coverage": sum(e["coverage_score"] for e in evals) / len(evals),
            "avg_keyword": sum(e["keyword_score"] for e in evals) / len(evals)
        }
    
    hdbscan_stats = calculate_stats(hdbscan_evals)
    kmeans_stats = calculate_stats(kmeans_evals)
    
    # Markdown形式で比較レポートを生成
    report = [
        "# クラスタリング手法の比較結果\n",
        "## 評価スコアの比較\n",
        "### HDBSCAN",
        f"- クラスタ数: {hdbscan_stats['cluster_count']}",
        f"- 平均スコア: {hdbscan_stats['avg_score']:.1f}",
        f"- 最高スコア: {hdbscan_stats['max_score']}",
        f"- 最低スコア: {hdbscan_stats['min_score']}",
        "\n項目別平均スコア:",
        f"- 一貫性: {hdbscan_stats['avg_consistency']:.1f}/30",
        f"- 具体性: {hdbscan_stats['avg_specificity']:.1f}/25",
        f"- 網羅性: {hdbscan_stats['avg_coverage']:.1f}/25",
        f"- キーワード: {hdbscan_stats['avg_keyword']:.1f}/20\n",
        "### k-means",
        f"- クラスタ数: {kmeans_stats['cluster_count']}",
        f"- 平均スコア: {kmeans_stats['avg_score']:.1f}",
        f"- 最高スコア: {kmeans_stats['max_score']}",
        f"- 最低スコア: {kmeans_stats['min_score']}",
        "\n項目別平均スコア:",
        f"- 一貫性: {kmeans_stats['avg_consistency']:.1f}/30",
        f"- 具体性: {kmeans_stats['avg_specificity']:.1f}/25",
        f"- 網羅性: {kmeans_stats['avg_coverage']:.1f}/25",
        f"- キーワード: {kmeans_stats['avg_keyword']:.1f}/20\n",
        "## 考察",
        "### スコアの差異",
        f"- 平均スコアの差: {abs(hdbscan_stats['avg_score'] - kmeans_stats['avg_score']):.1f}点",
        "- HDBSCANの特徴:",
        "  * 小規模で密度の高いクラスタを形成",
        "  * クラスタ内の意見の一貫性が高い傾向",
        "- k-meansの特徴:",
        "  * 大規模なクラスタを形成",
        "  * 広範な意見をカバーする傾向",
        "\n### 用途による使い分け",
        "- HDBSCAN: 具体的な意見グループの特定に適する",
        "- k-means: 全体的な意見傾向の把握に適する"
    ]
    
    # レポートの保存
    with open("clustering_comparison_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    compare_clustering_methods()
