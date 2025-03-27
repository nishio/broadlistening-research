"""クラスタラベルの質をテストするスクリプト"""

import json
import pandas as pd
from generate_cluster_labels import validate_label_quality

def test_label_quality(labels_file):
    """
    ラベル生成の質をテスト
    
    Parameters:
    -----------
    labels_file : str
        テストするラベルのJSONファイル
    
    Returns:
    --------
    dict
        テスト結果の統計情報
    """
    print(f"\nラベル品質テストを開始: {labels_file}")
    
    try:
        with open(labels_file, encoding='utf-8') as f:
            labels = json.load(f)
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました - {e}")
        return None
    
    results = {
        "total": len(labels),
        "valid": 0,
        "invalid": 0,
        "generic": 0,
        "too_short": 0,
        "too_long": 0,
        "no_domain_keywords": 0,
        "details": []
    }
    
    print("\nラベルの検証を開始...")
    for label_info in labels:
        label = label_info["label"]
        texts = label_info["texts"]
        is_valid, reason = validate_label_quality(label, texts)
        
        if is_valid:
            results["valid"] += 1
        else:
            results["invalid"] += 1
            results["details"].append({
                "cluster_id": label_info["cluster_id"],
                "label": label,
                "reason": reason
            })
            
            if reason == "一般的すぎるラベル":
                results["generic"] += 1
            elif reason == "ラベルが短すぎる":
                results["too_short"] += 1
            elif reason == "ラベルが長すぎる":
                results["too_long"] += 1
            elif reason == "ドメイン固有の単語が含まれていない":
                results["no_domain_keywords"] += 1
    
    # 結果の表示
    print("\n=== テスト結果 ===")
    print(f"総ラベル数: {results['total']}")
    print(f"有効なラベル: {results['valid']} ({results['valid']/results['total']*100:.1f}%)")
    print(f"無効なラベル: {results['invalid']} ({results['invalid']/results['total']*100:.1f}%)")
    print("\n無効なラベルの内訳:")
    print(f"- 一般的すぎる: {results['generic']}")
    print(f"- 短すぎる: {results['too_short']}")
    print(f"- 長すぎる: {results['too_long']}")
    print(f"- ドメイン固有語なし: {results['no_domain_keywords']}")
    
    if results["invalid"] > 0:
        print("\n=== 無効なラベルの詳細 ===")
        for detail in results["details"]:
            print(f"クラスタID {detail['cluster_id']}: {detail['label']}")
            print(f"理由: {detail['reason']}\n")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python test_label_quality.py <labels_file>")
        sys.exit(1)
    
    labels_file = sys.argv[1]
    test_label_quality(labels_file)
