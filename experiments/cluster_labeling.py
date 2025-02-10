import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
import json
import os
from openai import OpenAI

def create_cluster_labels(dataset_dir="dataset/aipubcom"):
    # データの読み込み
    arguments_df = pd.read_csv(f"{dataset_dir}/args.csv")
    arguments_array = arguments_df["argument"].values
    
    embeddings_df = pd.read_pickle(f"{dataset_dir}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    
    # HDBSCANクラスタリング
    hdb = HDBSCAN(
        min_cluster_size=5,
        max_cluster_size=30,
        min_samples=2,
        core_dist_n_jobs=-1  # 並列処理を有効化
    )
    hdb.fit(embeddings_array)
    
    # クラスタ情報を含むDataFrameを作成
    result = pd.DataFrame({
        "arg-id": arguments_df["arg-id"],
        "comment-id": arguments_df["comment-id"],
        "argument": arguments_df["argument"],
        "cluster-id": hdb.labels_
    })
    
    # ノイズクラスタ（-1）を除外
    result_filtered = result[result["cluster-id"] != -1]
    
    # クラスタごとの解説を生成
    client = OpenAI()
    cluster_descriptions = []
    
    for cluster_id in sorted(result_filtered["cluster-id"].unique()):
        cluster_texts = result_filtered[result_filtered["cluster-id"] == cluster_id]["argument"].tolist()
        
        # GPTによるクラスタの解説生成
        prompt = f"""以下の意見グループについて分析し、JSONフォーマットで回答してください：

意見グループ：
{chr(10).join(cluster_texts)}

必要な情報：
1. 解説：このグループの意見の共通点や特徴を説明
2. 表札：このグループを代表するタイトル（30文字以内）
3. 興味深さ：60-90点で評価（特に興味深い特徴がある場合は高得点）

回答フォーマット：
{{
    "解説": "グループの特徴の説明...",
    "表札": "グループのタイトル",
    "興味深さ": 点数
}}"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        description = json.loads(response.choices[0].message.content)
        description["cluster_id"] = cluster_id
        description["texts"] = cluster_texts
        cluster_descriptions.append(description)
    
    # 結果を保存
    with open("cluster_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(cluster_descriptions, f, ensure_ascii=False, indent=2)
    
    result_filtered.to_csv("clustered_arguments.csv", index=False)
    
    return cluster_descriptions

if __name__ == "__main__":
    create_cluster_labels()
