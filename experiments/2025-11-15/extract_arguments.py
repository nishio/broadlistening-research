#!/usr/bin/env python3
"""
Extract arguments data from the Tokyo Metropolitan Government broad listening page.
"""

import json
import csv
import sys

def extract_arguments(json_file, output_csv, output_json):
    """
    Extract arguments from the JSON data.

    Args:
        json_file: Path to the input JSON file
        output_csv: Path to the output CSV file
        output_json: Path to the output JSON file
    """
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract clusters
    clusters = data['props']['pageProps']['result']['clusters']

    # Collect all arguments
    all_arguments = []

    for cluster in clusters:
        cluster_name = cluster['cluster']
        cluster_id = cluster['cluster_id']
        takeaways = cluster['takeaways']

        for arg in cluster['arguments']:
            arg_data = {
                'arg_id': arg['arg_id'],
                'argument': arg['argument'],
                'comment_id': arg['comment_id'],
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'cluster_takeaways': takeaways,
                'x': arg.get('x', ''),
                'y': arg.get('y', ''),
                'p': arg.get('p', '')
            }
            all_arguments.append(arg_data)

    # Save as CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['arg_id', 'argument', 'comment_id', 'cluster_id', 'cluster_name',
                     'cluster_takeaways', 'x', 'y', 'p']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_arguments)

    # Save as JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'clusters': clusters,
            'arguments': all_arguments,
            'metadata': {
                'total_clusters': len(clusters),
                'total_arguments': len(all_arguments)
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(all_arguments)} arguments from {len(clusters)} clusters")
    print(f"CSV saved to: {output_csv}")
    print(f"JSON saved to: {output_json}")

    # Print cluster summary
    print("\nCluster summary:")
    for cluster in clusters:
        print(f"  - {cluster['cluster']} (ID: {cluster['cluster_id']}): {len(cluster['arguments'])} arguments")

if __name__ == "__main__":
    json_file = 'data.json'
    output_csv = 'arguments.csv'
    output_json = 'arguments_full.json'

    extract_arguments(json_file, output_csv, output_json)
