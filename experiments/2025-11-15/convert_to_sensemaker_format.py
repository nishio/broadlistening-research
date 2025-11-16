#!/usr/bin/env python3
"""
Convert arguments.csv to Jigsaw Sensemaker input format.

Sensemaker expects CSV with columns:
- comment-id (required)
- comment_text (required)
- Optional: vote data

Our arguments.csv has:
- arg_id
- argument
- comment_id
- cluster_id
- cluster_name
- cluster_takeaways
- x, y, p
"""

import pandas as pd
import sys

def convert_to_sensemaker_format(input_csv, output_csv):
    """
    Convert arguments.csv to Sensemaker format.

    Args:
        input_csv: Path to arguments.csv
        output_csv: Path to output CSV for Sensemaker
    """
    # Read the arguments CSV
    df = pd.read_csv(input_csv)

    print(f"Loaded {len(df)} arguments from {input_csv}")
    print(f"Columns: {list(df.columns)}")

    # Create Sensemaker format
    # Rename columns to match Sensemaker's expected format
    sensemaker_df = pd.DataFrame({
        'comment-id': df['arg_id'],
        'comment_text': df['argument']
    })

    # Save to CSV
    sensemaker_df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"\nConverted to Sensemaker format:")
    print(f"- Output file: {output_csv}")
    print(f"- Total comments: {len(sensemaker_df)}")
    print(f"\nFirst few rows:")
    print(sensemaker_df.head())

    # Print some statistics
    print(f"\nStatistics:")
    print(f"- Average comment length: {sensemaker_df['comment_text'].str.len().mean():.1f} characters")
    print(f"- Min comment length: {sensemaker_df['comment_text'].str.len().min()} characters")
    print(f"- Max comment length: {sensemaker_df['comment_text'].str.len().max()} characters")

if __name__ == "__main__":
    input_csv = "arguments.csv"
    output_csv = "arguments_sensemaker.csv"

    print("=" * 60)
    print("Converting arguments.csv to Jigsaw Sensemaker format")
    print("=" * 60)

    convert_to_sensemaker_format(input_csv, output_csv)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
