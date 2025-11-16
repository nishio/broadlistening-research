#!/usr/bin/env python3
"""
Create a 1,000 comment sample from arguments_sensemaker.csv for testing.
"""

import pandas as pd

def create_sample(input_csv, output_csv, sample_size=1000):
    """
    Create a sample dataset for testing.

    Args:
        input_csv: Full dataset
        output_csv: Sample dataset output
        sample_size: Number of samples to extract
    """
    # Read the full dataset
    df = pd.read_csv(input_csv)

    print(f"Full dataset: {len(df)} comments")

    # Random sample
    sample_df = df.sample(n=sample_size, random_state=42)

    # Save sample
    sample_df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Sample dataset: {len(sample_df)} comments")
    print(f"Saved to: {output_csv}")

    # Show statistics
    print(f"\nSample statistics:")
    print(f"- Average comment length: {sample_df['comment_text'].str.len().mean():.1f} characters")
    print(f"- Min comment length: {sample_df['comment_text'].str.len().min()} characters")
    print(f"- Max comment length: {sample_df['comment_text'].str.len().max()} characters")

    print(f"\nFirst 5 comments:")
    print(sample_df.head())

if __name__ == "__main__":
    create_sample(
        input_csv="arguments_sensemaker.csv",
        output_csv="arguments_sample_1000.csv",
        sample_size=1000
    )
