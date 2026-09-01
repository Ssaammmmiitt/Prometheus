"""Standalone script to evaluate seasonal forecast skill metrics."""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from prometheus.api.app import forecasts_root
from prometheus.infer import verify

def evaluate_season(year: int):
    """Run verification for an entire fire season and dump a report."""
    start_date = date(year, 1, 1)
    end_date = date(year, 6, 30) # Typical spring fire season in Nepal

    print(f"Evaluating seasonal skill for {year} fire season ({start_date} to {end_date})...")
    
    root = Path(forecasts_root())
    
    # Run the existing verification range which updates verification.csv
    df = verify.verify_range(start_date, end_date, root=root, append=True)
    
    if df.empty:
        print("No forecast data found for this season.")
        return
        
    df = df[df["valid"] == True]
    
    if df.empty:
        print("No valid days with fire observations found for this season.")
        return
        
    print(f"\nSeasonal Forecast Skill Report ({year}):")
    print("-" * 40)
    print(f"Total days evaluated: {len(df)}")
    print(f"Mean PR-AUC:          {df['pr_auc'].mean():.4f}")
    print(f"Mean Brier Score:     {df['brier'].mean():.4f}")
    
    if 'top10_capture' in df.columns:
        print(f"Mean Top-10% Capture: {df['top10_capture'].mean():.4f}")
        
    if 'fss' in df.columns:
        print(f"Mean FSS:             {df['fss'].mean():.4f}")
        
    if 'rev' in df.columns:
        print(f"Mean Economic Value:  {df['rev'].mean():.4f}")
        
    print("-" * 40)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Seasonal Forecast Skill")
    parser.add_argument("--year", type=int, default=2024, help="Year to evaluate")
    args = parser.parse_args()
    
    evaluate_season(args.year)
