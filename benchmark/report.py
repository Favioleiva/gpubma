"""Read historical benchmark tables without launching scientific computation."""
from pathlib import Path
import argparse
import pandas as pd

HERE=Path(__file__).resolve().parent

def recorded_report():
    metrics=pd.read_csv(HERE/'recorded/benchmark_metrics.csv')
    return metrics[metrics.algorithm.isin(['BFG','Baseline_Random','Baseline_Greedy'])].copy()

def recorded_rank_trajectory():
    """Running best of ALREADY RECORDED exact ranks; unknown ranks remain unknown.

    This is a report view of historical observations, not scoring a new run.
    """
    rows=pd.read_csv(HERE/'recorded/recorded_rank_observations.csv.gz')
    rows=rows.sort_values(['benchmark','algorithm','budget','seed','eval_order'])
    rank=rows.exact_rank.where(rows.exact_rank>0)
    rows['best_recorded_rank']=rank.groupby([rows[c] for c in
        ['benchmark','algorithm','budget','seed']]).cummin()
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks',action='store_true')
    args=parser.parse_args()
    print((recorded_rank_trajectory() if args.ranks else recorded_report()).to_string(index=False))

if __name__=='__main__':main()
