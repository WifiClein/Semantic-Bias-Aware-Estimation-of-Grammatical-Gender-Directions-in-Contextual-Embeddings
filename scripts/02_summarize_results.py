from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/root/autodl-tmp/fin")
IN_PATH = ROOT / "results/full_cv_12_methods/full_cv_12_methods_fold_results.csv"
OUT_DIR = ROOT / "results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "summary_12_methods_mean_std.csv"
RANKED_PATH = OUT_DIR / "ranked_12_methods.csv"
BEST_PATH = OUT_DIR / "best_method_by_language_model.csv"
CENTROID_PATH = OUT_DIR / "centroid_methods_only.csv"
ESTIMATOR_PATH = OUT_DIR / "estimator_overall_comparison.csv"
CONTEXT_PATH = OUT_DIR / "centroid_context_weight_overall_comparison.csv"


def main():
    df = pd.read_csv(IN_PATH)

    # -----------------------------
    # basic sanity
    # -----------------------------
    required = [
        "language", "model", "fold", "method",
        "context_source", "weighted", "estimator",
        "noun_gap_reduction",
        "raw_occ_gap",
        "clean_occ_gap",
        "occ_gap_preservation",
        "tradeoff_gmean",
        "tradeoff_hmean", 
        "weight_score",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # -----------------------------
    # rename model
    # -----------------------------
    df["base_model"] = df["model"]

    metric_cols = [
        "noun_gap_reduction",
        "occ_gap_preservation",
        "tradeoff_gmean",
        "tradeoff_hmean",   # ✅ FIX 2
    ]

    group_cols = [
        "language", "model", "base_model",
        "method", "context_source",
        "weighted", "weight_score","estimator"
    ]

    # -----------------------------
    # summary
    # -----------------------------
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()

    summary.to_csv(SUMMARY_PATH, index=False)

    # -----------------------------
    # ranked (IMPORTANT FIX)
    # -----------------------------
    ranked = summary.copy()

    ranked["rank_by_tradeoff"] = ranked.groupby(
        ["language", "model"]
    )["tradeoff_hmean_mean"].rank(ascending=False, method="dense")

    ranked["rank_by_gmean"] = ranked.groupby(
        ["language", "model"]
    )["tradeoff_gmean_mean"].rank(ascending=False, method="dense")

    ranked = ranked.sort_values(
        ["language", "model", "rank_by_tradeoff"]
    )

    ranked.to_csv(RANKED_PATH, index=False)

    # -----------------------------
    # best
    # -----------------------------
    best = ranked[ranked["rank_by_tradeoff"] == 1]
    best.to_csv(BEST_PATH, index=False)

    # -----------------------------
    # centroid only
    # -----------------------------
    centroid = summary[summary["estimator"] == "Centroid"]
    centroid.to_csv(CENTROID_PATH, index=False)

    # -----------------------------
    # estimator overall (FIXED)
    # -----------------------------
    estimator_overall = summary.groupby("estimator")[
        [
            "noun_gap_reduction_mean",
            "occ_gap_preservation_mean",
            "tradeoff_gmean_mean",
            "tradeoff_hmean_mean",   # ✅ FIX 3
        ]
    ].mean().reset_index()

    estimator_overall.to_csv(ESTIMATOR_PATH, index=False)

    # -----------------------------
    # context/weight overall
    # -----------------------------
    context_overall = summary.groupby("method")[
        [
            "noun_gap_reduction_mean",
            "occ_gap_preservation_mean",
            "tradeoff_gmean_mean",
            "tradeoff_hmean_mean",   # ✅ FIX 4
        ]
    ].mean().reset_index()

    context_overall.to_csv(CONTEXT_PATH, index=False)

    print("DONE: 02 summarize fixed (gmean + hmean fully consistent)")


if __name__ == "__main__":
    main()