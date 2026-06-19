from pathlib import Path
import pandas as pd

ROOT = Path("/root/autodl-tmp/fin")
TABLE_DIR = ROOT / "results" / "tables"

IN_PATH = TABLE_DIR / "centroid_methods_only.csv"

OUT_CSV = TABLE_DIR / "weight_sensitivity_analysis.csv"
OUT_SUMMARY_CSV = TABLE_DIR / "weight_sensitivity_analysis_summary.csv"
OUT_TEX = TABLE_DIR / "weight_sensitivity_analysis.tex"


def fmt_delta(x):
    return f"{float(x):+.3f}"


def fmt_pm(mean, std):
    if pd.isna(std):
        std = 0.0
    return f"{fmt_delta(mean)} $\\pm$ {float(std):.3f}"


def normalize_model(model):
    mapping = {
        "french_camembert": "CamemBERT",
        "spanish_beto": "BETO",
        "multilingual_mbert": "mBERT",
        "multilingual_xlmr": "XLM-R",
        "CamemBERT": "CamemBERT",
        "BETO": "BETO",
        "mBERT": "mBERT",
        "XLM-R": "XLM-R",
    }
    return mapping.get(str(model), str(model))


def parse_method(method):
    """
    Expected method names:
        Controlled-unweighted
        Controlled-mean-weighted
        Controlled-median-weighted
        Natural-unweighted
        Natural-mean-weighted
        Natural-median-weighted
    """
    method = str(method)

    if method.startswith("Controlled"):
        context = "Controlled"
    elif method.startswith("Natural"):
        context = "Natural"
    else:
        raise ValueError(f"Cannot infer context from method: {method}")

    if method.endswith("unweighted"):
        weight_variant = "unweighted"
    elif "mean-weighted" in method:
        weight_variant = "mean"
    elif "median-weighted" in method:
        weight_variant = "median"
    elif method.endswith("weighted"):
        weight_variant = "mean"
    else:
        raise ValueError(f"Cannot infer weight variant from method: {method}")

    return context, weight_variant


def main():
    df = pd.read_csv(IN_PATH)

    required = {
        "language",
        "model",
        "method",
        "noun_gap_reduction_mean",
        "occ_gap_preservation_mean",
        "tradeoff_gmean_mean",
        "tradeoff_hmean_mean",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {IN_PATH}: {missing}")

    if "base_model" not in df.columns:
        df["base_model"] = df["model"].apply(normalize_model)
    else:
        df["base_model"] = df["base_model"].apply(normalize_model)

    parsed = df["method"].apply(parse_method)
    df["context"] = parsed.apply(lambda x: x[0])
    df["weight_variant"] = parsed.apply(lambda x: x[1])

    rows = []

    group_cols = ["language", "model", "base_model", "context"]

    for keys, g in df.groupby(group_cols, sort=False):
        language, model, base_model, context = keys

        unweighted = g[g["weight_variant"] == "unweighted"]

        if len(unweighted) != 1:
            print(f"Skipping group without exactly one unweighted baseline: {keys}")
            continue

        u = unweighted.iloc[0]

        for variant in ["mean", "median"]:
            weighted = g[g["weight_variant"] == variant]

            if len(weighted) != 1:
                print(f"Skipping missing {variant}-weighted group: {keys}")
                continue

            w = weighted.iloc[0]

            rows.append({
                "language": language,
                "model": model,
                "base_model": base_model,
                "context": context,
                "weight_variant": variant,

                "delta_noun_reduction": (
                    w["noun_gap_reduction_mean"] - u["noun_gap_reduction_mean"]
                ),
                "delta_occ_preservation": (
                    w["occ_gap_preservation_mean"] - u["occ_gap_preservation_mean"]
                ),
                "delta_gtradeoff": (
                    w["tradeoff_gmean_mean"] - u["tradeoff_gmean_mean"]
                ),
                "delta_htradeoff": (
                    w["tradeoff_hmean_mean"] - u["tradeoff_hmean_mean"]
                ),
            })

    if not rows:
        raise RuntimeError(
            "No valid weight sensitivity rows were produced. "
            "Check method names in centroid_methods_only.csv."
        )

    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(OUT_CSV, index=False)

    summary = (
        delta_df.groupby(["context", "weight_variant"])[
            [
                "delta_noun_reduction",
                "delta_occ_preservation",
                "delta_gtradeoff",
                "delta_htradeoff",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in summary.columns
    ]

    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    context_order = {"Controlled": 0, "Natural": 1}
    variant_order = {"mean": 0, "median": 1}

    summary["_context_order"] = summary["context"].map(context_order).fillna(9)
    summary["_variant_order"] = summary["weight_variant"].map(variant_order).fillna(9)
    summary = summary.sort_values(
        ["_context_order", "_variant_order"]
    ).reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Weight sensitivity analysis under the centroid estimator. "
        r"Values report the average paired difference between weighted and unweighted variants "
        r"within the same language--model and context-source block. "
        r"Positive values indicate that weighting increases the metric.}"
    )
    lines.append(r"\label{tab:weight_sensitivity}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Context source & Weight score & $\Delta$ Noun red. & $\Delta$ Occ. pres. "
        r"& $\Delta$ G-Trade-off & $\Delta$ H-Trade-off \\"
    )
    lines.append(r"\midrule")

    for _, row in summary.iterrows():
        lines.append(
            f"{row['context']} & {row['weight_variant']} & "
            f"{fmt_pm(row['delta_noun_reduction_mean'], row['delta_noun_reduction_std'])} & "
            f"{fmt_pm(row['delta_occ_preservation_mean'], row['delta_occ_preservation_std'])} & "
            f"{fmt_pm(row['delta_gtradeoff_mean'], row['delta_gtradeoff_std'])} & "
            f"{fmt_pm(row['delta_htradeoff_mean'], row['delta_htradeoff_std'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print(OUT_CSV)
    print(OUT_SUMMARY_CSV)
    print(OUT_TEX)

    print("\nSummary:")
    print(
        summary.drop(columns=["_context_order", "_variant_order"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()