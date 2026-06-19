from pathlib import Path
import pandas as pd

ROOT = Path("/root/autodl-tmp/fin")
TABLE_DIR = ROOT / "results" / "tables"

CENTROID_CSV = TABLE_DIR / "centroid_methods_only.csv"
ESTIMATOR_CSV = TABLE_DIR / "estimator_overall_comparison.csv"
EXTERNAL_CSV = TABLE_DIR / "external_semantic_4_centroids_summary.csv"

CENTROID_TEX = TABLE_DIR / "centroid_methods_only.tex"
ESTIMATOR_TEX = TABLE_DIR / "estimator_overall_comparison.tex"
EXTERNAL_TEX = TABLE_DIR / "external_semantic_stability.tex"


def fmt3(x):
    return f"{float(x):.3f}"


def fmt4(x):
    return f"{float(x):.4f}"


def fmt_delta(x):
    return f"{float(x):+.4f}"


def bold(x):
    return r"\textbf{" + x + "}"


def underline(x):
    return r"\underline{" + x + "}"


def decorate_by_group(df, group_cols, metric_col, higher_is_better=True, fmt=fmt3):
    out = pd.Series(index=df.index, dtype=object)

    if not group_cols:
        groups = [(None, df)]
    else:
        groups = df.groupby(group_cols, sort=False)

    for _, g in groups:
        vals = g[metric_col].astype(float)
        order = vals.sort_values(ascending=not higher_is_better).index.tolist()

        best = order[0] if len(order) >= 1 else None
        second = order[1] if len(order) >= 2 else None

        for idx in g.index:
            s = fmt(df.loc[idx, metric_col])
            if idx == best:
                s = bold(s)
            elif idx == second:
                s = underline(s)
            out.loc[idx] = s

    return out


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


def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def generate_centroid_table():
    df = pd.read_csv(CENTROID_CSV)

    if "base_model" not in df.columns:
        df["base_model"] = df["model"].apply(normalize_model)
    else:
        df["base_model"] = df["base_model"].apply(normalize_model)

    needed = [
        "language",
        "base_model",
        "method",
        "noun_gap_reduction_mean",
        "occ_gap_preservation_mean",
        "tradeoff_gmean_mean",
        "tradeoff_hmean_mean",
    ]
    require_cols(df, needed, "centroid_methods_only.csv")
    df = df[needed].copy()

    method_order = {
        "Controlled-unweighted": 0,
        "Controlled-mean-weighted": 1,
        "Controlled-median-weighted": 2,
        "Natural-unweighted": 3,
        "Natural-mean-weighted": 4,
        "Natural-median-weighted": 5,
    }
    model_order = {
        "BETO": 0,
        "mBERT": 1,
        "XLM-R": 2,
        "CamemBERT": 0,
    }
    lang_order = {
        "spanish": 0,
        "french": 1,
    }

    df["_lang_order"] = df["language"].map(lang_order).fillna(9)
    df["_model_order"] = df["base_model"].map(model_order).fillna(9)
    df["_method_order"] = df["method"].map(method_order).fillna(9)

    df = df.sort_values(
        ["_lang_order", "language", "_model_order", "base_model", "_method_order"]
    ).reset_index(drop=True)

    group_cols = ["language", "base_model"]

    # Noun / occupation 按各自列加粗
    df["noun_fmt"] = decorate_by_group(
        df, group_cols, "noun_gap_reduction_mean", higher_is_better=True, fmt=fmt3
    )
    df["occ_fmt"] = decorate_by_group(
        df, group_cols, "occ_gap_preservation_mean", higher_is_better=True, fmt=fmt3
    )

    # G/H tradeoff 都按 hmean 排名标注，因为 hmean 是主选择指标
    df["gmean_fmt"] = df["tradeoff_gmean_mean"].apply(fmt3)
    df["hmean_fmt"] = decorate_by_group(
        df, group_cols, "tradeoff_hmean_mean", higher_is_better=True, fmt=fmt3
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Centroid-based context-source and weighting ablation. "
        r"Best results are bolded and second-best results are underlined within each language--model block. "
        r"The primary trade-off score is the harmonic trade-off.}"
    )
    lines.append(r"\label{tab:centroid_context_ablation}")
    lines.append(r"\begin{tabular}{lllcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Language & Model & Method & "
        r"Noun red. $\uparrow$ & Occ. pres. $\uparrow$ & "
        r"G-Trade-off $\uparrow$ & H-Trade-off $\uparrow$ \\"
    )
    lines.append(r"\midrule")

    first_lang = True
    for language, lang_g in df.groupby("language", sort=False):
        if not first_lang:
            lines.append(r"\midrule")
        first_lang = False

        lang_rows_total = len(lang_g)
        lang_printed = False

        model_keys = list(lang_g.groupby("base_model", sort=False).groups.keys())

        for model_i, (model, model_g) in enumerate(lang_g.groupby("base_model", sort=False)):
            model_rows_total = len(model_g)
            model_printed = False

            for _, row in model_g.iterrows():
                lang_cell = (
                    rf"\multirow{{{lang_rows_total}}}{{*}}{{{language.capitalize()}}}"
                    if not lang_printed
                    else ""
                )
                model_cell = (
                    rf"\multirow{{{model_rows_total}}}{{*}}{{{model}}}"
                    if not model_printed
                    else ""
                )

                lines.append(
                    f"{lang_cell} & {model_cell} & {row['method']} & "
                    f"{row['noun_fmt']} & {row['occ_fmt']} & "
                    f"{row['gmean_fmt']} & {row['hmean_fmt']} \\\\"
                )

                lang_printed = True
                model_printed = True

            if model_i != len(model_keys) - 1:
                lines.append(r"\cmidrule(lr){2-7}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    CENTROID_TEX.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", CENTROID_TEX)


def generate_estimator_table():
    df = pd.read_csv(ESTIMATOR_CSV)

    needed = [
        "estimator",
        "noun_gap_reduction_mean",
        "occ_gap_preservation_mean",
        "tradeoff_gmean_mean",
        "tradeoff_hmean_mean",
    ]
    require_cols(df, needed, "estimator_overall_comparison.csv")

    df = df[needed].copy()
    df = df.sort_values("tradeoff_hmean_mean", ascending=False).reset_index(drop=True)

    df["noun_fmt"] = decorate_by_group(
        df, [], "noun_gap_reduction_mean", higher_is_better=True, fmt=fmt3
    )
    df["occ_fmt"] = decorate_by_group(
        df, [], "occ_gap_preservation_mean", higher_is_better=True, fmt=fmt3
    )
    df["gmean_fmt"] = df["tradeoff_gmean_mean"].apply(fmt3)
    df["hmean_fmt"] = decorate_by_group(
        df, [], "tradeoff_hmean_mean", higher_is_better=True, fmt=fmt3
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Overall comparison of grammatical gender direction estimators, "
        r"averaged across languages, models, context sources, and weighting variants. "
        r"The primary trade-off score is the harmonic trade-off.}"
    )
    lines.append(r"\label{tab:estimator_overall}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Estimator & Noun red. $\uparrow$ & Occ. pres. $\uparrow$ & "
        r"G-Trade-off $\uparrow$ & H-Trade-off $\uparrow$ \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{row['estimator']} & {row['noun_fmt']} & {row['occ_fmt']} & "
            f"{row['gmean_fmt']} & {row['hmean_fmt']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    ESTIMATOR_TEX.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", ESTIMATOR_TEX)


def generate_external_table():
    df = pd.read_csv(EXTERNAL_CSV)

    needed = [
        "method",
        "raw_alignment_mean",
        "raw_alignment_std",
        "clean_alignment_mean",
        "clean_alignment_std",
        "delta_mean",
        "delta_std",
    ]
    require_cols(df, needed, "external_semantic_4_centroids_summary.csv")
    df = df[needed].copy()

    method_order = {
        "Controlled-unweighted": 0,
        "Controlled-weighted": 1,
        "Natural-unweighted": 2,
        "Natural-weighted": 3,
    }
    df["_order"] = df["method"].map(method_order).fillna(9)
    df = df.sort_values("_order").reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{External semantic stability of centroid-based grammatical projection. "
        r"Values report mean noun-level natural--controlled alignment before and after projection. "
        r"Delta is computed as Cleaned minus Raw.}"
    )
    lines.append(r"\label{tab:external_semantic_stability}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Raw alignment & Cleaned alignment & Delta \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        raw = f"{fmt4(row['raw_alignment_mean'])} $\\pm$ {fmt4(row['raw_alignment_std'])}"
        clean = f"{fmt4(row['clean_alignment_mean'])} $\\pm$ {fmt4(row['clean_alignment_std'])}"
        delta = f"{fmt_delta(row['delta_mean'])} $\\pm$ {fmt4(row['delta_std'])}"

        lines.append(
            f"{row['method']} & {raw} & {clean} & {delta} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    EXTERNAL_TEX.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", EXTERNAL_TEX)


def main():
    generate_centroid_table()
    generate_estimator_table()
    generate_external_table()


if __name__ == "__main__":
    main()