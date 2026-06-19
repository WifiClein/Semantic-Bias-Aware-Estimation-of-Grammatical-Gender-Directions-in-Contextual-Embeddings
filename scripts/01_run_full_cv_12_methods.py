from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
ROOT = Path("/root/autodl-tmp/fin")
EMB_ROOT = ROOT / "data" / "embeddings"
OUT_ROOT = ROOT / "results" / "full_cv_12_methods"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_ROOT / "full_cv_12_methods_fold_results.csv"
PARTIAL_PATH = OUT_ROOT / "full_cv_12_methods_fold_results.partial.csv"

# ============================================================
# CONFIG
# ============================================================
EPS = 1e-12
W_MIN = 0.05
N_SPLITS = 5
RANDOM_STATE = 42

METHOD_CONFIGS = [
    {
        "method": "Controlled-unweighted",
        "context_source": "Controlled",
        "weighted": False,
        "weight_score": "none",
    },
    {
        "method": "Controlled-mean-weighted",
        "context_source": "Controlled",
        "weighted": True,
        "weight_score": "mean_abs",
    },
    {
        "method": "Controlled-median-weighted",
        "context_source": "Controlled",
        "weighted": True,
        "weight_score": "median_abs",
    },
    {
        "method": "Natural-unweighted",
        "context_source": "Natural",
        "weighted": False,
        "weight_score": "none",
    },
    {
        "method": "Natural-mean-weighted",
        "context_source": "Natural",
        "weighted": True,
        "weight_score": "mean_abs",
    },
    {
        "method": "Natural-median-weighted",
        "context_source": "Natural",
        "weighted": True,
        "weight_score": "median_abs",
    },
]

ESTIMATORS = [
    "Centroid",
    "LDA",
    "LinearSVC",
]


# ============================================================
# BASIC UTILS
# ============================================================
def l2_normalize(x, axis=-1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, EPS)


def unit_direction(x):
    return x / max(np.linalg.norm(x), EPS)


def clean_direction(raw_sem, gram_dir):
    gram_dir = unit_direction(gram_dir)
    cleaned = raw_sem - float(np.dot(raw_sem, gram_dir)) * gram_dir
    return unit_direction(cleaned)


def get_word_gender_columns(meta):
    """
    noun files use:
        noun, gender

    pair files use:
        word, side
    """
    if "noun" in meta.columns and "gender" in meta.columns:
        return "noun", "gender"
    if "word" in meta.columns and "side" in meta.columns:
        return "word", "side"

    raise ValueError(f"Cannot infer word/gender columns: {list(meta.columns)}")


# ============================================================
# RAW SEMANTIC GENDER DIRECTION
# ============================================================
def build_raw_semantic_direction(gender_meta, gender_emb):
    """
    Build raw semantic gender direction from gender_pairs:

        raw_sem = mean(masculine - feminine)
    """
    if "pair_id" not in gender_meta.columns:
        raise ValueError("gender_pairs.meta.csv must contain pair_id")
    if "side" not in gender_meta.columns:
        raise ValueError("gender_pairs.meta.csv must contain side")

    df = gender_meta.copy()
    df["_idx"] = np.arange(len(df))

    diffs = []

    for pair_id, g in df.groupby("pair_id"):
        masc = g[g["side"] == "masculine"]
        fem = g[g["side"] == "feminine"]

        if len(masc) == 0 or len(fem) == 0:
            continue

        if "template_id" in g.columns:
            for template_id, mg in masc.groupby("template_id"):
                fg = fem[fem["template_id"] == template_id]
                if len(fg) == 0:
                    continue

                m_idx = mg["_idx"].to_numpy()
                f_idx = fg["_idx"].to_numpy()
                k = min(len(m_idx), len(f_idx))

                for i in range(k):
                    diffs.append(gender_emb[m_idx[i]] - gender_emb[f_idx[i]])
        else:
            m_idx = masc["_idx"].to_numpy()
            f_idx = fem["_idx"].to_numpy()
            k = min(len(m_idx), len(f_idx))

            for i in range(k):
                diffs.append(gender_emb[m_idx[i]] - gender_emb[f_idx[i]])

    if len(diffs) == 0:
        raise ValueError("No raw semantic gender differences were constructed.")

    return unit_direction(np.mean(np.vstack(diffs), axis=0))


# ============================================================
# AGGREGATION AND ALIGNMENT
# ============================================================
def aggregate_to_item_level(meta, emb, file_name):
    """
    Aggregate contextual embeddings to item-level.

    For noun files:
        noun, gender -> word, side

    For pair files:
        word, side -> word, side
    """
    word_col, gender_col = get_word_gender_columns(meta)

    df = meta.copy()
    df["_idx"] = np.arange(len(df))

    rows = []
    vecs = []

    for (word, side), g in df.groupby([word_col, gender_col], sort=True):
        idx = g["_idx"].to_numpy()
        vec = emb[idx].mean(axis=0)

        rows.append({
            "word": word,
            "side": side,
        })
        vecs.append(vec)

    item_df = pd.DataFrame(rows)
    item_emb = l2_normalize(np.vstack(vecs))

    return item_df, item_emb


def align_noun_spaces(natural_df, natural_emb, controlled_df, controlled_emb):
    """
    Align natural and controlled noun-level spaces by noun+gender key.
    This is the single source of alignment.
    """
    natural_keys = [(r.word, r.side) for r in natural_df.itertuples(index=False)]
    controlled_keys = [(r.word, r.side) for r in controlled_df.itertuples(index=False)]

    natural_map = {k: i for i, k in enumerate(natural_keys)}
    controlled_map = {k: i for i, k in enumerate(controlled_keys)}

    common = sorted(set(natural_map.keys()) & set(controlled_map.keys()))

    if len(common) == 0:
        raise ValueError("No overlapping nouns between natural and controlled files.")

    if len(common) < len(natural_keys) or len(common) < len(controlled_keys):
        print(
            f"  WARNING: natural nouns={len(natural_keys)}, "
            f"controlled nouns={len(controlled_keys)}, common={len(common)}. "
            f"Using intersection."
        )

    noun_df = pd.DataFrame([
        {"word": k[0], "side": k[1]}
        for k in common
    ])

    natural_aligned = np.vstack([natural_emb[natural_map[k]] for k in common])
    controlled_aligned = np.vstack([controlled_emb[controlled_map[k]] for k in common])

    return noun_df, l2_normalize(natural_aligned), l2_normalize(controlled_aligned)


# ============================================================
# CONTAMINATION WEIGHTS
# ============================================================
def compute_contamination_scores(natural_meta, natural_emb, raw_sem, score_mode="mean_abs"):
    """
    Compute noun-level semantic-contamination scores from natural contexts.

    score_mode:
        mean_abs   : original mean absolute projection
        median_abs : robust median absolute projection
    """
    word_col, gender_col = get_word_gender_columns(natural_meta)

    df = natural_meta.copy()
    df["_idx"] = np.arange(len(df))

    scores = {}

    for (word, side), g in df.groupby([word_col, gender_col], sort=True):
        idx = g["_idx"].to_numpy()
        vals = np.abs(natural_emb[idx] @ raw_sem)

        if score_mode == "mean_abs":
            score = float(np.mean(vals))
        elif score_mode == "median_abs":
            score = float(np.median(vals))
        else:
            raise ValueError(f"Unknown contamination score mode: {score_mode}")

        scores[(word, side)] = score

    return scores


def make_weights(train_keys, contamination_scores, weighted=True):
    if not weighted:
        return {k: 1.0 for k in train_keys}

    vals = np.array([contamination_scores[k] for k in train_keys], dtype=float)

    vmin = vals.min()
    vmax = vals.max()

    if abs(vmax - vmin) < EPS:
        weights = np.ones_like(vals)
    else:
        normalized = (vals - vmin) / (vmax - vmin)
        weights = np.maximum(1.0 - normalized, W_MIN)

    return dict(zip(train_keys, weights))


# ============================================================
# GRAMMATICAL DIRECTION ESTIMATORS
# ============================================================
def orient_direction_to_centroid(d, train_df, train_emb):
    """
    Ensure estimator direction has the same sign as:
        masculine centroid - feminine centroid
    """
    sides = train_df["side"].to_numpy()

    m = train_emb[sides == "masculine"].mean(axis=0)
    f = train_emb[sides == "feminine"].mean(axis=0)

    ref = unit_direction(m - f)

    if np.dot(d, ref) < 0:
        d = -d

    return unit_direction(d)


def estimate_centroid_direction(train_df, train_emb, weights):
    m_vecs, f_vecs = [], []
    m_w, f_w = [], []

    for i, row in train_df.reset_index(drop=True).iterrows():
        key = (row["word"], row["side"])
        w = weights[key]

        if row["side"] == "masculine":
            m_vecs.append(train_emb[i])
            m_w.append(w)
        elif row["side"] == "feminine":
            f_vecs.append(train_emb[i])
            f_w.append(w)

    m_vecs = np.vstack(m_vecs)
    f_vecs = np.vstack(f_vecs)
    m_w = np.asarray(m_w)
    f_w = np.asarray(f_w)

    mu_m = np.average(m_vecs, axis=0, weights=m_w)
    mu_f = np.average(f_vecs, axis=0, weights=f_w)

    return unit_direction(mu_m - mu_f)


def estimate_lda_direction(train_df, train_emb, weights):
    """
    Manual weighted shrinkage LDA direction.

    d = Sigma_shrink^{-1} (mu_m - mu_f)

    This avoids sklearn LDA sample_weight compatibility issues.
    """
    sides = train_df["side"].to_numpy()

    sample_weight = np.array([
        weights[(row["word"], row["side"])]
        for _, row in train_df.iterrows()
    ], dtype=float)

    m_mask = sides == "masculine"
    f_mask = sides == "feminine"

    X_m = train_emb[m_mask]
    X_f = train_emb[f_mask]

    w_m = sample_weight[m_mask]
    w_f = sample_weight[f_mask]

    mu_m = np.average(X_m, axis=0, weights=w_m)
    mu_f = np.average(X_f, axis=0, weights=w_f)

    delta = mu_m - mu_f

    X_centered = train_emb.copy()
    X_centered[m_mask] = X_centered[m_mask] - mu_m
    X_centered[f_mask] = X_centered[f_mask] - mu_f

    w = sample_weight / np.maximum(sample_weight.sum(), EPS)

    cov = (X_centered * w[:, None]).T @ X_centered

    alpha = 0.1
    tau = np.trace(cov) / cov.shape[0]
    cov_shrink = (1.0 - alpha) * cov + alpha * tau * np.eye(cov.shape[0])

    try:
        d = np.linalg.solve(cov_shrink, delta)
    except np.linalg.LinAlgError:
        d = np.linalg.pinv(cov_shrink) @ delta

    return orient_direction_to_centroid(d, train_df, train_emb)


def estimate_linearsvc_direction(train_df, train_emb, weights):
    y = (train_df["side"].to_numpy() == "masculine").astype(int)

    sample_weight = np.array([
        weights[(row["word"], row["side"])]
        for _, row in train_df.iterrows()
    ], dtype=float)

    clf = LinearSVC(
        C=1.0,
        class_weight=None,
        max_iter=20000,
        random_state=RANDOM_STATE,
        dual=False,
    )

    clf.fit(train_emb, y, sample_weight=sample_weight)

    d = clf.coef_.reshape(-1)

    return orient_direction_to_centroid(d, train_df, train_emb)


def estimate_gram_direction(estimator, train_df, train_emb, weights):
    if estimator == "Centroid":
        return estimate_centroid_direction(train_df, train_emb, weights)
    if estimator == "LDA":
        return estimate_lda_direction(train_df, train_emb, weights)
    if estimator == "LinearSVC":
        return estimate_linearsvc_direction(train_df, train_emb, weights)

    raise ValueError(f"Unknown estimator: {estimator}")


# ============================================================
# EVALUATION
# ============================================================
def gender_gap(item_df, item_emb, idx, d):
    sub_df = item_df.iloc[idx].reset_index(drop=True)
    sub_emb = item_emb[idx]

    proj = sub_emb @ d
    sides = sub_df["side"].to_numpy()

    m = proj[sides == "masculine"]
    f = proj[sides == "feminine"]

    if len(m) == 0 or len(f) == 0:
        raise ValueError("Gap calculation requires both masculine and feminine items.")

    return float(m.mean() - f.mean())


def occupation_gap(occupation_meta, occupation_emb, d):
    occ_df, occ_emb_item = aggregate_to_item_level(
        occupation_meta,
        occupation_emb,
        "occupation_pairs.meta.csv",
    )

    idx = np.arange(len(occ_df))
    return gender_gap(occ_df, occ_emb_item, idx, d)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    all_rows = []

    for lang_dir in sorted(EMB_ROOT.iterdir()):
        if not lang_dir.is_dir():
            continue

        language = lang_dir.name

        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            # skip non-model dirs if any
            if model_dir.name.startswith("."):
                continue

            model = model_dir.name

            print("=" * 100)
            print(f"Running: {language} / {model}")

            controlled_meta = pd.read_csv(model_dir / "controlled_nouns.meta.csv")
            controlled_emb = l2_normalize(np.load(model_dir / "controlled_nouns.npy"))

            natural_meta = pd.read_csv(model_dir / "natural_nouns.meta.csv")
            natural_emb = l2_normalize(np.load(model_dir / "natural_nouns.npy"))

            gender_meta = pd.read_csv(model_dir / "gender_pairs.meta.csv")
            gender_emb = l2_normalize(np.load(model_dir / "gender_pairs.npy"))

            occupation_meta = pd.read_csv(model_dir / "occupation_pairs.meta.csv")
            occupation_emb = l2_normalize(np.load(model_dir / "occupation_pairs.npy"))

            raw_sem = build_raw_semantic_direction(gender_meta, gender_emb)

            natural_df, natural_noun_emb = aggregate_to_item_level(
                natural_meta,
                natural_emb,
                "natural_nouns.meta.csv",
            )

            controlled_df, controlled_noun_emb = aggregate_to_item_level(
                controlled_meta,
                controlled_emb,
                "controlled_nouns.meta.csv",
            )

            noun_df, natural_noun_emb, controlled_noun_emb = align_noun_spaces(
                natural_df,
                natural_noun_emb,
                controlled_df,
                controlled_noun_emb,
            )

            contamination_by_score = {
            "mean_abs": compute_contamination_scores(
                natural_meta,
                natural_emb,
                raw_sem,
                score_mode="mean_abs",
            ),
            "median_abs": compute_contamination_scores(
                natural_meta,
                natural_emb,
                raw_sem,
                score_mode="median_abs",
            ),
        }

            print(f"  Number of aligned nouns: {len(noun_df)}")
            print(f"  Masculine nouns: {(noun_df['side'] == 'masculine').sum()}")
            print(f"  Feminine nouns: {(noun_df['side'] == 'feminine').sum()}")

            y = noun_df["side"].map({
                "masculine": 0,
                "feminine": 1,
            }).to_numpy()

            if np.any(pd.isna(y)):
                raise ValueError(
                    f"Unexpected gender labels in {language}/{model}: "
                    f"{noun_df['side'].unique()}"
                )

            skf = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            raw_occ_gap_cache = abs(
                occupation_gap(occupation_meta, occupation_emb, raw_sem)
            )

            for fold, (train_idx, test_idx) in enumerate(skf.split(noun_df, y), start=1):
                train_df = noun_df.iloc[train_idx].reset_index(drop=True)

                train_keys = [
                    (row["word"], row["side"])
                    for _, row in train_df.iterrows()
                ]

                for cfg in METHOD_CONFIGS:
                    method = cfg["method"]
                    context_source = cfg["context_source"]
                    weighted = cfg["weighted"]

                    if weighted:
                        contamination = contamination_by_score[cfg["weight_score"]]
                    else:
                        contamination = {k: 0.0 for k in train_keys}
                    
                    weights = make_weights(
                        train_keys,
                        contamination,
                        weighted=weighted,
                    )

                    if context_source == "Natural":
                        source_emb = natural_noun_emb
                    elif context_source == "Controlled":
                        source_emb = controlled_noun_emb
                    else:
                        raise ValueError(context_source)

                    train_emb = source_emb[train_idx]

                    for estimator in ESTIMATORS:
                        gram_dir = estimate_gram_direction(
                            estimator,
                            train_df,
                            train_emb,
                            weights,
                        )

                        clean_sem = clean_direction(raw_sem, gram_dir)

                        raw_noun_gap = abs(
                            gender_gap(noun_df, source_emb, test_idx, raw_sem)
                        )
                        clean_noun_gap = abs(
                            gender_gap(noun_df, source_emb, test_idx, clean_sem)
                        )

                        noun_reduction = 1.0 - clean_noun_gap / max(raw_noun_gap, EPS)

                        clean_occ_gap = abs(
                            occupation_gap(occupation_meta, occupation_emb, clean_sem)
                        )

                        occ_preservation = clean_occ_gap / max(raw_occ_gap_cache, EPS)

                        nr = max(noun_reduction, 0.0)
                        op = max(occ_preservation, 0.0)
                        
                        tradeoff_gmean = np.sqrt(nr * op)
                        tradeoff_hmean = 2.0 * nr * op / max(nr + op, EPS)

                        all_rows.append({
                            "language": language,
                            "model": model,
                            "fold": fold,
                            "method": method,
                            "context_source": context_source,
                            "weighted": weighted,
                            "estimator": estimator,
                            "weight_score": cfg["weight_score"],

                            "cos_raw_sem_gram": float(np.dot(raw_sem, gram_dir)),
                            "cos_raw_sem_clean": float(np.dot(raw_sem, clean_sem)),
                            "abs_cos_clean_gram": abs(float(np.dot(clean_sem, gram_dir))),

                            "raw_noun_gap": raw_noun_gap,
                            "clean_noun_gap": clean_noun_gap,
                            "noun_gap_reduction": noun_reduction,

                            "raw_occ_gap": raw_occ_gap_cache,
                            "clean_occ_gap": clean_occ_gap,
                            "occ_gap_preservation": occ_preservation,

                            "tradeoff_gmean": tradeoff_gmean,
                            "tradeoff_hmean": tradeoff_hmean,
                        })

            partial = pd.DataFrame(all_rows)
            partial.to_csv(PARTIAL_PATH, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print("=" * 100)
    print(f"Saved fold-level results to: {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()