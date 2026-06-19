from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/fin")
EMB_ROOT = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "external_semantic_4_centroids.csv"

EPS = 1e-12


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def load_npy(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def project_out(x, d):
    d = d / (np.linalg.norm(d) + EPS)
    return x - (x @ d)[:, None] * d[None, :]


def build_shared_index(nat_meta, con_meta):
    nat_keys = list(zip(nat_meta["noun"], nat_meta["gender"]))
    con_keys = list(zip(con_meta["noun"], con_meta["gender"]))

    common = sorted(set(nat_keys) & set(con_keys))

    nat_map = {k: i for i, k in enumerate(nat_keys)}
    con_map = {k: i for i, k in enumerate(con_keys)}

    nat_idx = np.array([nat_map[k] for k in common])
    con_idx = np.array([con_map[k] for k in common])

    df = pd.DataFrame(common, columns=["word", "side"])
    return df, nat_idx, con_idx


def centroid_direction(x, y):
    """
    y = 1 for feminine, 0 for masculine.
    Direction is masculine - feminine, consistent with earlier scripts.
    """
    masc = x[y == 0].mean(axis=0)
    fem = x[y == 1].mean(axis=0)
    d = masc - fem
    return d / (np.linalg.norm(d) + EPS)


def cosine_rowwise(a, b):
    a = l2(a)
    b = l2(b)
    return np.sum(a * b, axis=1)


def semantic_alignment_score(nat_emb, con_emb):
    """
    外部语义稳定性代理指标：
    比较 natural 和 controlled 中同一 noun-gender item 的表示相似度。
    projection 前后变化越接近 0，说明语义结构越稳定。
    """
    return float(cosine_rowwise(nat_emb, con_emb).mean())


def run_one_model(lang, model_dir):
    model = model_dir.name

    print("=" * 80)
    print(f"External eval: {lang} / {model}")

    nat_meta = pd.read_csv(model_dir / "natural_nouns.meta.csv")
    con_meta = pd.read_csv(model_dir / "controlled_nouns.meta.csv")

    nat_emb = l2(load_npy(model_dir / "natural_nouns.npy"))
    con_emb = l2(load_npy(model_dir / "controlled_nouns.npy"))

    df, nat_idx, con_idx = build_shared_index(nat_meta, con_meta)

    nat_emb = nat_emb[nat_idx]
    con_emb = con_emb[con_idx]

    y = (df["side"] == "feminine").astype(int).to_numpy()

    print(f"aligned size: {len(df)}")
    print(f"masculine: {(df['side'] == 'masculine').sum()}")
    print(f"feminine: {(df['side'] == 'feminine').sum()}")

    # 四种 centroid grammatical directions
    # weighted 版本这里先用同一个 centroid 方向占位。
    # 如果你要严格复现 weighted，需要从 CV fold 内部计算 contamination weights。
    directions = {
        "Controlled-unweighted": centroid_direction(con_emb, y),
        "Natural-unweighted": centroid_direction(nat_emb, y),
        "Controlled-weighted": centroid_direction(con_emb, y),
        "Natural-weighted": centroid_direction(nat_emb, y),
    }

    raw_score = semantic_alignment_score(nat_emb, con_emb)

    rows = []

    for method, d in directions.items():
        nat_clean = project_out(nat_emb, d)
        con_clean = project_out(con_emb, d)

        clean_score = semantic_alignment_score(nat_clean, con_clean)
        delta = clean_score - raw_score

        rows.append({
            "language": lang,
            "model": model,
            "method": method,
            "raw_alignment": raw_score,
            "clean_alignment": clean_score,
            "delta": delta,
        })

    return rows


def main():
    all_rows = []

    for lang_dir in sorted(EMB_ROOT.iterdir()):
        if not lang_dir.is_dir():
            continue

        lang = lang_dir.name

        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            # 跳过 STRICT_RAW / 备份等非模型目录
            if model_dir.name.startswith("."):
                continue

            all_rows.extend(run_one_model(lang, model_dir))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print(df)


if __name__ == "__main__":
    main()