# Semantic-Bias-Aware Estimation of Grammatical Gender Directions in Contextual Embeddings

This repository contains code and result tables for estimating grammatical gender directions in Spanish and French contextual embeddings under contextual semantic contamination.

Large embedding files are not stored in this GitHub repository. They will be released separately via Hugging Face Datasets.

## Reproduction pipeline

After downloading the embedding data, run:

```bash
python scripts/01_run_full_cv_12_methods.py
python scripts/02_summarize_results.py
python scripts/03_external_semantic_eval_4_centroids.py
python scripts/04_generate_latex_tables.py
python scripts/05_generate_weight_sensitivity_table.py
results/tables/
data/embeddings/french/french_camembert/
data/embeddings/french/multilingual_mbert/
data/embeddings/french/multilingual_xlmr/
data/embeddings/spanish/spanish_beto/
data/embeddings/spanish/multilingual_mbert/
data/embeddings/spanish/multilingual_xlmr/

### 2. 创建 requirements.txt

```bash
cat > requirements.txt <<'EOF'
numpy
pandas
scikit-learn
scipy
