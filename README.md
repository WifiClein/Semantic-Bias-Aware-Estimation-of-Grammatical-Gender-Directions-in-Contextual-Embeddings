# Semantic-Bias-Aware Estimation of Grammatical Gender Directions in Contextual Embeddings

This repository contains code and result tables for estimating grammatical gender directions in Spanish and French contextual embeddings under contextual semantic contamination.

Large embedding files are not stored in this GitHub repository. They will be released separately via Hugging Face Datasets.

## Repository structure

```text
scripts/
results/tables/
requirements.txt
README.md


Data
The expected embedding data structure is:
data/embeddings/french/french_camembert/
data/embeddings/french/multilingual_mbert/
data/embeddings/french/multilingual_xlmr/
data/embeddings/spanish/spanish_beto/
data/embeddings/spanish/multilingual_mbert/
data/embeddings/spanish/multilingual_xlmr/


Reproduction pipeline
After downloading and extracting the embedding data, run:
python scripts/01_run_full_cv_12_methods.py
python scripts/02_summarize_results.py
python scripts/03_external_semantic_eval_4_centroids.py
python scripts/04_generate_latex_tables.py
python scripts/05_generate_weight_sensitivity_table.py
Main outputs

The main result tables are stored in:
results/tables/
