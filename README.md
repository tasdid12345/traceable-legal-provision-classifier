# traceable-legal-provision-classifier

Compares a baseline (word TF-IDF + logistic regression) with an improved model
(word + char_wb TF-IDF, class-balanced logistic regression, C chosen by 3-fold CV on the
training set) for LEDGAR provision-type classification.

## Running

```
pip install -r requirements.txt
python analysis.py
```

Outputs `results.json` and `figures/baseline_vs_improved.png`.

## Data source

Rows are downloaded from the Hugging Face datasets-server. The original `lex_glue` id
returned HTTP 404 in CI, so the script now tries `coastalcph/lex_glue` and then the
Hub's auto-converted parquet files for the same `ledgar` config. The subsample (first
10,000 train and 3,500 test rows), the split, the baseline and the evaluation are unchanged.

## Results (latest run)

On the LEDGAR provision-type dataset (lex_glue/ledgar, subsampled), a standard TF-IDF logistic regression reached 63.4% macro-F1 on the held-out test split, while my class-balanced word-plus-character TF-IDF model with CV-tuned C reached 74.0%, from a single split.

| Metric | Value |
|---|---|
| baseline_score | 0.6342 |
| improved_score | 0.7399 |
| improvement | 0.1056 |
| baseline_f1 | 0.757 |
| improved_f1 | 0.8197 |
| baseline_accuracy | 0.7822 |
| improved_accuracy | 0.8252 |
| baseline_nn_agreement | 0.7677 |
| improved_nn_agreement | 0.8086 |
| improved_best_C | 30 |
| runtime_seconds | 480.4 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Subsampled to the first rows of the official partitions for the CI time budget; labels with fewer than 4 rows dropped. Data is fetched from datasets-server with a fallback to the Hub's parquet conversion if the original id returns 404. Both models are deterministic (lbfgs, seeded CV), so a single run with seed 42 is reported. Baseline C was not tuned, which is the intended 'standard' comparison. Nearest-neighbour agreement is a proxy for traceability, not a verified support check. Results are from one split and one subsample; differences may be small.

## Reference

This project is motivated by the following paper:

> H Le, N Luu, T Nguyen, T Dao, S Dinh (2025). *Optimizing answer generator in Vietnamese legal question answering systems using language models*. ACM Transactions on Asian and. https://dl.acm.org/doi/abs/10.1145/3732938

Paper author: Prof. Le Thanh Huong, Hanoi University of Science and Technology (huonglt@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
