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

## Reference

This project is motivated by the following paper:

> H Le, N Luu, T Nguyen, T Dao, S Dinh (2025). *Optimizing answer generator in Vietnamese legal question answering systems using language models*. ACM Transactions on Asian and. https://dl.acm.org/doi/abs/10.1145/3732938

Paper author: Prof. Le Thanh Huong, Hanoi University of Science and Technology (huonglt@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
