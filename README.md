# Word+Character TF-IDF Provision Classification on LEDGAR

Motivation: Vietnamese legal question-answering systems rely on language-model answer generators, and the paper "Optimizing answer generator in Vietnamese legal question answering systems using language models" (2025) aims to make those answers better and more reliable. A first step toward traceable legal answers is to label each clause with its provision type and to check whether each label can be tied to supporting training text. This repository tests a cheap, fully reproducible classifier for that step.

Question: Can a class-balanced word-plus-character TF-IDF model with CV-tuned regularisation beat a standard TF-IDF logistic regression at assigning the correct provision type to legal clauses?

Dataset: LEDGAR contract-provision clauses from the `lex_glue` collection (CC-licensed), downloaded over HTTP from the Hugging Face datasets-server. This is not the paper's own dataset. A subsample of the official train and test partitions is split once, stratified, into train and test.

Baseline: word TF-IDF (1-2 grams) with an unweighted logistic regression. It is a stand-in for the paper's language-model answer generator, which is not public, and is not the paper's method.

Improved model: adds character n-gram TF-IDF, class-balanced weights, and a 3-fold cross-validated choice of C on the training set only.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results are written to results.json (and a chart to figures/).

## Reference

This project is motivated by the following paper:

> H Le, N Luu, T Nguyen, T Dao, S Dinh (2025). *Optimizing answer generator in Vietnamese legal question answering systems using language models*. ACM Transactions on Asian and. https://dl.acm.org/doi/abs/10.1145/3732938

Paper author: Prof. Le Thanh Huong, Hanoi University of Science and Technology (huonglt@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
