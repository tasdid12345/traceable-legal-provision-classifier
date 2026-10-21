"""Baseline (word TF-IDF + logistic regression) vs improved (word+char TF-IDF,
class-balanced LR, CV-tuned C) for LEDGAR provision-type classification."""
import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless rendering on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
N_TRAIN_FETCH, N_TEST_FETCH = 10000, 3500   # subsample of official partitions (CI budget)
BASE = "https://datasets-server.huggingface.co/rows"
# The datasets-server returned 404 for the "lex_glue" id, so we try the other
# known ids and finally the Hub's auto-converted parquet files.
DATASET_IDS = ["lex_glue", "coastalcph/lex_glue"]
PARQUET_BASE = ("https://huggingface.co/datasets/coastalcph/lex_glue/"
                "resolve/refs%2Fconvert%2Fparquet/ledgar")
DATASET_URL = "https://huggingface.co/datasets/lex_glue"
RETRY_CODES = {429, 500, 502, 503, 504}
np.random.seed(SEED)


def _fetch_rows_api(dataset_id, split, n):
    """Rows via datasets-server (<=100 rows/request). Raises HTTPError on failure."""
    rows, offset = [], 0
    while offset < n:
        url = (f"{BASE}?dataset={dataset_id}&config=ledgar&split={split}"
               f"&offset={offset}&length={min(100, n - offset)}")
        for attempt in range(6):  # retry with exponential backoff on rate limits / server errors
            r = requests.get(url, timeout=60)
            if r.status_code not in RETRY_CODES:
                break
            print(f"  HTTP {r.status_code}, retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
        r.raise_for_status()
        batch = r.json()["rows"]
        if not batch:
            break
        rows += [(b["row"]["text"], int(b["row"]["label"])) for b in batch]
        offset += len(batch)
    return rows


def _fetch_rows_parquet(split, n):
    """Rows from the Hub's auto-converted parquet shards (same first-n-rows subsample)."""
    rows, shard = [], 0
    while len(rows) < n:
        url = f"{PARQUET_BASE}/{split}/{shard:04d}.parquet"
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            break
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
        rows += list(zip(df["text"].astype(str), df["label"].astype(int)))
        shard += 1
    return rows[:n]


def fetch_split(split, n):
    """Download the first n rows of a LEDGAR split, trying each source in turn."""
    rows = []
    for ds in DATASET_IDS:
        try:
            rows = _fetch_rows_api(ds, split, n)
            if rows:
                print(f"  {split}: datasets-server id '{ds}'")
                break
        except requests.HTTPError as e:
            print(f"  datasets-server id '{ds}' failed ({e}); trying next source")
    if not rows:
        print("  falling back to parquet files on the Hub")
        rows = _fetch_rows_parquet(split, n)
    print(f"  {split}: fetched {len(rows)} rows")
    return rows


def nn_agree(Q, R, y_ref, pred):
    """Share of predictions whose nearest training clause (cosine on word TF-IDF) has the same label."""
    nn = np.empty(Q.shape[0], dtype=int)
    for s in range(0, Q.shape[0], 500):  # chunked to bound memory
        nn[s:s + 500] = (Q[s:s + 500] @ R.T).toarray().argmax(axis=1)  # rows are L2-normalised
    return float(np.mean(y_ref[nn] == pred))


def evaluate(name, pred, yte, Wte, Wtr, ytr):
    """Single evaluation function used for both models."""
    res = {"macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
           "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
           "accuracy": accuracy_score(yte, pred),
           "nn_agreement": nn_agree(Wte, Wtr, ytr, pred)}
    print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items()))
    return {k: float(v) for k, v in res.items()}


def main():
    t0 = time.time()
    print("Downloading LEDGAR rows...")
    rows = fetch_split("train", N_TRAIN_FETCH) + fetch_split("test", N_TEST_FETCH)
    if len(rows) < 1000:
        sys.exit("Too few rows downloaded; cannot produce meaningful metrics")
    X = np.array([t for t, _ in rows], dtype=object)
    y = np.array([l for _, l in rows])

    # Keep labels with >=4 rows so the stratified split and 3-fold CV are valid.
    labels, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, labels[counts >= 4])
    X, y = X[keep], y[keep]

    # Single deterministic stratified split (25% test); the test set is untouched until the end.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"Rows kept: {len(y)} (classes: {len(np.unique(y))}), train {len(ytr)}, test {len(yte)}")

    # ---- Baseline: word TF-IDF + unweighted multinomial logistic regression ----
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000)
    Wtr, Wte = wv.fit_transform(Xtr), wv.transform(Xte)
    base = LogisticRegression(max_iter=200, random_state=SEED).fit(Wtr, ytr)
    print("Baseline trained.")

    # ---- Improved: word + char TF-IDF, balanced weights, CV-tuned C (train folds only) ----
    cv_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True, max_features=30000)
    Itr = hstack([Wtr, cv_.fit_transform(Xtr)]).tocsr()
    Ite = hstack([Wte, cv_.transform(Xte)]).tocsr()
    search = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=200, random_state=SEED),
        {"C": [3.0, 30.0]}, scoring="f1_macro", n_jobs=-1,
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    search.fit(Itr, ytr)
    print(f"Improved: best C by 3-fold CV on train = {search.best_params_['C']}")

    # ---- Test set used once, for both models ----
    b = evaluate("Baseline", base.predict(Wte), yte, Wte, Wtr, ytr)
    i = evaluate("Improved", search.predict(Ite), yte, Wte, Wtr, ytr)
    if not all(np.isfinite([b["macro_f1"], i["macro_f1"]])):
        sys.exit(1)

    # ---- Outputs ----
    os.makedirs("figures", exist_ok=True)
    names = ["macro_f1", "accuracy", "nn_agreement"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, [b[k] for k in names], 0.4, label="baseline (word TF-IDF LR)")
    ax.bar(x + 0.2, [i[k] for k in names], 0.4, label="improved (word+char, balanced, CV-C)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("LEDGAR test metrics")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)

    results = {
        "question": ("Can a class-balanced word-plus-character TF-IDF model with CV-tuned "
                     "regularisation beat a standard TF-IDF logistic regression at assigning "
                     "the correct provision type to legal clauses?"),
        "method": ("Stratified 75/25 split of subsampled LEDGAR train+test partitions; identical "
                   "data, preprocessing and evaluation for both models; macro-F1 primary, "
                   "accuracy and nearest-neighbour label agreement secondary."),
        "dataset": {"name": "lex_glue/ledgar", "source_url": DATASET_URL,
                    "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "macro-F1 over provision labels",
        "higher_is_better": True,
        "baseline": "Word TF-IDF (1-2 grams) + unweighted multinomial logistic regression (C=1).",
        "improved": ("Word + char_wb (3-5) TF-IDF, class_weight='balanced' logistic regression, "
                     "C chosen from {3, 30} by 3-fold CV on the training set."),
        "metrics": {
            "baseline_score": b["macro_f1"], "improved_score": i["macro_f1"],
            "improvement": i["macro_f1"] - b["macro_f1"],
            "baseline_f1": b["f1_weighted"], "improved_f1": i["f1_weighted"],
            "baseline_accuracy": b["accuracy"], "improved_accuracy": i["accuracy"],
            "baseline_nn_agreement": b["nn_agreement"], "improved_nn_agreement": i["nn_agreement"],
            "improved_best_C": float(search.best_params_["C"]),
            "runtime_seconds": float(time.time() - t0),
        },
        "notes": ("Subsampled to the first rows of the official partitions for the CI time budget; "
                  "labels with fewer than 4 rows dropped. Data is fetched from datasets-server "
                  "with a fallback to the Hub's parquet conversion if the original id returns 404. "
                  "Both models are deterministic (lbfgs, seeded CV), so a single run with seed 42 "
                  "is reported. Baseline C was not tuned, which is the intended 'standard' "
                  "comparison. Nearest-neighbour agreement is a proxy for traceability, not a "
                  "verified support check. Results are from one split and one subsample; "
                  "differences may be small."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
