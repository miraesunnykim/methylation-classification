# Standard library
import sys, os
import time
import random
import warnings
import multiprocessing as mp
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# Third-party
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, jaccard_score

# Local
sys.path.append('./../src/')
sys.modules['sklearn.externals.joblib'] = joblib
import utils
import dill

# Paths
DATA_PATH = f'/grain/mk98/methyl/old-methylation-classification/data/GEO'
PREPROCESSED_PATH = f'{DATA_PATH}/preprocessed'
REVISION_RESULTS_PATH = f'/grain/mk98/methyl/methylation-classification/_revision_results'
os.makedirs(REVISION_RESULTS_PATH, exist_ok=True)

# Config
RANDOM_SEED = 9
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
TIME_LIMIT = 12 * 60 * 60  # 12 hours

# ------------------------------------------------
def timestamped_print(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}", flush=True)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    Mv_location = f'{PREPROCESSED_PATH}/training.dill'
    timestamped_print(f"loading Mv, meta from {Mv_location}")
    return dill.load(open(Mv_location, 'rb'))

def load_fold_data() -> Dict:
    with open(f'{PREPROCESSED_PATH}/training_folds.dill', 'rb') as f:
        return dill.load(f)

# ------------------------------------------------
def fit_with_timeout(clf, X, y, timeout: int = TIME_LIMIT):
    """Fit clf with timeout. Returns (clf, fit_time) or (None, None) if timeout."""
    def target(pipe, clf, X, y):
        start = time.time()
        clf.fit(X, y)
        pipe.send((clf, time.time() - start))

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=target, args=(child_conn, clf, X, y))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, None
    else:
        return parent_conn.recv()

# ------------------------------------------------
def evaluate_fold(rest_Mv, rest_meta, holdout_Mv, holdout_meta):
    """Train and evaluate multilabel MLP with timeout"""
    rest_multi = utils.propagate_parent(utils.training_ontology, rest_meta, tissue_col='training.ID', outdict=False)
    rest_mlb = utils.mlb.transform(rest_multi['training.ID'].values)
    holdout_multi = utils.propagate_parent(utils.training_ontology, holdout_meta, tissue_col='training.ID', outdict=False)
    holdout_mlb = utils.mlb.transform(holdout_multi['training.ID'].values)

    base_clf = MLPClassifier(
        hidden_layer_sizes=(200, 100),
        activation="relu",
        solver="adam",
        max_iter=300,
        alpha=1e-4,
        random_state=RANDOM_SEED,
        verbose=False
    )
    clf = MultiOutputClassifier(base_clf)

    fitted, fit_time = fit_with_timeout(clf, rest_Mv.values, rest_mlb, timeout=TIME_LIMIT)

    if fitted is None:  # training timed out
        timestamped_print("Training timed out (>12h). Skipping this fold.")
        metrics = {
            'accuracy': np.nan,
            'jaccard': np.nan,
            'custom_f1': np.nan,
            'fit_time': np.inf,
            'timed_out': True
        }
        return metrics, None

    # Prediction
    pred_prob = np.array([y_pred[:, 1] for y_pred in fitted.predict_proba(holdout_Mv.values)]).T
    pred_bin = (pred_prob >= 0.5).astype(int)
    pred = pred_bin

    tissue_precision = precision_score(holdout_mlb, pred, average=None)
    tissue_names = utils.mlb.classes_
    sample_counts = holdout_mlb.sum(axis=0)

    tissue_metrics = {
        tissue: None if count == 0 else score
        for tissue, score, count in zip(tissue_names, tissue_precision, sample_counts)
    }

    metrics = {
        'accuracy': accuracy_score(holdout_mlb, pred),
        'jaccard': jaccard_score(holdout_mlb, pred, average="samples"),
        'custom_f1': utils.custom_tissue_f1_score(holdout_mlb, pred),
        'fit_time': fit_time,
        'timed_out': False,
        **tissue_metrics
    }

    timestamped_print(
        f"acc {metrics['accuracy']:.4f} | jacc {metrics['jaccard']:.4f} | "
        f"median prec {np.nanmedian(tissue_precision):.4f} | f1 {metrics['custom_f1']:.4f} | "
        f"time {fit_time:.2f}s"
    )
    return metrics, fitted

# ------------------------------------------------
def main():
    Mv, meta = load_data()
    fold_Mvs = load_fold_data()

    fold_results = {}
    fold_clfs = {}

    for fold, (rest_Mv, rest_meta, holdout_Mv, holdout_meta) in fold_Mvs.items():
        timestamped_print(f"\nFold {fold}:")
        metrics, clf = evaluate_fold(rest_Mv, rest_meta, holdout_Mv, holdout_meta)
        fold_results[fold] = metrics
        fold_clfs[fold] = clf

    # Save results
    timestamped_print("\nSaving results...")
    with open(f"{REVISION_RESULTS_PATH}/mlp_results.dill", "wb") as f:
        dill.dump(fold_results, f)
    with open(f"{REVISION_RESULTS_PATH}/mlp_clfs.dill", "wb") as f:
        dill.dump(fold_clfs, f)

    # Export to CSV
    pd.DataFrame(fold_results).T.to_csv(f"{REVISION_RESULTS_PATH}/mlp_results.csv")

    timestamped_print("Done.")

# ------------------------------------------------
if __name__ == "__main__":
    main()
