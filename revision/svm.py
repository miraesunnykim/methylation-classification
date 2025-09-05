# Standard library
import sys, os
import time
import random
import warnings
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# Third-party
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, jaccard_score

# Local
sys.path.append('./../src/')
sys.modules['sklearn.externals.joblib'] = joblib
import utils
import dill

DATA_PATH = f'/grain/mk98/methyl/old-methylation-classification/data/GEO'
PREPROCESSED_PATH = f'{DATA_PATH}/preprocessed'
REVISION_RESULTS_PATH = f'/grain/mk98/methyl/methylation-classification/revision_results'
if not os.path.exists(REVISION_RESULTS_PATH):
    os.makedirs(REVISION_RESULTS_PATH, exist_ok=True)

# Configuration
RANDOM_SEED = 9
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def timestamped_print(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    Mv_location = f'{PREPROCESSED_PATH}/training.dill'
    timestamped_print(f"loading Mv, meta from {Mv_location}")
    return dill.load(open(Mv_location, 'rb'))

def load_fold_data() -> Dict:
    with open(f'{PREPROCESSED_PATH}/training_folds.dill', 'rb') as f:
        return dill.load(f)

def evaluate_fold(rest_Mv, rest_meta, holdout_Mv, holdout_meta):
    """Train and evaluate without feature selection"""
    rest_multi = utils.propagate_parent(utils.training_ontology, rest_meta, tissue_col='training.ID', outdict=False)
    rest_mlb = utils.mlb.transform(rest_multi['training.ID'].values)
    holdout_multi = utils.propagate_parent(utils.training_ontology, holdout_meta, tissue_col='training.ID', outdict=False)
    holdout_mlb = utils.mlb.transform(holdout_multi['training.ID'].values)

    clf = MultiOutputClassifier(
        SVC(class_weight='balanced', kernel='linear', random_state=RANDOM_SEED, probability=True)
    )
    start = time.time()
    clf.fit(rest_Mv.values, rest_mlb)
    fit_time = time.time() - start

    pred_prob = np.array([y_pred[:, 1] for y_pred in clf.predict_proba(holdout_Mv.values)]).T
    pred_proba_bin = (pred_prob >= 0.5).astype(int)
    pred_proba_pred = [np.where(row)[0] for row in pred_proba_bin]
    pred_proba_pred = [tuple(utils.mlb.classes_[idx]) for idx in pred_proba_pred]
    pred = utils.mlb.transform(pred_proba_pred)

    tissue_precision = precision_score(holdout_mlb, pred, average=None)
    tissue_names = utils.mlb.classes_
    sample_counts = holdout_mlb.sum(axis=0)

    tissue_metrics = {tissue: None if count == 0 else score 
                     for tissue, score, count in zip(tissue_names, tissue_precision, sample_counts)}

    metrics = {
        'accuracy': accuracy_score(holdout_mlb, pred),
        'jaccard': jaccard_score(holdout_mlb, pred, average="samples"),
        'custom_f1': utils.custom_tissue_f1_score(holdout_mlb, pred),
        'fit_time': fit_time,
        **tissue_metrics
    }

    timestamped_print(
        f"acc {metrics['accuracy']:.4f} | jacc {metrics['jaccard']:.4f} | "
        f"median prec {np.nanmedian(tissue_precision):.4f} | f1 {metrics['custom_f1']:.4f} | "
        f"time {fit_time:.2f}s"
    )
    return metrics, clf


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
    with open(f"{REVISION_RESULTS_PATH}/svm_results.dill", "wb") as f:
        dill.dump(fold_results, f)
    with open(f"{REVISION_RESULTS_PATH}/svm_clfs.dill", "wb") as f:
        dill.dump(fold_clfs, f)

    # Export also to CSV for easy inspection
    pd.DataFrame(fold_results).T.to_csv(f"{REVISION_RESULTS_PATH}/svm_results.csv")

    timestamped_print("Done.")

if __name__ == "__main__":
    main()
