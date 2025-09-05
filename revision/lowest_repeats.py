import warnings
warnings.filterwarnings('ignore')
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
sys.path.append('./../src/')

import dill
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, f1_score, precision_score
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import utils
import pickle
import numpy as np
import random
from typing import Dict
import os

# =========================
# Config
# =========================
DATA_PATH = f'/grain/mk98/methyl/old-methylation-classification/data/GEO'
PREPROCESSED_PATH = f'{DATA_PATH}/preprocessed'
MINIPATCH_PATH  = f"{DATA_PATH}/minipatch"
REIVSION_PATH = f"/grain/mk98/methyl/methylation-classification/_revision_results/"

RANDOM_SEED = 9
SELECTION_FREQ_RANGE = "[0.8,0.2]"
LOWEST_PROBES = 190
REPEATS = 5

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =========================
# Loaders
# =========================
def load_fold_data() -> Dict:
    with open(f'{PREPROCESSED_PATH}/training_folds.dill', 'rb') as f:
        return pickle.load(f)

def load_selectors() -> Dict:
    minipatch_location = f"{MINIPATCH_PATH}/crossvalidation_selectors_{SELECTION_FREQ_RANGE}"
    fold_selectors = dill.load(open(minipatch_location, 'rb'))
    return fold_selectors

# =========================
# Evaluation per fold
# =========================
def evaluate_fold(
    fold: str, 
    rest_Mv: pd.DataFrame, 
    rest_meta: pd.DataFrame,
    holdout_Mv: pd.DataFrame, 
    holdout_meta: pd.DataFrame,
    selector,
    probe_info: pd.DataFrame,
    lowest_probes_n: int = 190,
    probe_mode: str = "all_zero"  # "all_zero" or "island_zero"
):

    selection_freq = pd.DataFrame(selector.Pi_hat_last_k_, index=rest_Mv.columns, columns=['freq'])
    probe_info_sub = probe_info.set_index('probeID')['CGIposition'].fillna('Open Sea')
    selection_freq = selection_freq.join(probe_info_sub, how='left')

    if probe_mode == "all_zero":
        valid_probes = selection_freq[selection_freq['freq'] == 0]
    elif probe_mode == "island_zero":
        valid_probes = selection_freq[(selection_freq['freq'] == 0) & (selection_freq['CGIposition'] == 'Island')]
    else:
        raise ValueError("probe_mode must be 'all_zero' or 'island_zero'")

    if len(valid_probes) < lowest_probes_n:
        lowest_probes = list(valid_probes.index)
    else:
        lowest_probes = list(np.random.choice(valid_probes.index, size=lowest_probes_n, replace=False))

    rest_Mv_selected = rest_Mv[lowest_probes]
    holdout_Mv_selected = holdout_Mv[lowest_probes]

    rest_multi = utils.propagate_parent(utils.training_ontology, rest_meta, tissue_col='training.ID', outdict=False)
    rest_mlb = utils.mlb.transform(rest_multi['training.ID'].values)
    holdout_multi = utils.propagate_parent(utils.training_ontology, holdout_meta, tissue_col='training.ID', outdict=False)
    holdout_mlb = utils.mlb.transform(holdout_multi['training.ID'].values)

    clf = MultiOutputClassifier(
        SVC(class_weight='balanced', kernel='linear', probability=True, random_state=RANDOM_SEED)
    )
    clf.fit(rest_Mv_selected.values, rest_mlb)

    pred_prob = np.array([y_pred[:, 1] for y_pred in clf.predict_proba(holdout_Mv_selected.values)]).T
    pred_bin = (pred_prob >= 0.5).astype(int)
    pred_set = utils.mlb.inverse_transform(pred_bin)

    pred = utils.mlb.transform(pred_set)
    metrics = {
        'acc-samp': accuracy_score(holdout_mlb, pred),
        'jacc-samp': jaccard_score(holdout_mlb, pred, average="samples"),
        'prec-samp': precision_score(holdout_mlb, pred, average="samples"),
        'prec-tiss': utils.custom_macro_precision(holdout_mlb, pred),
        'f1-tiss': utils.custom_tissue_f1_score(holdout_mlb, pred)
    }

    return metrics

# =========================
# Main
# =========================
def main():
    fold_selectors = load_selectors()
    fold_Mvs = load_fold_data()
    
    manifest_path = "/grain/mk98/methyl/old-methylation-classification/_annotation/HM450.hg38.manifest.gencode.v36.tsv.gz"
    probe_info = pd.read_csv(manifest_path, sep='\t', usecols=['probeID', 'CGIposition'])

    all_results = []

    for probe_mode in ["all_zero", "island_zero"]:
        print(f"\n=== Mode: {probe_mode} ===")
        repeat_metrics_all = []

        for repeat in range(REPEATS):
            print(f"\n--- Repeat {repeat+1} ---")
            for fold, (rest_Mv, rest_meta, holdout_Mv, holdout_meta) in fold_Mvs.items():
                fold_probes = pd.DataFrame(fold_selectors[fold].Pi_hat_last_k_, index=rest_Mv.columns, columns=['selection_freq'])
                fold_probes = fold_probes[fold_probes['selection_freq'] > 0.65]
                
                print(f"\n--- Fold: {fold} --- | Selected Probes: {len(fold_probes)} ---")
                metrics = evaluate_fold(
                    fold, rest_Mv, rest_meta, holdout_Mv, holdout_meta,
                    fold_selectors[fold], probe_info, lowest_probes_n=len(fold_probes), probe_mode=probe_mode
                )
                row = {'mode': probe_mode, 'repeat': repeat+1, 'fold': fold}
                row.update(metrics)
                all_results.append(row)
                repeat_metrics_all.append(metrics)

        # Print average metrics per mode (over all repeats and folds)
        avg_metrics = pd.DataFrame(repeat_metrics_all).mean()
        print(f"Average metrics for {probe_mode}:\n{avg_metrics.round(4)}\n")

    # Save all individual results to CSV
    all_results_df = pd.DataFrame(all_results)
    csv_path = f"{REIVSION_PATH}/crossvalidation_all_results_lowest_perfold.csv"
    all_results_df.to_csv(csv_path, index=False)
    print(f"All results saved to {csv_path}")

if __name__ == "__main__":
    main()
