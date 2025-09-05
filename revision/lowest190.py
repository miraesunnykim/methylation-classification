import warnings
warnings.filterwarnings('ignore')
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
sys.path.append('./../src/')

import dill
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, f1_score
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import utils
import pickle
import numpy as np
import random
from typing import Dict, Tuple, List
import os

# =========================
# Config
# =========================
DATA_PATH = f'/grain/mk98/methyl/old-methylation-classification/data/GEO'
PREPROCESSED_PATH = f'{DATA_PATH}/preprocessed'
MINIPATCH_PATH  = f"{DATA_PATH}/minipatch"
REIVSION_PATH = f"/grain/mk98/methyl/methylation-classification/revision_results/"

RANDOM_SEED = 9
SELECTION_FREQ_RANGE = "[0.8,0.2]"
LOWEST_PROBES = 190   # instead of threshold
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# =========================
# Loaders
# =========================
def load_fold_data() -> Dict:
    """Load cross-validation fold data."""
    with open(f'{PREPROCESSED_PATH}/training_folds.dill', 'rb') as f:
        return pickle.load(f)

def load_selectors() -> Dict:
    """Load trained selectors only (not classifiers)."""
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
    lowest_probes_n: int = 190
):

    # --- selection frequency ---
    selection_freq = pd.DataFrame(selector.Pi_hat_last_k_, index=rest_Mv.columns, columns=['freq'])
    
    # Join CGI position info, fill NaN as 'Open Sea'
    probe_info_sub = probe_info.set_index('probeID')['CGIposition'].fillna('Open Sea')
    selection_freq = selection_freq.join(probe_info_sub, how='left')
    
    # Filter: zero freq & Island
    valid_probes = selection_freq[(selection_freq['freq'] == 0) & (selection_freq['CGIposition'] == 'Island')]
    
    # Pick lowest N
    lowest_probes = list(valid_probes.sort_values('freq', ascending=True).head(lowest_probes_n).index)
    
    if len(lowest_probes) == 0:
        print(f"Warning: No probes selected for fold {fold}")
        return None, None, None

    # --- subset matrices ---
    rest_Mv_selected = rest_Mv[lowest_probes]
    holdout_Mv_selected = holdout_Mv[lowest_probes]

    # --- labels ---
    rest_multi = utils.propagate_parent(utils.training_ontology, rest_meta, tissue_col='training.ID', outdict=False)
    rest_mlb = utils.mlb.transform(rest_multi['training.ID'].values)

    holdout_multi = utils.propagate_parent(utils.training_ontology, holdout_meta, tissue_col='training.ID', outdict=False)
    holdout_mlb = utils.mlb.transform(holdout_multi['training.ID'].values)

    # --- train classifier ---
    clf = MultiOutputClassifier(
        SVC(class_weight='balanced', kernel='linear', probability=True, random_state=RANDOM_SEED)
    )
    clf.fit(rest_Mv_selected.values, rest_mlb)

    # --- predict ---
    pred_prob = np.array([y_pred[:, 1] for y_pred in clf.predict_proba(holdout_Mv_selected.values)]).T
    pred_bin = (pred_prob >= 0.5).astype(int)
    pred_set = utils.mlb.inverse_transform(pred_bin)

    return pred_prob, pred_set, holdout_mlb



# =========================
# Main
# =========================
def main():
    fold_selectors = load_selectors()
    fold_Mvs = load_fold_data()
    
    manifest_path = "/grain/mk98/methyl/old-methylation-classification/_annotation/HM450.hg38.manifest.gencode.v36.tsv.gz"
    probe_info = pd.read_csv(manifest_path, sep='\t', usecols=['probeID', 'CGIposition'])

    pred_res = pd.DataFrame(columns=['predict_proba', 'pred', 'true'])

    for fold, (rest_Mv, rest_meta, holdout_Mv, holdout_meta) in fold_Mvs.items():
        print(f"\n=== Fold {fold} ===")
        print(f"Pre-selection shapes: {rest_Mv.shape, holdout_Mv.shape}")

        pred_prob, pred_set, true_set = evaluate_fold(
            fold, rest_Mv, rest_meta, holdout_Mv, holdout_meta, fold_selectors[fold], probe_info
        )

        if pred_prob is not None:
            for i, gse in enumerate(holdout_meta.index):
                pred_res.loc[f'f{fold}.{gse}'] = [pred_prob[i], pred_set[i], true_set[i]]

            # --- metrics ---
            pred = utils.mlb.transform(pred_set)
            metrics = {
                'acc-samp': accuracy_score(true_set, pred),
                'jacc-samp': jaccard_score(true_set, pred, average="samples"),
                'prec-samp': precision_score(true_set, pred, average="samples"),
                'prec-tiss': utils.custom_macro_precision(true_set, pred),
                'f1-tiss': utils.custom_tissue_f1_score(true_set, pred)
            }

            print('\n'.join(f'{k}: {round(v,4)}' for k, v in metrics.items()))

            with open(f"{REIVSION_PATH}/crossvalidation_results_lowest{LOWEST_PROBES}.txt", 'a') as f:
                f.write('\t'.join(str(round(v,4)) for v in metrics.values()) + '\n')

    # Save preds
    with open(f"{REIVSION_PATH}/crossvalidation_pred_lowest{LOWEST_PROBES}.dill", 'wb') as f:
        pickle.dump(pred_res, f)


if __name__ == "__main__":
    main()
