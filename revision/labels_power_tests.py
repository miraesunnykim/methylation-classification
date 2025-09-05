# --- CV loop: train minipatch selector and SVM for each fold ---
from mplearn.feature_selection._adaptive_stable_minipatch_selection import AdaSTAMPS
from mplearn.feature_selection.base_selector import DecisionTreeSelector
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, jaccard_score
from utils import custom_macro_precision, custom_tissue_f1_score
import dill
import pandas as pd
import numpy as np
import random
import networkx as nx
import logging
from datetime import datetime
from utils import propagate_parent, id_to_name, mlb, name_to_id
import warnings
warnings.filterwarnings("ignore")

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_file = 'labels_power_tests.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
else:
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def make_subset(meta_df, ontology, system_ontology, method='system', percentage=0.5, seed=9):
    """
    Create a subset of samples from meta based on ontology structure.

    method: 
        'system' - select X% of system nodes and all their descendants (including nodes with multiple system parents).
                   Only directly chosen systems are considered chosen, even if a child is shared between systems.
        'stratified' - select X% of labels per system node (stratified)
    percentage: float in (0,1]
    """
    if seed is not None:
        random.seed(seed)
    system_nodes = list(system_ontology.successors('root'))
    subset_labels = set()

    if method == 'system':
        n = max(1, int(len(system_nodes) * percentage))
        # Randomly select n system nodes
        chosen_systems = set(random.sample(system_nodes, k=n))
        # Find all descendants of chosen systems
        all_descendants = set()
        for sys in chosen_systems:
            all_descendants |= set(nx.descendants(ontology, sys))
        # Also include the chosen system nodes themselves
        subset_labels = all_descendants | chosen_systems

    elif method == 'stratified':
        for sys in system_nodes:
            sys_labels = list(nx.descendants(ontology, sys)) + [sys]
            n = max(1, int(len(sys_labels) * percentage))
            subset_labels |= set(random.sample(sys_labels, k=n))
    else:
        raise ValueError("method must be 'system' or 'stratified'")

    subset_samples = meta_df[meta_df['training.ID'].isin(subset_labels)]
    return subset_samples

ANNOTATION_PATH = "/grain/mk98/methyl/old-methylation-classification/annotation"

Mv, meta = dill.load(open(f'/grain/mk98/methyl/old-methylation-classification/data/GEO/preprocessed/training.dill', 'rb'))
folds = dill.load(open(f'/grain/mk98/methyl/old-methylation-classification/data/GEO/preprocessed/training_folds.dill', 'rb'))
fold_selectors = dill.load(open('/grain/mk98/methyl/old-methylation-classification/data/GEO/minipatch/crossvalidation_selectors_[0.8,0.2]', 'rb'))
fold_clfs = dill.load(open('/grain/mk98/methyl/old-methylation-classification/data/GEO/minipatch/crossvalidation_clfs_[0.8,0.2]', 'rb'))

threshold = 0.65

for fold, (train_mv, train_meta, test_mv, test_meta) in folds.items():
    logger.info(f"Fold {fold}: train_mv={len(train_mv)}, train_meta={len(train_meta)}, test_mv={len(test_mv)}, test_meta={len(test_meta)}")
    selector = fold_selectors[fold]
    selection_freq = pd.DataFrame(selector.Pi_hat_last_k_, index=train_mv.columns)
    minipatch_probes = list(selection_freq[selection_freq[0] >= threshold].index)

    logger.info(f"Fold {fold}: minipatch_probes={len(minipatch_probes)}")

with open(f'{ANNOTATION_PATH}/ontologies.dill', 'rb') as f:
    [full_ontology, training_ontology] = dill.load(f)

def initialize_selector(rest_Mv: pd.DataFrame, seed=9) -> AdaSTAMPS:
    """Initialize the AdaSTAMPS selector."""
    m_ratio = np.sqrt(rest_Mv.shape[1])/rest_Mv.shape[1]
    n_ratio = np.sqrt(rest_Mv.shape[0])/rest_Mv.shape[0]

    clf = DecisionTreeSelector(random_state=seed)
    return AdaSTAMPS(base_selector=clf,
                     minipatch_m_ratio=m_ratio,
                     minipatch_n_ratio=n_ratio,
                     random_state=seed,
                     verbose=1,
                     )

results = []
for RANDOM_SEED in [9, 10, 11, 12, 15]:
    for subset_type in ['stratified','system']:
        for subset_pct in [0.7, 0.5, 0.2]:
            logger.info(f"Subset type: {subset_type}, Subset percentage: {subset_pct}")
            for fold, (train_mv, train_meta, test_mv, test_meta) in folds.items():
                logger.info(f"Fold {fold}:")
                # Subset selection on training meta
                train_subset_meta = make_subset(train_meta, full_ontology, training_ontology, method=subset_type, percentage=subset_pct, seed=9)
                train_subset_mv = train_mv.loc[train_subset_meta.index]
                test_subset_meta = test_meta[test_meta['training.ID'].isin(train_subset_meta['training.ID'])]
                test_subset_mv = test_mv.loc[test_subset_meta.index]

                logger.info(f"n tissues subset: {train_subset_meta['training.ID'].unique().size}")
                logger.info(f"n samples subset: {train_subset_mv.shape[0]}")

                # Train minipatch selector
                selector = initialize_selector(train_subset_mv, seed=RANDOM_SEED)
                selector.fit(train_subset_mv.values, train_subset_meta['training.ID'])
                selection_freq = pd.DataFrame(selector.Pi_hat_last_k_, index=train_subset_mv.columns)
                minipatch_probes = list(selection_freq[selection_freq[0] >= 0.65].index)
                train_subset_mv_minipatch = train_subset_mv[minipatch_probes]

                logger.info(f"n probes subset: {train_subset_mv_minipatch.shape[1]}")

                subset_mlb = MultiLabelBinarizer()

                # Multioutput SVM
                train_subset_meta_multi = propagate_parent(training_ontology, train_subset_meta, tissue_col='training.ID',outdict=False)
                train_subset_meta_mlb = subset_mlb.fit_transform(train_subset_meta_multi['training.ID'].values)
                clf = MultiOutputClassifier(SVC(class_weight='balanced', kernel='linear', random_state=RANDOM_SEED, probability=True))
                clf.fit(train_subset_mv_minipatch.values, train_subset_meta_mlb)

                test_subset_mv_minipatch = test_subset_mv[minipatch_probes]
                test_subset_meta_multi = propagate_parent(training_ontology, test_subset_meta, tissue_col='training.ID',outdict=False)
                test_subset_meta_mlb = subset_mlb.transform(test_subset_meta_multi['training.ID'].values)

                pred_prob = np.array([y_pred[:, 1] for y_pred in clf.predict_proba(test_subset_mv_minipatch.values)]).T
                pred_proba_bin = (pred_prob >= 0.5).astype(int)
                pred_proba_pred = [np.where(row)[0] for row in pred_proba_bin]
                pred_proba_pred = [tuple(subset_mlb.classes_[idx]) for idx in pred_proba_pred]
                pred = subset_mlb.transform(pred_proba_pred)

                metrics = {
                    'acc-samp': accuracy_score(test_subset_meta_mlb, pred),
                    'jacc-samp': jaccard_score(test_subset_meta_mlb, pred, average="samples", zero_division=0),
                    'prec-samp': precision_score(test_subset_meta_mlb, pred, average="samples", zero_division=0),
                    'prec-tiss': custom_macro_precision(test_subset_meta_mlb, pred),
                    'f1-tiss': custom_tissue_f1_score(test_subset_meta_mlb, pred)
                }
                logger.info(f"Fold {fold}: subset shape {train_subset_mv_minipatch.shape}")
                logger.info(f"Metrics: {metrics}")

                results.append(
                    {
                    'random_seed': RANDOM_SEED,
                    'subset_type': subset_type,
                    'subset_pct' : subset_pct,
                    'subset_tissue' : train_subset_meta['training.ID'].unique().tolist(),
                    'subset_tissue_name' : [id_to_name.get(tid, tid) for tid in train_subset_meta['training.ID'].unique().tolist()],
                    'subset_mv': train_subset_mv_minipatch,
                    'metrics': metrics
                    }
                )
logger.info("All folds complete.")

# Save results as DataFrame and dill
results_df = pd.DataFrame(results)
dill.dump(results_df, open('_results_df.dill', 'wb'))