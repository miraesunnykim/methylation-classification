import dill
import numpy as np
import pandas as pd
import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, jaccard_score, precision_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ----------------------------
# 1. Load data
# ----------------------------
diffmeths = []
for fold in range(3):
    diffmeth = dill.load(open(
        f"/grain/mk98/methyl/old-methylation-classification/data/GEO/diffmeth/diffmeth_fold{fold}",
        "rb"
    ))
    diffmeths.append(diffmeth)

training_folds = dill.load(open(
    "/grain/mk98/methyl/old-methylation-classification/data/GEO/preprocessed/training_folds.dill",
    "rb"
))

# ----------------------------
# 2. Evaluation function
# ----------------------------
def evaluate_fold_tree(
    fold: int,
    rest_Mv: pd.DataFrame,
    rest_meta: pd.DataFrame,
    holdout_Mv: pd.DataFrame,
    holdout_meta: pd.DataFrame,
    diffmeth: dict,
    top_n: int = 5
):
    """
    Train/test decision tree on top-N differentially methylated probes pooled across tissues.
    """

    # --- 1. Collect top-N probes across all tissues ---
    top_probes = []
    for tissue, df in diffmeth.items():
        df_sorted = df.sort_values("FDR_QValue")
        top_probes.extend(df_sorted.index[:top_n].tolist())
    top_probes = list(set(top_probes))  # remove duplicates

    if len(top_probes) == 0:
        return None

    # --- 2. Subset methylation matrices ---
    rest_Mv_selected = rest_Mv[top_probes].copy()
    holdout_Mv_selected = holdout_Mv[top_probes].copy()

    # --- 3. Prepare multilabel y ---
    rest_multi = utils.propagate_parent(
        utils.training_ontology, rest_meta, tissue_col="training.ID", outdict=False
    )
    rest_mlb = utils.mlb.transform(rest_multi["training.ID"].values)

    holdout_multi = utils.propagate_parent(
        utils.training_ontology, holdout_meta, tissue_col="training.ID", outdict=False
    )
    holdout_mlb = utils.mlb.transform(holdout_multi["training.ID"].values)

    # --- 4. Decision Tree with manual CV on max_depth ---
    depths = [1, 2, 3, 4, 5, 6, None]
    all_metrics = []
    for depth in depths:
        base_clf = DecisionTreeClassifier(class_weight="balanced", random_state=42, max_depth=depth)
        clf = MultiOutputClassifier(base_clf)
        clf.fit(rest_Mv_selected.values, rest_mlb)

        # Predict on holdout
        pred_bin = clf.predict(holdout_Mv_selected.values)
        pred_set = utils.mlb.inverse_transform(pred_bin)
        pred = utils.mlb.transform(pred_set)

        metrics = {
            "fold": fold,
            "topn": top_n,
            "max_depth": depth,
            "acc-samp": accuracy_score(holdout_mlb, pred),
            "jacc-samp": jaccard_score(holdout_mlb, pred, average="samples", zero_division=0),
            "prec-samp": precision_score(holdout_mlb, pred, average="samples", zero_division=0),
            "prec-tiss": utils.custom_macro_precision(holdout_mlb, pred),
            "f1-tiss": utils.custom_tissue_f1_score(holdout_mlb, pred),
            "n_probes": len(top_probes),
        }
        all_metrics.append(metrics)
    return all_metrics

# ----------------------------
# 3. Run evaluation
# ----------------------------
all_results = []

for fold, (train_mv, train_meta, test_mv, test_meta) in training_folds.items():
    print(f"Fold {fold}")
    diffmeth = diffmeths[fold]
    for topn in [1, 3, 5]:
        print(f"top n: {topn}")
        metrics_list = evaluate_fold_tree(
            fold, train_mv, train_meta, test_mv, test_meta, diffmeth, top_n=topn
        )
        if metrics_list is not None:
            for metrics in metrics_list:
                print(metrics)
                all_results.append(metrics)

# ----------------------------
# 4. Save results as DataFrame
# ----------------------------
results_df = pd.DataFrame(all_results)
# display(results_df.sort_values(by='topn'))

# optional: save to disk
results_df.to_csv("./../_revision_results/decision_tree_diffmeth_results.csv", index=False)
