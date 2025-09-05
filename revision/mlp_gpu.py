# Standard library
import sys
import os
import time
import random
import warnings
import multiprocessing as mp
import tempfile

from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# Third-party
import dill
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, jaccard_score

# Local
sys.path.append('./../src/')
sys.modules['sklearn.externals.joblib'] = joblib
import utils

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Paths
PREPROCESSED_PATH = f'/scratch/mk98'
REVISION_RESULTS_PATH = f'/grain/mk98/methyl/methylation-classification/revision'
os.makedirs(REVISION_RESULTS_PATH, exist_ok=True)

# Config
RANDOM_SEED = 9
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
TIME_LIMIT = 12 * 60 * 60  # 12 hours

# Training hyperparams (match sklearn-ish)
HIDDEN_SIZES = (200, 100)
MAX_EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.001  # default for Adam
ALPHA = 1e-4  # weight decay -> L2 regularization

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
class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim)  # logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------
def _train_worker(save_path: str, X_np: np.ndarray, y_np: np.ndarray, epochs: int, batch_size: int, lr: float, alpha: float, seed: int):
    """
    Runs inside a separate process. Trains the model on GPU (if available),
    saves the state_dict to save_path and writes training time to stdout via pipe.
    """
    import time, torch, numpy as _np
    torch.manual_seed(seed)
    _np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    X = torch.from_numpy(X_np.astype('float32'))
    y = torch.from_numpy(y_np.astype('float32'))  # multi-label, shape (N, L), values 0/1

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = TorchMLP(input_dim, output_dim).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    # Simple training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        # optional: no verbose per epoch to keep logs clean
    fit_time = time.time() - start_time

    # Save state_dict to file accessible by parent process
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_sizes': HIDDEN_SIZES,
        'fit_time': fit_time
    }, save_path)

# ------------------------------------------------
def fit_with_timeout_torch(X: np.ndarray, y: np.ndarray, timeout: int = TIME_LIMIT) -> Tuple[object, float]:
    """
    Trains a PyTorch MLP in a separate process with a timeout.

    Returns (model_instance, fit_time) or (None, None) if timed out.
    The returned 'model_instance' is a CPU-based TorchMLP with loaded state_dict.
    """
    parent_conn, child_conn = mp.Pipe()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    tmp_path = tmp.name
    tmp.close()

    # Worker target simply trains and writes to tmp_path
    p = mp.Process(target=_train_worker, args=(tmp_path, X, y, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE, ALPHA, RANDOM_SEED))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None, None

    # At this point, tmp_path should exist with saved tok
    if not os.path.exists(tmp_path):
        # something went wrong
        return None, None

    # Load checkpoint and reconstruct model on CPU
    ckpt = torch.load(tmp_path, map_location='cpu')
    input_dim = ckpt['input_dim']
    output_dim = ckpt['output_dim']
    hidden_sizes = ckpt.get('hidden_sizes', HIDDEN_SIZES)
    fit_time = ckpt.get('fit_time', None)

    model = TorchMLP(input_dim, output_dim, hidden_sizes=hidden_sizes)
    model.load_state_dict(ckpt['state_dict'])
    model.to('cpu')
    model.eval()

    # remove temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return model, fit_time

# ------------------------------------------------
def evaluate_fold(rest_Mv, rest_meta, holdout_Mv, holdout_meta):
    """Train and evaluate multilabel MLP with timeout (PyTorch GPU)"""
    rest_multi = utils.propagate_parent(utils.training_ontology, rest_meta, tissue_col='training.ID', outdict=False)
    rest_mlb = utils.mlb.transform(rest_multi['training.ID'].values)
    holdout_multi = utils.propagate_parent(utils.training_ontology, holdout_meta, tissue_col='training.ID', outdict=False)
    holdout_mlb = utils.mlb.transform(holdout_multi['training.ID'].values)

    # Prepare numpy arrays
    X_train = rest_Mv.values.astype('float32')
    y_train = rest_mlb.astype('float32')
    X_holdout = holdout_Mv.values.astype('float32')
    y_holdout = holdout_mlb.astype('float32')

    # Train on GPU (worker process). Returns CPU model or None on timeout.
    fitted_model, fit_time = fit_with_timeout_torch(X_train, y_train, timeout=TIME_LIMIT)

    if fitted_model is None:  # training timed out
        timestamped_print("Training timed out (>12h) or failed. Skipping this fold.")
        metrics = {
            'accuracy': np.nan,
            'jaccard': np.nan,
            'custom_f1': np.nan,
            'fit_time': float('inf'),
            'timed_out': True
        }
        return metrics, None

    # Prediction: compute sigmoid(logits) to get probabilities
    with torch.no_grad():
        xb = torch.from_numpy(X_holdout)
        logits = fitted_model(xb)
        probs = torch.sigmoid(logits).numpy()  # shape (n_samples, n_labels)

    pred_bin = (probs >= 0.5).astype(int)
    pred = pred_bin

    tissue_precision = precision_score(y_holdout, pred, average=None, zero_division=0)
    tissue_names = utils.mlb.classes_
    sample_counts = y_holdout.sum(axis=0)

    tissue_metrics = {
        tissue: None if count == 0 else float(score)
        for tissue, score, count in zip(tissue_names, tissue_precision, sample_counts)
    }

    metrics = {
        'accuracy': float(accuracy_score(y_holdout, pred)),
        'jaccard': float(jaccard_score(y_holdout, pred, average="samples")),
        'custom_f1': float(utils.custom_tissue_f1_score(y_holdout, pred)),
        'fit_time': fit_time,
        'timed_out': False,
        **tissue_metrics
    }

    timestamped_print(
        f"acc {metrics['accuracy']:.4f} | jacc {metrics['jaccard']:.4f} | "
        f"median prec {np.nanmedian(tissue_precision):.4f} | f1 {metrics['custom_f1']:.4f} | "
        f"time {fit_time:.2f}s"
    )

    # We will return the fitted model's state_dict (so saving is robust)
    try:
        # return state_dict (CPU tensors) so dill can save it, and metadata to reconstruct
        fitted_info = {
            'state_dict': fitted_model.state_dict(),
            'input_dim': fitted_model.net[0].in_features,
            'output_dim': fitted_model.net[-1].out_features,
            'hidden_sizes': HIDDEN_SIZES
        }
    except Exception:
        fitted_info = fitted_model  # fallback: return the model object (may still be picklable)

    return metrics, fitted_info

# ------------------------------------------------
def main():
    Mv, meta = load_data()
    fold_Mvs = load_fold_data()

    fold_results = {}
    fold_clfs = {}

    for fold, (rest_Mv, rest_meta, holdout_Mv, holdout_meta) in fold_Mvs.items():
        timestamped_print(f"\nFold {fold}:")
        metrics, clf_info = evaluate_fold(rest_Mv, rest_meta, holdout_Mv, holdout_meta)
        fold_results[fold] = metrics
        fold_clfs[fold] = clf_info

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
