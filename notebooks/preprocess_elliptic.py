"""
Preprocess the Elliptic Bitcoin Dataset for PyTorch Geometric.

Inputs (in --raw_dir):
    elliptic_txs_features.csv  (no header: col 0 = txId, col 1 = time step, 165 remaining cols = features)
    elliptic_txs_classes.csv   (txId, class: '1' = illicit, '2' = licit, 'unknown')
    elliptic_txs_edgelist.csv  (txId1, txId2)

Outputs (in --out_dir):
    txid_to_idx.csv                mapping original txId -> node index (0..N-1)
    elliptic_txs_edgelist_idx.csv  edge list using the new node indices
    elliptic_features_scaled.csv   standardized features (+ node_idx, time_step, label)  [optional]
    elliptic_processed.pt          dict of tensors: x, y, edge_index, time_step, train_mask, test_mask

Usage:
    python preprocess_elliptic.py --raw_dir raw --out_dir data
    (in Colab:  %run preprocess_elliptic.py --raw_dir /content/raw --out_dir /content/data)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="raw", help="folder containing the 3 raw Kaggle CSV files")
    p.add_argument("--out_dir", default="data", help="folder for processed outputs")
    p.add_argument("--train_last_step", type=int, default=34,
                   help="last time step used for training (train: 1-34, test: 35-49)")
    p.add_argument("--no_time_step", action="store_true",
                   help="exclude time step from features (165 features instead of 166)")
    p.add_argument("--no_csv", action="store_true",
                   help="do not save the (large) scaled features CSV")
    # parse_known_args so it also works inside notebooks
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load data
    feat_df = pd.read_csv(os.path.join(args.raw_dir, "elliptic_txs_features.csv"), header=None)
    cls_df = pd.read_csv(os.path.join(args.raw_dir, "elliptic_txs_classes.csv"))
    edge_df = pd.read_csv(os.path.join(args.raw_dir, "elliptic_txs_edgelist.csv"))

    feat_df = feat_df.rename(columns={0: "txId", 1: "time_step"})
    feat_df["txId"] = feat_df["txId"].astype(np.int64)
    cls_df["txId"] = cls_df["txId"].astype(np.int64)
    edge_df["txId1"] = edge_df["txId1"].astype(np.int64)
    edge_df["txId2"] = edge_df["txId2"].astype(np.int64)
    print(f"Features: {feat_df.shape} | Classes: {cls_df.shape} | Edges: {edge_df.shape}")

    # 2. Build txId -> node index dictionary (0..N-1, following the features file order)
    assert feat_df["txId"].is_unique, "Duplicate txId found in the features file!"
    tx_ids = feat_df["txId"].values
    txid_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
    N = len(txid_to_idx)
    print(f"Number of nodes N = {N}")

    pd.DataFrame({"txId": tx_ids, "node_idx": np.arange(N)}).to_csv(
        os.path.join(args.out_dir, "txid_to_idx.csv"), index=False
    )

    # 3. Remap the edge list to node indices
    edge_idx = pd.DataFrame({
        "txId1": edge_df["txId1"].map(txid_to_idx),
        "txId2": edge_df["txId2"].map(txid_to_idx),
    })
    n_bad = int(edge_idx.isna().any(axis=1).sum())
    if n_bad:
        print(f"[Warning] Dropping {n_bad} edges whose txId is missing from the features file")
        edge_idx = edge_idx.dropna()
    edge_idx = edge_idx.astype(np.int64)
    assert edge_idx.values.min() >= 0 and edge_idx.values.max() < N
    edge_idx.to_csv(os.path.join(args.out_dir, "elliptic_txs_edgelist_idx.csv"), index=False)
    print(f"New edge list: {edge_idx.shape}")

    # 4. Labels: illicit (class '1') -> 1, licit (class '2') -> 0, unknown -> -1
    label_map = {"1": 1, "2": 0, "unknown": -1}
    cls_df["label"] = cls_df["class"].astype(str).map(label_map)
    assert cls_df["label"].notna().all(), "Unexpected class value found!"
    y = (
        cls_df.set_index("txId")["label"]
        .reindex(tx_ids)  # align with node index order
        .fillna(-1)
        .astype(np.int64)
        .values
    )
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))),
          "(-1 = unknown, 0 = licit, 1 = illicit)")

    # 5. Handle NaN / inf in features
    time_step = feat_df["time_step"].values.astype(np.int64)
    feature_cols = [c for c in feat_df.columns if c != "txId"]
    if args.no_time_step:
        feature_cols.remove("time_step")
    X = feat_df[feature_cols].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    n_nan = int(X.isna().sum().sum())
    print(f"Number of NaN/inf values: {n_nan}")

    train_mask = time_step <= args.train_last_step
    test_mask = ~train_mask
    if n_nan:
        # fill with the per-column median of the TRAIN set (avoids data leakage)
        X = X.fillna(X[train_mask].median())
    X = X.values
    print(f"Number of features used: {X.shape[1]}")

    # 6. Standardize - fit ONLY on the training time steps, then transform everything
    scaler = StandardScaler().fit(X[train_mask])
    X_scaled = scaler.transform(X).astype(np.float32)
    assert np.isfinite(X_scaled).all()
    print(f"Train nodes: {int(train_mask.sum())} | Test nodes: {int(test_mask.sum())}")

    # 7. Save outputs
    if not args.no_csv:
        out = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
        out.insert(0, "node_idx", np.arange(N))
        out.insert(1, "time_step", time_step)
        out["label"] = y
        out.to_csv(os.path.join(args.out_dir, "elliptic_features_scaled.csv"), index=False)

    try:
        import torch

        torch.save(
            {
                "x": torch.from_numpy(X_scaled),                           # [N, F] float32
                "y": torch.from_numpy(y),                                  # [N] (-1 unknown, 0 licit, 1 illicit)
                "edge_index": torch.from_numpy(edge_idx.values.T.copy()),  # [2, E] long
                "time_step": torch.from_numpy(time_step),                  # [N]
                "train_mask": torch.from_numpy(train_mask),                # [N] bool
                "test_mask": torch.from_numpy(test_mask),                  # [N] bool
            },
            os.path.join(args.out_dir, "elliptic_processed.pt"),
        )
        print("Saved elliptic_processed.pt")
    except ImportError:
        print("torch is not installed -> skipping the .pt file (pip install torch)")

    print("Done! Outputs are in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
