"""Load the Elliptic Bitcoin dataset (3 CSV files) into pandas DataFrames."""
import os
import pandas as pd

CANDIDATES = [
    os.environ.get("ELLIPTIC_DIR", ""),
    "/content/drive/MyDrive/Explainable-AML-GAT-ResNet/data",
]
DATA_DIR = next((p for p in CANDIDATES if p and os.path.isdir(p)), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Data directory not found. Please: (1) mount Google Drive in Colab, "
        "(2) add a shortcut of the shared folder 'Explainable-AML-GAT-ResNet' "
        "to 'My Drive', or set the ELLIPTIC_DIR environment variable."
    )


def load_elliptic():
    """Return (features, edges, classes) as pandas DataFrames."""
    # The features file has no header:
    # column 0 = txId, column 1 = time step, remaining 165 columns = features
    features = pd.read_csv(
        os.path.join(DATA_DIR, "elliptic_txs_features.csv"),
        header=None,
        encoding="utf-8",
    )
    edges = pd.read_csv(
        os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"), encoding="utf-8"
    )
    classes = pd.read_csv(
        os.path.join(DATA_DIR, "elliptic_txs_classes.csv"), encoding="utf-8"
    )
    return features, edges, classes


if __name__ == "__main__":
    features, edges, classes = load_elliptic()
    for name, df in [("features", features), ("edges", edges), ("classes", classes)]:
        print(f"\n===== {name} =====")
        print("shape:", df.shape)
        print(df.head())
        df.info()

    n_nodes = classes.shape[0]
    assert n_nodes == 203769, f"expected 203,769 nodes, got {n_nodes}"
    assert features.shape[0] == n_nodes, "features/classes row count mismatch"
    assert features.shape[1] - 1 == 166, "expected 166 columns besides txId"
    assert features[0].is_unique, "txId in features must be unique"
    assert set(classes["txId"]) == set(features[0]), "txId sets differ"
    assert set(classes["class"].astype(str).unique()) <= {"1", "2", "unknown"}, "unexpected class label"
    print(f"\nAll checks passed: {n_nodes:,} nodes, {features.shape[1] - 1} columns besides txId.")
