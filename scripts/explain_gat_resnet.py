"""Run BF-23 inference and export BF-25 explanations keyed by transaction ID."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.utils import k_hop_subgraph, to_undirected

from models.gat_resnet import GATResNet


FEATURE_COUNT = 165
TRAIN_END = 34
TEST_END = 49


def load_test_graph(data_dir):
    features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    tx_ids = pd.Index(features[0])
    times = features[1].to_numpy()
    if len(features.columns) != FEATURE_COUNT + 2 or not tx_ids.is_unique:
        raise ValueError("Expected unique transaction IDs and 165 raw feature columns")
    if not np.isin(times, np.arange(1, TEST_END + 1)).all():
        raise ValueError("Elliptic time steps must be in 1–49")
    train = times <= TRAIN_END
    test = ~train
    if not train.any() or not test.any():
        raise ValueError("Both temporal splits must contain nodes")

    raw = features.iloc[:, 2:].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Raw features must be finite")
    logged = np.sign(raw) * np.log1p(np.abs(raw))
    x = StandardScaler().fit(logged[train]).transform(logged[test]).astype(np.float32)

    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv", dtype={"class": str})
    labels = classes.set_index("txId")["class"]
    if not labels.index.is_unique:
        raise ValueError("Class data contains duplicate transaction IDs")
    if not labels.isin(["1", "2", "unknown"]).all():
        raise ValueError("Unexpected Elliptic class")
    y = labels.map({"1": 1, "2": 0, "unknown": -1}).reindex(tx_ids[test]).fillna(-1)

    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    src = tx_ids.get_indexer(edges["txId1"])
    dst = tx_ids.get_indexer(edges["txId2"])
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("An edge references a missing transaction ID")
    keep = test[src] & test[dst]
    test_index = np.cumsum(test, dtype=np.int64) - 1
    edge_index = to_undirected(
        torch.from_numpy(np.stack((test_index[src[keep]], test_index[dst[keep]]))),
        num_nodes=len(x),
    )
    nodes = pd.DataFrame({"tx_id": tx_ids[test], "time_step": times[test], "label": y.to_numpy(dtype=int)})
    return torch.from_numpy(x), edge_index, nodes


def evaluate_labeled(nodes, reference_path, reference_seed):
    labeled = nodes[nodes["label"] >= 0]
    labels = labeled["label"]
    if labels.nunique() != 2:
        if reference_path:
            raise ValueError("Both labeled classes are required for the BF-23 comparison")
        return {}, {}
    scores = labeled["prob_illicit"]
    metrics = {"pr_auc": float(average_precision_score(labels, scores))}
    for threshold in (0.50, 0.90):
        predicted = scores >= threshold
        suffix = f"t{int(threshold * 100)}"
        metrics.update({
            f"p_{suffix}": float(precision_score(labels, predicted, zero_division=0)),
            f"r_{suffix}": float(recall_score(labels, predicted, zero_division=0)),
            f"f1_{suffix}": float(f1_score(labels, predicted, zero_division=0)),
            f"micro_{suffix}": float(f1_score(labels, predicted, average="micro")),
            f"mcc_{suffix}": float(matthews_corrcoef(labels, predicted)),
        })
    if reference_path is None:
        return metrics, {}
    reference_table = pd.read_csv(reference_path).set_index("seed")
    if not reference_table.index.is_unique:
        raise ValueError("Reference metrics contain duplicate seeds")
    reference = reference_table.loc[reference_seed]
    deltas = {key: value - float(reference[key]) for key, value in metrics.items()}
    return metrics, deltas


def explain_node(model, x, edge_index, index, expected_logit, top_k, steps):
    # Three message-passing layers. The induced neighborhood retains all inputs
    # that can affect the target logit in evaluation mode.
    subset, edges, target, _ = k_hop_subgraph(
        index, 3, edge_index, relabel_nodes=True, num_nodes=len(x),
    )
    point = x[subset].unsqueeze(0)
    target_index = int(target.item())

    def illicit_logit(features):
        return model(features[0], edges)[target_index, 1].reshape(1)

    with torch.no_grad():
        original = illicit_logit(point)
        torch.testing.assert_close(original[0], expected_logit, rtol=1e-4, atol=1e-4)
    attributions, delta = IntegratedGradients(illicit_logit).attribute(
        point, baselines=0, n_steps=steps, internal_batch_size=1,
        return_convergence_delta=True,
    )
    signed = attributions[0].sum(dim=0)
    magnitude = attributions[0].abs().sum(dim=0)
    ranked = torch.argsort(magnitude, descending=True, stable=True)[:top_k]
    masked = point.clone()
    masked[:, :, ranked] = 0
    with torch.no_grad():
        score_drop = (original - illicit_logit(masked)).item()
    return {
        "convergence_delta": delta.item(),
        "top_k_logit_drop": score_drop,
        "features": [
            {
                "feature_index": col,
                "csv_column": col + 2,
                "signed_attribution": signed[col].item(),
                "absolute_attribution": magnitude[col].item(),
                "target_attribution": attributions[0, target_index, col].item(),
            }
            for col in ranked.tolist()
        ],
    }


def export_attention(path, attention, nodes):
    tx_ids = nodes["tx_id"].to_numpy()
    flagged = nodes["flagged"].to_numpy()
    for layer, (edges, weights) in enumerate(attention, start=1):
        src, dst = edges.cpu().numpy()
        coeffs = weights.cpu().numpy()
        heads = coeffs.shape[1]
        frame = pd.DataFrame({
            "src_tx_id": np.repeat(tx_ids[src], heads),
            "dst_tx_id": np.repeat(tx_ids[dst], heads),
            "layer": layer,
            "head": np.tile(np.arange(heads), len(src)),
            "attention": coeffs.reshape(-1),
            "is_self_loop": np.repeat(src == dst, heads),
            "touches_flagged_node": np.repeat(flagged[src] | flagged[dst], heads),
        })
        frame.to_csv(path, mode="w" if layer == 1 else "a", header=layer == 1, index=False)


def run(args, output):
    print("Loading raw Elliptic data and rebuilding the BF-23 test graph", flush=True)
    x, edge_index, nodes = load_test_graph(args.data_dir)
    x, edge_index = x.to(args.device), edge_index.to(args.device)
    model = GATResNet().to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    with torch.inference_mode():
        expected = model(x, edge_index)
        logits, embeddings, attention = model.forward_with_attention(x, edge_index)
        torch.testing.assert_close(logits, expected)
        scores = logits.softmax(dim=1)[:, 1].cpu().numpy()
    nodes["prob_illicit"] = scores
    nodes["flagged"] = scores >= args.threshold
    order = nodes.sort_values(["prob_illicit", "tx_id"], ascending=[False, True])
    flagged = order[order["flagged"]]
    metrics, deltas = evaluate_labeled(nodes, args.reference, args.reference_seed)
    print(f"Scored {len(nodes):,} test nodes. Flagged {len(flagged):,}.", flush=True)

    flagged[["tx_id", "prob_illicit"]].to_csv(output / "predictions.csv", index=False)
    nodes.to_csv(output / "nodes.csv", index=False)
    np.savez_compressed(
        output / "embeddings.npz", tx_id=nodes["tx_id"].to_numpy(), embedding=embeddings.cpu().numpy(),
    )
    export_attention(output / "attention.csv", attention, nodes)

    # Include correct positive and negative examples as controls, where available.
    cases = {index: ["high_risk"] for index in flagged.head(args.max_nodes).index}
    for label, role, matches in (
        (1, "labeled_illicit_flagged", order["flagged"]),
        (0, "labeled_licit_unflagged", ~order["flagged"]),
    ):
        for index in order[(order["label"] == label) & matches].head(args.controls).index:
            cases.setdefault(index, []).append(role)
    explanations = []
    for index, roles in cases.items():
        row = nodes.loc[index]
        explanations.append({
            "tx_id": int(row["tx_id"]),
            "prob_illicit": float(row["prob_illicit"]),
            "roles": roles,
            **explain_node(model, x, edge_index, int(index), logits[index, 1], args.top_k, args.steps),
        })
    (output / "explanations.json").write_text(
        json.dumps(explanations, indent=2, allow_nan=False), encoding="utf-8",
    )
    with args.checkpoint.open("rb") as checkpoint:
        checkpoint_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    metadata = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "train_time_steps": [1, TRAIN_END],
        "test_time_steps": [TRAIN_END + 1, TEST_END],
        "device": args.device,
        "flag_threshold": args.threshold,
        "test_nodes": len(nodes),
        "test_labeled_nodes": int((nodes["label"] >= 0).sum()),
        "flagged_nodes": len(flagged),
        "explained_nodes": len(explanations),
        "method": "integrated_gradients",
        "target": "illicit_logit",
        "baseline": "zero in standardized feature space",
        "top_k": args.top_k,
        "integration_steps": args.steps,
        "metrics_labeled": metrics,
        "reference_seed": args.reference_seed if args.reference else None,
        "metric_delta_vs_reference": deltas,
        "max_abs_convergence_delta": max(
            (abs(item["convergence_delta"]) for item in explanations), default=None,
        ),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_gat_resnet.pt"))
    parser.add_argument("--output", type=Path, required=True, help="New directory for the completed export")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-nodes", type=int, default=5, help="Maximum high-risk nodes to explain")
    parser.add_argument("--controls", type=int, default=1, help="Maximum examples per labeled control group")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--reference", type=Path, help="Optional BF-23 multiseed metrics CSV")
    parser.add_argument("--reference-seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 1 <= args.top_k <= FEATURE_COUNT:
        parser.error("Expected threshold in [0, 1] and top-k in [1, 165]")
    if args.steps < 2 or min(args.max_nodes, args.controls) < 0:
        parser.error("Expected steps >= 2 and nonnegative case limits")
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".bf25-") as temporary:
        export = Path(temporary) / "export"
        export.mkdir()
        run(args, export)
        export.rename(args.output.resolve())
    print(f"Exported BF-25 artifacts to {args.output}")


if __name__ == "__main__":
    main()
