import json
import sys

import numpy as np
import pandas as pd
import torch

from models.gat_resnet import GATResNet
from scripts.explain_gat_resnet import load_test_graph, main


def test_bf25_exports_use_test_graph_transaction_ids(tmp_path, monkeypatch):
    tx_ids = [90_000_001, 91_000_003, 92_000_005, 93_000_007, 94_000_009]
    times = [34, 1, 35, 49, 35]
    values = np.linspace(-1, 1, 165)
    rows = [[tx, time, *(values + i)] for i, (tx, time) in enumerate(zip(tx_ids, times))]
    pd.DataFrame(rows).to_csv(tmp_path / "elliptic_txs_features.csv", index=False, header=False)
    pd.DataFrame({
        "txId": list(reversed(tx_ids)), "class": ["unknown", "2", "1", "2", "1"],
    }).to_csv(tmp_path / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame({
        "txId1": [tx_ids[0], tx_ids[2]], "txId2": [tx_ids[2], tx_ids[3]],
    }).to_csv(tmp_path / "elliptic_txs_edgelist.csv", index=False)

    checkpoint = tmp_path / "state.pt"
    torch.manual_seed(25)
    model = GATResNet().eval()
    torch.save(model.state_dict(), checkpoint)
    x, edges, nodes = load_test_graph(tmp_path)
    assert nodes["tx_id"].tolist() == tx_ids[2:]
    assert nodes["label"].tolist() == [1, 0, -1]
    assert edges.tolist() == [[0, 1], [1, 0]]

    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "explain_gat_resnet",
        "--data-dir", str(tmp_path), "--checkpoint", str(checkpoint),
        "--output", str(output), "--threshold", "0", "--max-nodes", "3",
        "--controls", "0", "--top-k", "3", "--steps", "32",
    ])
    main()
    predictions = pd.read_csv(output / "predictions.csv").set_index("tx_id")
    assert set(predictions.index) == set(tx_ids[2:])
    with torch.inference_mode():
        np.testing.assert_allclose(
            predictions.loc[tx_ids[2:], "prob_illicit"], model(x, edges, True)[:, 1].numpy(),
            atol=1e-6,
        )
        with np.load(output / "embeddings.npz") as artifacts:
            assert artifacts["tx_id"].tolist() == tx_ids[2:]
            np.testing.assert_allclose(artifacts["embedding"], model.embed(x, edges).numpy(), atol=1e-6)

    attention = pd.read_csv(output / "attention.csv")
    assert attention.groupby("layer")["head"].nunique().to_dict() == {1: 4, 2: 4, 3: 1}
    assert attention["is_self_loop"].sum() == 27
    assert set(attention["src_tx_id"]) == set(tx_ids[2:])
    assert set(attention["dst_tx_id"]) == set(tx_ids[2:])
    np.testing.assert_allclose(
        attention.groupby(["layer", "head", "dst_tx_id"])["attention"].sum(), 1.0, atol=1e-6,
    )
    explanations = json.loads((output / "explanations.json").read_text(encoding="utf-8"))
    assert {item["tx_id"] for item in explanations} == set(tx_ids[2:])
    for item in explanations:
        assert len(item["features"]) == 3
        ranks = [feature["absolute_attribution"] for feature in item["features"]]
        assert ranks == sorted(ranks, reverse=True)
        assert abs(item["convergence_delta"]) < 0.1
