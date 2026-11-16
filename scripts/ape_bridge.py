"""
Ape Framework integration bridge between AI AML models and the on-chain
AMLGatekeeper Vyper smart contract.

Usage with Ape CLI (recommended):
    ape run ape_bridge

Direct Python execution:
    python scripts/ape_bridge.py
"""

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ape import accounts, networks, project


DECISION_ALLOW = 0
DECISION_WARN = 1
DECISION_BLOCK = 2

DECISION_LABELS = {
    DECISION_ALLOW: "ALLOW",
    DECISION_WARN: "WARN",
    DECISION_BLOCK: "BLOCK",
}


@dataclass(frozen=True)
class AIRiskPrediction:
    """Normalized AML risk prediction ready for on-chain submission."""

    tx_id: int
    risk_score: int
    prob_illicit: Optional[float] = None
    predicted_label: str = "Licit"
    model_version: str = "hybrid-gat-resnet-v1"


@dataclass(frozen=True)
class OnChainSubmissionResult:
    """Confirmed on-chain transaction and verified contract state."""

    tx_id: int
    risk_score: int
    decision_code: int
    decision_label: str
    is_blocked: bool
    tx_hash: str
    block_number: int
    gas_used: int
    chain_id: int
    contract_address: str
    submitter: str
    event_verified: bool
    state_verified: bool
    timestamp_utc: str


def prob_to_risk_score(prob_illicit: float) -> int:
    """
    Convert an AI illicit probability in [0.0, 1.0] to an integer risk score in [0, 100].
    """
    if isinstance(prob_illicit, bool) or not isinstance(prob_illicit, (int, float)):
        raise ValueError(f"Invalid probability type: {type(prob_illicit).__name__}")
    prob_val = float(prob_illicit)
    if math.isnan(prob_val) or math.isinf(prob_val) or prob_val < 0.0 or prob_val > 1.0:
        raise ValueError(f"Illicit probability must be between 0.0 and 1.0, got {prob_illicit}")
    return int(round(prob_val * 100))


def validate_risk_score(score: int) -> int:
    """Ensure risk score is an integer in [0, 100] before sending an on-chain transaction."""
    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError(f"Risk score must be an integer, got {type(score).__name__}")
    if score < 0 or score > 100:
        raise ValueError(f"Risk score must be between 0 and 100, got {score}")
    return score


def normalize_ai_record(
    record: Dict[str, Any],
    default_model_version: str = "hybrid-gat-resnet-v1",
) -> AIRiskPrediction:
    """
    Normalize a raw prediction dictionary from AI model outputs into AIRiskPrediction.

    Supported keys:
      - Identifier: 'tx_id', 'txId', or 'node_idx'
      - Score / probability:
          * 'risk_score' (int in 0..100), OR
          * 'prob_illicit' / 'illicit_prob' (float in 0.0..1.0), OR
          * 'probabilities' ([prob_licit, prob_illicit] from Softmax / predict_proba)
    """
    raw_id = record.get("tx_id", record.get("txId", record.get("node_idx")))
    if raw_id is None:
        raise ValueError(f"Missing transaction identifier ('tx_id' or 'txId') in record: {record}")
    tx_id = int(raw_id)
    if tx_id < 0:
        raise ValueError(f"Transaction identifier must be non-negative, got {tx_id}")

    prob_illicit: Optional[float] = None
    if "risk_score" in record and record["risk_score"] is not None and str(record["risk_score"]) != "":
        raw_score = float(record["risk_score"])
        if not raw_score.is_integer():
            raise ValueError(f"Risk score must be an integer value, got {record['risk_score']}")
        risk_score = validate_risk_score(int(raw_score))
        if "prob_illicit" in record and record["prob_illicit"] is not None:
            prob_illicit = float(record["prob_illicit"])
    elif "prob_illicit" in record and record["prob_illicit"] is not None:
        prob_illicit = float(record["prob_illicit"])
        risk_score = prob_to_risk_score(prob_illicit)
    elif "illicit_prob" in record and record["illicit_prob"] is not None:
        prob_illicit = float(record["illicit_prob"])
        risk_score = prob_to_risk_score(prob_illicit)
    elif "probabilities" in record and record["probabilities"] is not None:
        probs = list(record["probabilities"])
        if len(probs) != 2:
            raise ValueError(f"Expected 2 class probabilities [Licit, Illicit], got {probs}")
        prob_illicit = float(probs[1])
        risk_score = prob_to_risk_score(prob_illicit)
    else:
        raise ValueError(
            f"Record for tx_id={tx_id} must contain 'risk_score', 'prob_illicit', or 'probabilities'"
        )

    predicted_label = str(
        record.get(
            "predicted_label",
            "Illicit" if risk_score >= 50 else "Licit",
        )
    )
    model_version = str(record.get("model_version", default_model_version))

    return AIRiskPrediction(
        tx_id=tx_id,
        risk_score=risk_score,
        prob_illicit=prob_illicit,
        predicted_label=predicted_label,
        model_version=model_version,
    )


def extract_predictions_from_model_probs(
    tx_ids: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    model_version: str = "gat-resnet-v1",
) -> List[AIRiskPrediction]:
    """
    Convert 2-class probability outputs (e.g. GATResNet Softmax output [N, 2] or
    sklearn predict_proba [N, 2]) into a list of AIRiskPrediction objects.
    """
    if len(tx_ids) != len(probabilities):
        raise ValueError(
            f"Length mismatch: {len(tx_ids)} tx_ids vs {len(probabilities)} probability rows"
        )
    predictions: List[AIRiskPrediction] = []
    for tx_id, prob_row in zip(tx_ids, probabilities):
        predictions.append(
            normalize_ai_record(
                {
                    "tx_id": int(tx_id),
                    "probabilities": list(prob_row),
                    "model_version": model_version,
                }
            )
        )
    return predictions


def load_ai_predictions_from_file(file_path: Union[str, Path]) -> List[AIRiskPrediction]:
    """
    Load AI predictions from a JSON or CSV batch output file.
    Raises FileNotFoundError or ValueError transparently if the file is missing or malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"AI prediction input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload.get("predictions", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"JSON file must contain a list of prediction objects: {path}")
        return [normalize_ai_record(item) for item in records]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [normalize_ai_record(row) for row in reader]

    raise ValueError(f"Unsupported AI prediction file format '{suffix}'. Use .json or .csv.")


def run_sklearn_model_inference(
    model_path: Union[str, Path],
    feature_csv_path: Union[str, Path],
    limit: int = 10,
) -> List[AIRiskPrediction]:
    """
    Load a trained scikit-learn baseline artifact (.joblib) from `models/` and run
    predict_proba on scaled feature rows to generate AIRiskPrediction items.
    Raises FileNotFoundError transparently if model or feature file is missing.
    """
    m_path = Path(model_path)
    f_path = Path(feature_csv_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found at: {m_path}")
    if not f_path.is_file():
        raise FileNotFoundError(f"Feature CSV file not found at: {f_path}")

    import joblib
    import pandas as pd

    model = joblib.load(m_path)
    df = pd.read_csv(f_path, nrows=limit)

    meta_cols = {"node_idx", "txId", "tx_id", "time_step_raw", "y"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if len(feature_cols) != 166:
        raise ValueError(
            f"Feature schema mismatch: expected 166 features, got {len(feature_cols)}"
        )

    if "txId" in df.columns:
        tx_ids = df["txId"].astype(int).tolist()
    elif "tx_id" in df.columns:
        tx_ids = df["tx_id"].astype(int).tolist()
    elif "node_idx" in df.columns:
        tx_ids = df["node_idx"].astype(int).tolist()
    else:
        tx_ids = list(range(len(df)))

    probs = model.predict_proba(df[feature_cols].to_numpy())
    return extract_predictions_from_model_probs(
        tx_ids=tx_ids,
        probabilities=probs.tolist(),
        model_version=m_path.stem,
    )


def get_default_sample_predictions() -> List[AIRiskPrediction]:
    """
    Sample batch of AI predictions covering all 3 compliance tiers (ALLOW, WARN, BLOCK)
    using Elliptic transaction IDs for local testnet verification.
    """
    sample_records = [
        {
            "tx_id": 230425980,
            "prob_illicit": 0.12,
            "predicted_label": "Licit",
            "model_version": "gat-resnet-demo-v1",
        },
        {
            "tx_id": 230425981,
            "prob_illicit": 0.58,
            "predicted_label": "Illicit",
            "model_version": "gat-resnet-demo-v1",
        },
        {
            "tx_id": 232022460,
            "prob_illicit": 0.91,
            "predicted_label": "Illicit",
            "model_version": "gat-resnet-demo-v1",
        },
    ]
    return [normalize_ai_record(rec) for rec in sample_records]


def resolve_sender_account():
    """
    Resolve the transaction signer account via Ape Framework.
    Uses APE_ACCOUNT_ALIAS if configured in environment; otherwise falls back to
    accounts.test_accounts[0] on local test networks.
    """
    account_alias = os.environ.get("APE_ACCOUNT_ALIAS", "").strip()
    if account_alias:
        return accounts.load(account_alias)
    if len(accounts.test_accounts) > 0:
        return accounts.test_accounts[0]
    raise RuntimeError(
        "No test accounts available on current network and APE_ACCOUNT_ALIAS is not set."
    )


def get_or_deploy_gatekeeper(sender=None, contract_address: Optional[str] = None):
    """
    Connect to an existing AMLGatekeeper contract via CONTRACT_ADDRESS (or argument),
    or deploy a fresh instance on the active local testnet using Ape Framework.
    """
    signer = sender or resolve_sender_account()
    resolved_address = (contract_address or os.environ.get("CONTRACT_ADDRESS", "")).strip()

    if resolved_address:
        return project.AMLGatekeeper.at(resolved_address), signer, False

    contract = project.AMLGatekeeper.deploy(sender=signer)
    return contract, signer, True


def submit_prediction_onchain(
    contract,
    prediction: AIRiskPrediction,
    sender=None,
) -> OnChainSubmissionResult:
    """
    Send an on-chain transaction via Ape to record a suspicious transaction's AML risk score
    and verify both the emitted event and stored contract state.
    """
    signer = sender or resolve_sender_account()
    validate_risk_score(prediction.risk_score)

    receipt = contract.record_tx_risk(
        prediction.tx_id,
        prediction.risk_score,
        sender=signer,
    )

    if receipt.failed:
        raise RuntimeError(
            f"Transaction failed for tx_id={prediction.tx_id} (tx_hash={receipt.txn_hash})"
        )

    # Verify emitted RiskScoreUpdated event in receipt
    decoded_events = list(receipt.decode_logs(contract.RiskScoreUpdated))
    event_verified = False
    if decoded_events:
        evt = decoded_events[-1]
        event_verified = (
            int(evt.tx_id) == prediction.tx_id
            and int(evt.risk_score) == prediction.risk_score
            and str(evt.submitter).lower() == str(signer.address).lower()
        )

    # Read back on-chain state from AMLGatekeeper view functions
    stored = contract.get_tx_assessment(prediction.tx_id)
    decision_code = int(stored.decision)
    decision_label = str(contract.get_decision_label(decision_code))
    is_blocked = bool(contract.is_tx_blocked(prediction.tx_id))

    state_verified = (
        bool(stored.assessed)
        and int(stored.risk_score) == prediction.risk_score
        and int(contract.risk_score()) == prediction.risk_score
        and int(contract.current_decision()) == decision_code
    )

    if not event_verified or not state_verified:
        raise RuntimeError(
            f"On-chain verification mismatch for tx_id={prediction.tx_id}: "
            f"event_verified={event_verified}, state_verified={state_verified}"
        )

    chain_id = int(networks.provider.chain_id)

    return OnChainSubmissionResult(
        tx_id=prediction.tx_id,
        risk_score=prediction.risk_score,
        decision_code=decision_code,
        decision_label=decision_label,
        is_blocked=is_blocked,
        tx_hash=str(receipt.txn_hash),
        block_number=int(receipt.block_number),
        gas_used=int(receipt.gas_used),
        chain_id=chain_id,
        contract_address=str(contract.address),
        submitter=str(signer.address),
        event_verified=event_verified,
        state_verified=state_verified,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def submit_batch_predictions(
    predictions: Iterable[AIRiskPrediction],
    contract=None,
    sender=None,
) -> List[OnChainSubmissionResult]:
    """Submit a sequence of AIRiskPrediction items on-chain and return verified results."""
    signer = sender or resolve_sender_account()
    active_contract = contract
    if active_contract is None:
        active_contract, signer, _ = get_or_deploy_gatekeeper(sender=signer)

    results: List[OnChainSubmissionResult] = []
    for pred in predictions:
        res = submit_prediction_onchain(active_contract, pred, sender=signer)
        results.append(res)
    return results


def main():
    """Entrypoint for `ape run ape_bridge`."""
    model_path = os.environ.get("MODEL_PATH", "").strip()
    feature_csv_path = os.environ.get("FEATURE_INPUT_PATH", "").strip()
    input_file = os.environ.get("AI_OUTPUT_PATH", "").strip()

    if model_path:
        if not feature_csv_path:
            raise ValueError(
                "FEATURE_INPUT_PATH must be provided when MODEL_PATH is set."
            )
        print(f"[INFO] Running inference with model '{model_path}' on '{feature_csv_path}'")
        predictions = run_sklearn_model_inference(model_path, feature_csv_path)
    elif input_file:
        print(f"[INFO] Loading AI predictions from file: {input_file}")
        predictions = load_ai_predictions_from_file(input_file)
    else:
        print(
            "[INFO] Using sample AI model predictions "
            "(ALLOW / WARN / BLOCK tiers) for local testnet integration run."
        )
        predictions = get_default_sample_predictions()

    contract, signer, was_deployed = get_or_deploy_gatekeeper()
    chain_id = int(networks.provider.chain_id)
    network_name = networks.provider.network.name

    action_str = "Deployed new" if was_deployed else "Connected to existing"
    print(f"[INFO] Network: {network_name} (chain_id={chain_id})")
    print(f"[INFO] {action_str} AMLGatekeeper at: {contract.address}")
    print(f"[INFO] Submitter / Owner account: {signer.address}")
    print(
        f"[INFO] Active thresholds: WARN >= {contract.warn_threshold()}, "
        f"BLOCK >= {contract.block_threshold()}\n"
    )

    results = submit_batch_predictions(predictions, contract=contract, sender=signer)

    print("=== ON-CHAIN AML RISK SUBMISSION RESULTS ===")
    for item in results:
        print(
            f"- tx_id={item.tx_id} | score={item.risk_score}/100 "
            f"| decision={item.decision_label} ({item.decision_code}) "
            f"| blocked={item.is_blocked} | gas={item.gas_used} "
            f"| block={item.block_number} | tx_hash={item.tx_hash}"
        )

    print(
        f"\n[SUCCESS] Total assessments recorded on-chain: {contract.total_assessed()} "
        f"| Latest score: {contract.risk_score()} "
        f"({contract.get_decision_label(contract.current_decision())})"
    )

    output_json = os.environ.get("BRIDGE_RECEIPT_OUT", "").strip()
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"[INFO] Saved transaction receipts to {out_path}")


if __name__ == "__main__":
    with networks.parse_network_choice("ethereum:local:test"):
        main()
