# pragma version ^0.4.0
"""
@title AML Compliance Gatekeeper
@notice On-chain compliance contract that stores AML risk scores (0-100),
        classifies transactions into 3 alert tiers (ALLOW / WARN / BLOCK),
        emits audit events, and enforces access control.
"""

# ==============================================================================
# CONSTANTS
# ==============================================================================
MAX_SCORE: public(constant(uint256)) = 100

DECISION_ALLOW: public(constant(uint8)) = 0  # Allow transaction
DECISION_WARN: public(constant(uint8)) = 1   # Warning / Manual review required
DECISION_BLOCK: public(constant(uint8)) = 2  # Block transaction

DEFAULT_WARN_THRESHOLD: constant(uint256) = 40
DEFAULT_BLOCK_THRESHOLD: constant(uint256) = 75


# ==============================================================================
# STRUCTS & EVENTS
# ==============================================================================
struct Assessment:
    risk_score: uint256
    decision: uint8
    timestamp: uint256
    assessed: bool


event RiskScoreUpdated:
    tx_id: indexed(uint256)
    risk_score: uint256
    decision: uint8
    submitter: indexed(address)
    timestamp: uint256


event ThresholdsUpdated:
    warn_threshold: uint256
    block_threshold: uint256
    updated_by: indexed(address)


event OperatorUpdated:
    operator: indexed(address)
    is_authorized: bool


event OwnershipTransferred:
    previous_owner: indexed(address)
    new_owner: indexed(address)


# ==============================================================================
# STATE VARIABLES
# ==============================================================================
owner: public(address)
authorized_operators: public(HashMap[address, bool])

# Thresholds for 3-tier classification (0 < warn_threshold < block_threshold <= 100)
warn_threshold: public(uint256)
block_threshold: public(uint256)

# Latest AML risk score and decision
risk_score: public(uint256)
current_decision: public(uint8)

# Per-transaction assessment records (indexed by tx_id / node_idx)
tx_assessments: public(HashMap[uint256, Assessment])
total_assessed: public(uint256)


# ==============================================================================
# CONSTRUCTOR
# ==============================================================================
@deploy
def __init__():
    self.owner = msg.sender
    self.authorized_operators[msg.sender] = True
    self.warn_threshold = DEFAULT_WARN_THRESHOLD
    self.block_threshold = DEFAULT_BLOCK_THRESHOLD
    self.risk_score = 0
    self.current_decision = DECISION_ALLOW
    self.total_assessed = 0


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================
@internal
@view
def _check_owner():
    assert msg.sender == self.owner, "Only owner can perform this action"


@internal
@view
def _check_authorized():
    assert (
        msg.sender == self.owner or self.authorized_operators[msg.sender]
    ), "Caller is not authorized"


@internal
@view
def _evaluate_decision(score: uint256) -> uint8:
    assert score <= MAX_SCORE, "Risk score must be between 0 and 100"
    if score >= self.block_threshold:
        return DECISION_BLOCK
    elif score >= self.warn_threshold:
        return DECISION_WARN
    else:
        return DECISION_ALLOW


# ==============================================================================
# WRITE FUNCTIONS
# ==============================================================================
@external
def set_risk_score(new_score: uint256):
    """
    @notice Update the latest AML risk score (v1 compatible) and classify decision.
    @param new_score AML risk score in range 0-100.
    """
    self._check_authorized()
    decision: uint8 = self._evaluate_decision(new_score)

    self.risk_score = new_score
    self.current_decision = decision
    self.total_assessed += 1

    log RiskScoreUpdated(
        tx_id=0,
        risk_score=new_score,
        decision=decision,
        submitter=msg.sender,
        timestamp=block.timestamp,
    )


@external
def record_tx_risk(tx_id: uint256, new_score: uint256) -> uint8:
    """
    @notice Record AML risk score for a specific transaction (tx_id) and return decision tier.
    @param tx_id Transaction identifier (Elliptic txId or node index).
    @param new_score AML risk score in range 0-100.
    @return Decision tier (0 = ALLOW, 1 = WARN, 2 = BLOCK).
    """
    self._check_authorized()
    decision: uint8 = self._evaluate_decision(new_score)

    self.risk_score = new_score
    self.current_decision = decision
    self.tx_assessments[tx_id] = Assessment(
        risk_score=new_score,
        decision=decision,
        timestamp=block.timestamp,
        assessed=True,
    )
    self.total_assessed += 1

    log RiskScoreUpdated(
        tx_id=tx_id,
        risk_score=new_score,
        decision=decision,
        submitter=msg.sender,
        timestamp=block.timestamp,
    )
    return decision


@external
def set_thresholds(new_warn: uint256, new_block: uint256):
    """
    @notice Update warning and blocking score thresholds.
    @param new_warn Score threshold for WARN tier.
    @param new_block Score threshold for BLOCK tier.
    """
    self._check_owner()
    assert new_warn > 0, "Warn threshold must be greater than 0"
    assert new_warn < new_block, "Warn threshold must be less than block threshold"
    assert new_block <= MAX_SCORE, "Block threshold must not exceed 100"

    self.warn_threshold = new_warn
    self.block_threshold = new_block

    log ThresholdsUpdated(
        warn_threshold=new_warn,
        block_threshold=new_block,
        updated_by=msg.sender,
    )


@external
def set_operator(operator: address, is_authorized: bool):
    """
    @notice Grant or revoke permission for an operator to submit risk scores.
    """
    self._check_owner()
    assert operator != empty(address), "Invalid operator address"
    self.authorized_operators[operator] = is_authorized

    log OperatorUpdated(operator=operator, is_authorized=is_authorized)


@external
def transfer_ownership(new_owner: address):
    """
    @notice Transfer contract ownership to a new address.
    """
    self._check_owner()
    assert new_owner != empty(address), "Invalid new owner address"
    previous_owner: address = self.owner
    self.owner = new_owner
    self.authorized_operators[new_owner] = True

    log OwnershipTransferred(previous_owner=previous_owner, new_owner=new_owner)


# ==============================================================================
# VIEW / PURE FUNCTIONS
# ==============================================================================
@external
@view
def classify_score(score: uint256) -> uint8:
    """
    @notice Preview the decision tier for a given score (0-100) under current thresholds.
    """
    return self._evaluate_decision(score)


@external
@pure
def get_decision_label(decision: uint8) -> String[10]:
    """
    @notice Return human-readable label for a decision tier for Dashboard display.
    """
    if decision == DECISION_ALLOW:
        return "ALLOW"
    elif decision == DECISION_WARN:
        return "WARN"
    elif decision == DECISION_BLOCK:
        return "BLOCK"
    else:
        return "UNKNOWN"


@external
@view
def get_tx_assessment(tx_id: uint256) -> Assessment:
    """
    @notice Retrieve the stored assessment record for a given tx_id.
    """
    return self.tx_assessments[tx_id]


@external
@view
def is_tx_blocked(tx_id: uint256) -> bool:
    """
    @notice Check whether a specific transaction has been assessed and blocked.
    """
    assessment: Assessment = self.tx_assessments[tx_id]
    return assessment.assessed and assessment.decision == DECISION_BLOCK
