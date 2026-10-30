import pytest
from ape import reverts


DECISION_ALLOW = 0
DECISION_WARN = 1
DECISION_BLOCK = 2


@pytest.fixture
def gatekeeper(project, accounts):
    owner = accounts[0]
    return project.AMLGatekeeper.deploy(sender=owner)


def test_set_risk_score_updates_state(gatekeeper, accounts):
    owner = accounts[0]
    risk_score = 55

    gatekeeper.set_risk_score(risk_score, sender=owner)

    assert gatekeeper.risk_score() == risk_score
    assert gatekeeper.current_decision() == DECISION_WARN
    assert gatekeeper.total_assessed() == 1


@pytest.mark.parametrize(
    ("risk_score", "expected_decision"),
    [
        (0, DECISION_ALLOW),
        (39, DECISION_ALLOW),
        (40, DECISION_WARN),
        (74, DECISION_WARN),
        (75, DECISION_BLOCK),
        (100, DECISION_BLOCK),
    ],
    ids=[
        "allow-minimum",
        "allow-upper-boundary",
        "warn-lower-boundary",
        "warn-upper-boundary",
        "block-lower-boundary",
        "block-maximum",
    ],
)
def test_set_risk_score_classifies_threshold_boundaries(
    gatekeeper, accounts, risk_score, expected_decision
):
    owner = accounts[0]

    gatekeeper.set_risk_score(risk_score, sender=owner)

    assert gatekeeper.risk_score() == risk_score
    assert gatekeeper.current_decision() == expected_decision
    assert gatekeeper.total_assessed() == 1


def test_set_risk_score_rejects_score_above_maximum(gatekeeper, accounts):
    owner = accounts[0]

    with reverts("Risk score must be between 0 and 100"):
        gatekeeper.set_risk_score(101, sender=owner)

    assert gatekeeper.risk_score() == 0
    assert gatekeeper.total_assessed() == 0


def test_set_risk_score_rejects_unauthorized_caller(gatekeeper, accounts):
    unauthorized_account = accounts[1]

    with reverts("Caller is not authorized"):
        gatekeeper.set_risk_score(55, sender=unauthorized_account)

    assert gatekeeper.risk_score() == 0
    assert gatekeeper.total_assessed() == 0
