import pytest


@pytest.fixture
def gatekeeper(project, accounts):
    owner = accounts[0]
    return project.AMLGatekeeper.deploy(sender=owner)


def test_set_risk_score_updates_state(gatekeeper, accounts):
    owner = accounts[0]
    risk_score = 55

    gatekeeper.set_risk_score(risk_score, sender=owner)

    assert gatekeeper.risk_score() == risk_score
    assert gatekeeper.current_decision() == 1
    assert gatekeeper.total_assessed() == 1