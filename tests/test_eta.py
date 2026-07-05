"""M4 acceptance tests for the ETA estimator and formatter."""

import pytest

from torchwatch.eta import EtaEstimator, format_eta


def test_steady_rate_and_eta():
    est = EtaEstimator()
    for i in range(10):
        est.observe(i * 0.5, i)  # 2 steps per second
    assert est.steps_per_sec() == pytest.approx(2.0)
    # at step 9 of 29: 20 steps left at 2/s
    assert est.eta_seconds(9, 29) == pytest.approx(10.0)


def test_rate_uses_only_the_window():
    est = EtaEstimator(window=4)
    for i in range(4):
        est.observe(float(i), i)  # slow: 1 step/s
    for j, t in enumerate((3.1, 3.2, 3.3)):
        est.observe(t, 13 + 10 * j)  # fast burst
    # window holds (3.0,3),(3.1,13),(3.2,23),(3.3,33): 30 steps in 0.3s
    assert est.steps_per_sec() == pytest.approx(100.0)


def test_not_computable_yet():
    est = EtaEstimator()
    assert est.steps_per_sec() is None
    assert est.eta_seconds(5, 100) is None
    est.observe(1.0, 5)
    assert est.steps_per_sec() is None


def test_repeated_steps_and_zero_timespan():
    est = EtaEstimator()
    # tqdm redraws the same step; also all-same timestamps must not divide by zero
    est.observe(1.0, 7)
    est.observe(1.0, 7)
    assert est.steps_per_sec() is None  # zero time span → unknowable
    est.observe(2.0, 7)
    # 0 steps in 1s: rate 0 → no finite ETA
    assert est.steps_per_sec() == pytest.approx(0.0)
    assert est.eta_seconds(7, 100) is None


def test_step_regression_resets():
    est = EtaEstimator()
    est.observe(0.0, 400)
    est.observe(1.0, 450)
    assert est.steps_per_sec() == pytest.approx(50.0)
    est.observe(2.0, 0)  # new epoch: inner-loop counter restarted
    assert est.steps_per_sec() is None  # history discarded
    est.observe(3.0, 50)
    assert est.steps_per_sec() == pytest.approx(50.0)


def test_format_eta():
    assert format_eta(None) == "—"
    assert format_eta(42) == "~42s"
    assert format_eta(872) == "~14m 32s"
    assert format_eta(3725) == "~1h 2m"
    assert format_eta(0) == "~0s"
