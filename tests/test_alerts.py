"""Acceptance tests for alert rules."""

from torchwatch.alerts import is_spiking, is_stalled, temp_warning, vram_suggestion


def test_no_suggestion_below_alert_threshold():
    assert vram_suggestion(50.0) is None
    assert vram_suggestion(94.9) is None  # warn zone colors, but no nagging


def test_suggestion_at_alert_threshold():
    hint = vram_suggestion(95.0)
    assert hint is not None
    lowered = hint.lower()
    assert "batch" in lowered  # cheapest fix must be mentioned
    assert "amp" in lowered or "precision" in lowered
    assert "checkpoint" in lowered


def test_flat_loss_stalls():
    assert is_stalled([1.0] * 100)


def test_stall_needs_a_full_window():
    assert not is_stalled([1.0] * 99)
    assert not is_stalled([])


def test_improving_loss_is_not_stalled():
    losses = [1.0 - 0.005 * i for i in range(100)]
    assert not is_stalled(losses)


def test_stall_looks_only_at_recent_window():
    # improved long ago, flat for the last 100 → stalled
    losses = [2.0 - 0.01 * i for i in range(100)] + [0.5] * 100
    assert is_stalled(losses)


def test_spike_detected():
    assert is_spiking([1.0] * 19 + [2.5])


def test_steady_loss_no_spike():
    assert not is_spiking([1.0] * 20)


def test_spike_needs_a_full_window():
    assert not is_spiking([1.0, 5.0])


def test_no_temp_warning_below_alert_threshold():
    assert temp_warning(72) is None
    assert temp_warning(99.9) is None  # warn zone colors the panel, no alert yet


def test_no_temp_warning_when_sensor_unreadable():
    assert temp_warning(None) is None


def test_temp_warning_at_alert_threshold():
    hint = temp_warning(100.0)
    assert hint is not None
    lowered = hint.lower()
    # must name at least one physical check — heat isn't fixed in code
    assert "airflow" in lowered or "fan" in lowered or "cool" in lowered
    assert temp_warning(107.0) is not None  # and stays on above it
