"""Acceptance tests for AlertLog: dedup, lingering, and expiry."""

from torchwatch.alerts import AlertLog


def test_reported_alert_is_active():
    log = AlertLog(ttl_s=10.0)
    log.report("vram:0", "vram high", now=0.0)
    (alert,) = log.active(now=0.0)
    assert alert.key == "vram:0"
    assert alert.message == "vram high"
    assert alert.first_seen == 0.0
    assert alert.last_seen == 0.0


def test_empty_log_has_no_active_alerts():
    assert AlertLog().active(now=100.0) == []


def test_rereport_dedups_by_key():
    log = AlertLog(ttl_s=10.0)
    log.report("vram:0", "vram at 95.1%", now=0.0)
    log.report("vram:0", "vram at 96.4%", now=1.0)
    (alert,) = log.active(now=1.0)
    assert alert.message == "vram at 96.4%"  # latest numbers win
    assert alert.first_seen == 0.0  # but the alert is still the same event
    assert alert.last_seen == 1.0


def test_alert_lingers_after_condition_stops():
    log = AlertLog(ttl_s=10.0)
    log.report("spike", "loss spike", now=0.0)
    assert len(log.active(now=5.0)) == 1  # condition gone, still readable
    assert len(log.active(now=10.0)) == 1  # boundary: now - last_seen == ttl


def test_alert_expires_after_ttl():
    log = AlertLog(ttl_s=10.0)
    log.report("spike", "loss spike", now=0.0)
    assert log.active(now=10.1) == []


def test_rereport_refreshes_lifespan():
    log = AlertLog(ttl_s=10.0)
    log.report("stall", "loss stalled", now=0.0)
    log.report("stall", "loss stalled", now=8.0)
    assert len(log.active(now=15.0)) == 1  # 15 - 8 = 7 < ttl
    assert log.active(now=18.1) == []


def test_active_orders_by_first_seen():
    log = AlertLog(ttl_s=100.0)
    log.report("b-later-refire", "b", now=0.0)
    log.report("a-newcomer", "a", now=1.0)
    log.report("b-later-refire", "b", now=2.0)  # re-fire must not reorder
    assert [a.key for a in log.active(now=2.0)] == ["b-later-refire", "a-newcomer"]


def test_expired_alert_can_fire_again_as_new():
    log = AlertLog(ttl_s=10.0)
    log.report("spike", "loss spike", now=0.0)
    assert log.active(now=50.0) == []  # long gone
    log.report("spike", "loss spike again", now=50.0)
    (alert,) = log.active(now=50.0)
    assert alert.first_seen == 50.0  # a fresh event, not the old one revived
