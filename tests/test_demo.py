from torchwatch.collector.demo import DemoMetrics


def test_demo_source_shape():
    demo = DemoMetrics(total_steps=50, seed=1)
    updates = [demo.next_update() for _ in range(120)]
    assert all(u.loss is not None and u.loss > 0 for u in updates)
    assert all(u.total_steps == 50 for u in updates)
    assert updates[0].step == 1
    assert updates[49].step == 50
    assert updates[50].step == 1  # loops, exercising the ETA reset path
