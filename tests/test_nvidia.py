from torchwatch.collector.nvidia import MockCollector, create_collector


def test_mock_collector_shape():
    collector = MockCollector(count=4, seed=1)
    samples = collector.sample()
    assert len(samples) == 4
    for gpu in samples:
        assert gpu.utilization_pct is not None and 0 <= gpu.utilization_pct <= 100
        assert 0 < gpu.vram_used_bytes <= gpu.vram_total_bytes
        assert 0.0 <= gpu.vram_pct <= 100.0


def test_create_collector_never_raises():
    collector, _reason = create_collector()
    assert len(collector.sample()) == collector.gpu_count()
    collector.close()
