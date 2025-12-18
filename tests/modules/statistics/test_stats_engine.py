import pytest

from coherent.engine.stats_engine import StatsEngine


@pytest.fixture
def stats():
    return StatsEngine()


def test_describe(stats):
    data = [1, 2, 3, 4]
    result = stats.describe(data)
    assert result["count"] == 4
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["quartiles"]["q1"] == pytest.approx(1.75)
    assert result["quartiles"]["q3"] == pytest.approx(3.25)


def test_distribution_info_normal(stats):
    info = stats.distribution_info("normal", 0, params={"mean": 0, "std": 1})
    assert info.pdf == pytest.approx(0.3989, rel=1e-3)
    assert info.cdf == pytest.approx(0.5, rel=1e-3)


def test_distribution_info_uniform(stats):
    info = stats.distribution_info("uniform", 0.5, params={"a": 0, "b": 1})
    assert info.pdf == pytest.approx(1.0)
    assert info.cdf == pytest.approx(0.5)


def test_visualize(stats):
    data = [1, 2, 2, 3, 4]
    viz = stats.visualize(data, bins=2)
    assert len(viz["bins"]) == 2
    assert sum(bin_info["count"] for bin_info in viz["bins"]) == len(data)
