import numpy as np

from fremont_bridge_analysis.analysis_utils import robust_zscore


def test_robust_zscore_centers_on_median() -> None:
    scores = robust_zscore(np.array([1.0, 2.0, 3.0, 100.0]))

    assert scores[1] < 0 < scores[2]
    assert scores[-1] > 10


def test_robust_zscore_constant_input_returns_zero() -> None:
    scores = robust_zscore(np.array([4.0, 4.0, 4.0]))

    np.testing.assert_array_equal(scores, np.zeros(3))
