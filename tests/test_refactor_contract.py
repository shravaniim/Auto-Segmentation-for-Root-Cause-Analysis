import numpy as np

from core.candidate_filtering import derive_min_abs_count, derive_min_support, passes_support_filter
from metrics.drift_metrics import calculate_psi
from metrics.performance_metrics import calculate_auc_gini, calculate_ks
from metrics.significance import two_proportion_ztest_pvalue, ks_score_shift_pvalue, adjust_p_values_bh
from techniques.base import BaseSegmentationTechnique


class DummyTechnique(BaseSegmentationTechnique):
    def generate_candidates(self):
        return []

    def evaluate_candidates(self, candidates):
        return []

    def rank_candidates(self, candidates):
        return []

    def run(self):
        return {"status": "ok"}


def test_base_interface_can_be_subclassed():
    technique = DummyTechnique()
    assert technique.run() == {"status": "ok"}


def test_filtering_helpers_match_expected_thresholds():
    assert derive_min_abs_count(0.1, 30, None) == 300
    assert derive_min_support(300, 1000, 1000, None) == 0.3
    assert passes_support_filter(300, 300, 0.3, 0.3, 300, 0.3, 0.35)
    assert not passes_support_filter(300, 300, 0.4, 0.3, 300, 0.3, 0.35)


def test_metrics_are_calculated_consistently():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    auc, gini = calculate_auc_gini(y_true, y_score)
    assert np.isclose(auc, 1.0)
    assert np.isclose(gini, 1.0)
    assert np.isclose(calculate_ks(y_true, y_score), 1.0)
    assert np.isclose(calculate_psi(0.2, 0.4), 0.4 * np.log(2.0))

    p_value = two_proportion_ztest_pvalue(10, 100, 20, 100)
    assert np.isfinite(p_value)
    assert np.isfinite(ks_score_shift_pvalue(np.array([0.1, 0.2]), np.array([0.8, 0.9])))

    adjusted = adjust_p_values_bh(np.array([0.01, 0.04, 0.03]))
    assert adjusted.shape == (3,)
