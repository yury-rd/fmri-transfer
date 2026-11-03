import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from fmri_response import response
from fmri_response.controls import (
    audit_predictor_feature_lineage,
    frozen_complete_lineages,
)


class ResponseTests(unittest.TestCase):
    def test_holdout_matches_direct_fit(self):
        rng = np.random.default_rng(20260918)
        fields = rng.normal(size=(7, 3, 5, 13))
        x, _, y, _ = response.fields(fields)
        with threadpool_limits(limits=1):
            prediction, _ = response.lopo(x, y)
            for held in range(len(fields)):
                train = np.arange(len(fields)) != held
                expected = x[held] @ response.fit(x[train], y[train])
                np.testing.assert_allclose(prediction[held], expected, rtol=0, atol=1e-11)

    def test_held_target_does_not_enter_predictors(self):
        fields = np.random.default_rng(20260918).normal(size=(7, 3, 5, 13))
        before = response.fields(fields)[0]
        fields[:, 2, 4] += 1000
        np.testing.assert_array_equal(response.fields(fields)[0], before)

    def test_memorization_oracle_is_excluded_from_controls(self):
        audits = {
            item.predictor: audit_predictor_feature_lineage(item, target_path="FA")
            for item in frozen_complete_lineages()
        }
        self.assertFalse(audits["memorization_oracle"].admissible)
        self.assertEqual(
            {name for name, audit in audits.items() if audit.admissible},
            {"atlas", "universal", "additive", "seen_path", "static_identity",
             "future_base", "reliability", "design_summary"},
        )


if __name__ == "__main__":
    unittest.main()
