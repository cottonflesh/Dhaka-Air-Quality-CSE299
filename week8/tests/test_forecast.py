"""Basic checks for the Week 8 forecast code and saved models."""
from pathlib import Path
import sys
import unittest
import json

import numpy as np
import pandas as pd
import joblib

WEEK8 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK8 / 'src'))
from forecast_core import CONTEXT_HOURS, baseline, build_features, forecast_ml

class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.history = np.arange(1, CONTEXT_HOURS + 1, dtype=float)
        self.origin = pd.Timestamp('2024-03-01 23:00:00')

    def test_daily_baseline(self):
        result = baseline(self.history, 168, 'Daily repeat')
        self.assertEqual(len(result), 168)
        np.testing.assert_equal(result[:24], self.history[-24:])
        np.testing.assert_equal(result[24:48], self.history[-24:])

    def test_persistence(self):
        result = baseline(self.history, 24, 'Persistence')
        np.testing.assert_equal(result, np.full(24, 168.))

    def test_feature_rows(self):
        for horizon in (24, 168):
            features = build_features(self.history, 'BARC', self.origin, horizon)
            self.assertEqual(len(features), horizon)
            self.assertEqual(int(features.lead_hour.iloc[-1]), horizon)
            self.assertTrue(np.isfinite(features.to_numpy()).all())
            self.assertEqual(features.recent_pm25.iloc[0], self.history[-1])

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            build_features(self.history[:24], 'BARC', self.origin, 24)

    def test_model_outputs(self):
        for horizon in (24, 168):
            bundle = joblib.load(WEEK8 / 'models' / f'pm25_{horizon}h_gradient_boosting.joblib')
            self.assertEqual(bundle['horizon'], horizon)
            self.assertEqual(bundle['context_hours'], 168)
            forecast = forecast_ml(bundle['model'], self.history, 'BARC', self.origin, horizon)
            self.assertEqual(forecast.shape, (horizon,))
            self.assertTrue(np.isfinite(forecast).all())
            self.assertTrue((forecast >= 0).all())

if __name__ == '__main__':
    unittest.main()
