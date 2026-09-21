"""Tests for Module 2 (analytics / ML).

Checks the saved end-to-end pipeline accepts RAW passenger data (including a
missing age) and returns valid predictions, and that the reported metrics are sane.
"""
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYTICS = ROOT / "analytics"
PIPELINE_FILE = ANALYTICS / "best_titanic_pipeline.joblib"

try:
    import joblib
    import sklearn  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "scikit-learn / joblib not installed")
class SavedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.pipeline = joblib.load(PIPELINE_FILE)
        except ModuleNotFoundError as exc:  # e.g. imbalanced-learn missing
            raise unittest.SkipTest(f"cannot load pipeline: {exc}")

    def _raw_rows(self):
        return pd.DataFrame([
            {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0,
             "parch": 0, "fare": 80.0, "embarked": "S"},
            {"pclass": 3, "sex": "male", "age": None, "sibsp": 0,
             "parch": 0, "fare": 8.0, "embarked": "S"},   # missing age
        ])

    def test_predicts_on_raw_data_with_missing_age(self):
        preds = self.pipeline.predict(self._raw_rows())
        self.assertEqual(len(preds), 2)
        self.assertTrue(set(preds) <= {0, 1})

    def test_probabilities_are_valid(self):
        proba = self.pipeline.predict_proba(self._raw_rows())
        self.assertEqual(proba.shape, (2, 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        self.assertTrue(abs(proba.sum(axis=1) - 1).max() < 1e-6)

    def test_first_class_woman_more_likely_to_survive_than_third_class_man(self):
        proba = self.pipeline.predict_proba(self._raw_rows())[:, 1]
        self.assertGreater(proba[0], proba[1])


class ReportedMetricsTests(unittest.TestCase):
    def test_model_comparison_metrics_are_valid(self):
        df = pd.read_csv(ANALYTICS / "task10_model_comparison.csv")
        self.assertEqual(len(df), 3)
        for col in ["accuracy", "precision", "recall", "f1", "auc"]:
            self.assertTrue(df[col].between(0, 1).all(), col)

    def test_split_is_stratified(self):
        df = pd.read_csv(ANALYTICS / "task7_split_summary.csv")
        rates = df.set_index("dataset")["survived_percent"]
        self.assertLess(abs(rates["Training set"] - rates["Test set"]), 1.0)


if __name__ == "__main__":
    unittest.main()
