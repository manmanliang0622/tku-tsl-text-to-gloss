import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import metrics  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_evaluate_reports_groups_ngrams_and_bootstrap(self):
        refs = ["我/水/喝/要", "你/好", "今天/天氣/非常/好"]
        hyps = ["我/水/喝/要", "你/好", "今天/天氣/好/非常"]
        groups = ["dialogue:a", "dialogue:a", "dialogue:b"]
        result = metrics.evaluate(
            refs, hyps, {"我", "水", "喝", "要", "你", "好", "今天", "天氣", "非常"},
            groups=groups, bootstrap_samples=100, bootstrap_seed=7)

        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["Reference4Grams"], 2)
        self.assertEqual(result["Hypothesis4Grams"], 2)
        self.assertEqual(len(result["BLEU-4_95%CI"]), 2)
        self.assertLessEqual(
            result["BLEU-4_95%CI"][0], result["BLEU-4_95%CI"][1])
        self.assertEqual(result["BLEU-bootstrap-unit"], "group")

    def test_group_bootstrap_is_deterministic(self):
        refs = [metrics.tokenize(x) for x in ["a/b/c/d", "a/b/c/d", "e/f/g/h"]]
        hyps = [metrics.tokenize(x) for x in ["a/b/c/d", "a/b/c/x", "e/f/g/h"]]
        groups = ["g1", "g1", "g2"]
        first = metrics.bootstrap_bleu_ci(
            refs, hyps, groups, n_samples=200, seed=42)
        second = metrics.bootstrap_bleu_ci(
            refs, hyps, groups, n_samples=200, seed=42)
        self.assertEqual(first, second)

    def test_group_length_must_match(self):
        with self.assertRaises(AssertionError):
            metrics.evaluate(["a/b"], ["a/b"], {"a", "b"}, groups=[])


if __name__ == "__main__":
    unittest.main()
