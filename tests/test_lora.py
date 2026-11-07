import random
import unittest

from loralab.eval import DIM, build_fixture, full_params, run_evaluation
from loralab.lora import LoRAClassifier


class TestAdaptation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = build_fixture()
        cls.results = run_evaluation(cls.fx)

    def test_shift_is_real(self):
        r = self.results
        self.assertEqual(r["base_acc_A"], 1.0)
        self.assertLess(r["base_acc_B"], 0.5)

    def test_rank2_matches_full_ft_accuracy(self):
        r = self.results
        rank2 = next(e for e in r["lora"] if e["rank"] == 2)
        self.assertEqual(rank2["acc_B"], r["full_ft"]["acc_B"])

    def test_rank2_cheaper_than_full(self):
        r = self.results
        rank2 = next(e for e in r["lora"] if e["rank"] == 2)
        self.assertLess(rank2["trainable_params"], r["full_params"])

    def test_pinned_fixture_numbers(self):
        r = self.results
        rank1 = next(e for e in r["lora"] if e["rank"] == 1)
        self.assertAlmostEqual(rank1["acc_B"], 0.9333, places=4)
        self.assertAlmostEqual(r["full_ft"]["retention_A"], 0.9556, places=4)
        self.assertLess(r["merge_max_abs_diff"], 1e-12)

    def test_determinism(self):
        fx2 = build_fixture()
        self.assertEqual(run_evaluation(fx2), self.results)


class TestLoRAMechanics(unittest.TestCase):
    def test_zero_init_B_is_identity_adaptation(self):
        fx = build_fixture()
        adapter = LoRAClassifier(fx["base"], 2, random.Random(0))
        x, _ = fx["testB"][0]
        self.assertEqual(adapter.logits(x), fx["base"].logits(x))

    def test_trainable_params_formula(self):
        fx = build_fixture()
        adapter = LoRAClassifier(fx["base"], 3, random.Random(0))
        self.assertEqual(adapter.trainable_params(), 3 * DIM + 3 * 3)

    def test_merge_reproduces_logits(self):
        # Merged weights (W + B@A) x reproduce adapter logits W x + B(A x) to
        # within floating-point reassociation error; pinned below 1e-12.
        fx = build_fixture()
        adapter = LoRAClassifier(fx["base"], 4, random.Random(0))
        for _ in range(10):
            adapter.train_epoch(fx["trainB"][:10], 0.5)
        diff = 0.0
        for x, _ in fx["testB"]:
            for p, q in zip(adapter.logits(x), adapter.merged_logits(x)):
                diff = max(diff, abs(p - q))
        self.assertLess(diff, 1e-12)
        self.assertGreater(diff, 0.0)  # reassociation is real, not claimed away

    def test_gradients_move_loss_down(self):
        fx = build_fixture()
        adapter = LoRAClassifier(fx["base"], 2, random.Random(0))
        first = adapter.train_epoch(fx["trainB"], 0.5)
        for _ in range(10):
            last = adapter.train_epoch(fx["trainB"], 0.5)
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
