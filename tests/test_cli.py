import io
import json
import unittest
from contextlib import redirect_stdout

from loralab.cli import main


def run_cli(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


class TestCli(unittest.TestCase):
    def test_evaluate_terminal(self):
        code, out = run_cli("evaluate")
        self.assertEqual(code, 0)
        self.assertIn("lora r=2", out)
        self.assertIn("full ft", out)

    def test_evaluate_json(self):
        code, out = run_cli("evaluate", "--format", "json")
        data = json.loads(out)
        self.assertEqual(data["base_acc_A"], 1.0)
        self.assertLess(data["base_acc_B"], 0.5)

    def test_achievable_gates_pass(self):
        code, out = run_cli("gates", "--min-base-drop", "0.3",
                            "--min-adapted-accuracy", "0.95",
                            "--max-rank2-param-share", "0.7",
                            "--max-merge-diff", "1e-12")
        self.assertEqual(code, 0)
        self.assertIn("All gates passed", out)

    def test_impossible_gate_fails(self):
        code, out = run_cli("gates", "--max-rank2-param-share", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("GATE FAILURES", out)

    def test_shift_gate_bites_when_shift_gone(self):
        code, out = run_cli("gates", "--min-base-drop", "0.9")
        self.assertEqual(code, 1)
        self.assertIn("shift", out)


if __name__ == "__main__":
    unittest.main()
