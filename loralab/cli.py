"""LoRALab CLI: evaluate adaptation trade-offs, enforce gates both sides."""

from __future__ import annotations

import argparse
import json
import sys

from .eval import build_fixture, format_terminal, run_evaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="loralab",
        description="Low-rank adaptation with measured accuracy, cost, and retention.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--corpus-per-class", type=int, default=40)
        p.add_argument("--test-per-class", type=int, default=15)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--lr", type=float, default=0.5)

    p_eval = sub.add_parser("evaluate")
    common(p_eval)
    p_eval.add_argument("--format", choices=("terminal", "json", "markdown"),
                        default="terminal")

    p_gate = sub.add_parser("gates")
    common(p_gate)
    p_gate.add_argument("--min-base-drop", type=float, default=None,
                        help="minimum accuracy gap between base-on-A and base-on-B "
                             "(proves the shift is real)")
    p_gate.add_argument("--min-adapted-accuracy", type=float, default=None,
                        help="minimum acc on B for the best rank-2 adapter")
    p_gate.add_argument("--max-rank2-param-share", type=float, default=None,
                        help="maximum trainable-param share for the rank-2 adapter")
    p_gate.add_argument("--max-merge-diff", type=float, default=None,
                        help="maximum merged-vs-adapter logit diff")

    p_coord = sub.add_parser("coordinates")
    p_coord.add_argument("--input", required=True, help="strict JSON problem file, or - for stdin")
    p_coord.add_argument("--gauge-scale", type=float, default=4.0)
    p_coord.add_argument("--learning-rate", type=float, default=0.1)
    p_trace = sub.add_parser("trajectory")
    p_trace.add_argument("--rank", type=int, choices=(1, 2, 4, 8), default=2)
    p_trace.add_argument("--epoch", type=int, default=30)
    p_trace.add_argument("--batch", choices=("first_train", "full_train"), default="first_train")
    p_trace.add_argument("--gauge-scale", type=float, default=4.0)
    p_trace.add_argument("--learning-rate", type=float, default=0.1)

    args = parser.parse_args(argv)
    if args.command in ("coordinates", "trajectory"):
        try:
            if args.command == "coordinates":
                from .coordinate_report import report_json
                if args.input == "-":
                    text = sys.stdin.read(1_000_001)
                else:
                    from pathlib import Path
                    with Path(args.input).open(encoding="utf-8") as source:
                        text = source.read(1_000_001)
                print(report_json(text, args.gauge_scale, args.learning_rate))
            else:
                import gzip
                from importlib.resources import files
                from .coordinate_fixtures import trajectory_report
                archive = json.loads(gzip.decompress(files("loralab").joinpath(
                    "coordinate_data/6-original-epochs.json.gz").read_bytes()))
                result = trajectory_report(archive, args.rank, args.epoch, args.batch,
                                           args.gauge_scale, args.learning_rate)
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        except (ValueError, OSError, UnicodeError) as exc:
            print("coordinate error: " + str(exc), file=sys.stderr)
            return 2
        return 0
    fx = build_fixture(args.corpus_per_class, args.test_per_class,
                       base_epochs=args.epochs, lr=args.lr)
    results = run_evaluation(fx)

    if args.command == "evaluate":
        if args.format == "json":
            print(json.dumps(results, indent=2, sort_keys=True))
        elif args.format == "markdown":
            print("| method | acc on B | retention on A | params | share |")
            print("|---|---|---|---|---|")
            for e in results["lora"]:
                print(f"| lora r={e['rank']} | {e['acc_B']:.4f} | {e['retention_A']:.4f} "
                      f"| {e['trainable_params']} | {e['param_share']:.1%} |")
            f = results["full_ft"]
            print(f"| full ft | {f['acc_B']:.4f} | {f['retention_A']:.4f} "
                  f"| {f['trainable_params']} | 100.0% |")
        else:
            print(format_terminal(results))
        return 0

    failures = []
    drop = results["base_acc_A"] - results["base_acc_B"]
    if args.min_base_drop is not None and drop < args.min_base_drop:
        failures.append(f"base accuracy drop {drop:.4f} < required {args.min_base_drop:.4f} "
                        "(shift no longer exercises adaptation)")
    rank2 = next((e for e in results["lora"] if e["rank"] == 2), None)
    if args.min_adapted_accuracy is not None and (rank2 is None or rank2["acc_B"] < args.min_adapted_accuracy):
        failures.append(f"rank-2 adapted accuracy {rank2 and rank2['acc_B']} "
                        f"< required {args.min_adapted_accuracy}")
    if args.max_rank2_param_share is not None and (rank2 is None or rank2["param_share"] > args.max_rank2_param_share):
        failures.append(f"rank-2 param share {rank2 and rank2['param_share']:.3f} "
                        f"> allowed {args.max_rank2_param_share}")
    if args.max_merge_diff is not None and results["merge_max_abs_diff"] > args.max_merge_diff:
        failures.append(f"merge diff {results['merge_max_abs_diff']:.2e} "
                        f"> allowed {args.max_merge_diff:.2e}")

    print(format_terminal(results))
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
