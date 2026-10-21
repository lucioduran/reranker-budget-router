import argparse
import json
from pathlib import Path

from .core import BM25, read, route, validate
from .experiment import configuration, download, run, verify


def main() -> int:
    parser = argparse.ArgumentParser(description="Local reranking with hard candidate budgets")
    parser.add_argument("--root", type=Path, default=Path("."))
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--cache", type=Path, default=Path(".cache/models"))
    execute = sub.add_parser("run")
    execute.add_argument("--split", choices=["dev", "eval"], required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--policies", type=Path)
    execute.add_argument("--cache", type=Path, default=Path(".cache/models"))
    check = sub.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    decide = sub.add_parser("route")
    decide.add_argument("question")
    decide.add_argument("--budget", type=int, default=12)
    decide.add_argument("--policies", type=Path, default=Path("results/dev/policies.json"))
    decide.add_argument("--policy", default="router-b12-p0")
    args = parser.parse_args()
    try:
        if args.command == "download":
            download(configuration(args.root / "config/experiment.json"), args.cache)
        elif args.command == "run":
            run(args.root, args.output, args.split, args.cache, args.policies)
        elif args.command == "verify":
            print(json.dumps(verify(args.root, args.output)))
        else:
            docs, _ = validate(
                read(args.root / "data/corpus.json"), read(args.root / "data/queries.json")
            )
            engine = BM25(docs)
            retrieval = engine.retrieve(args.question)
            signals = engine.signals(args.question, retrieval)
            policy = next(
                (p for p in read(args.policies)["policies"] if p["name"] == args.policy), None
            )
            if policy is None:
                raise ValueError("Unknown policy")
            depth = route(signals, policy | {"budget": args.budget}, len(retrieval))
            print(
                json.dumps(
                    {
                        "depth": depth,
                        "signals": signals,
                        "candidate_ids": [d for d, _ in retrieval[:depth]],
                    }
                )
            )
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
