"""Run both splits under an exclusive local Unix benchmark lock."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(".cache/benchmark.lock"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a") as lock:
        print("Waiting for exclusive benchmark lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = os.environ | {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for split in ("dev", "eval"):
            command = [
                sys.executable,
                "-m",
                "budget_router",
                "run",
                "--split",
                split,
                "--output",
                str(args.output / split),
            ]
            if split == "eval":
                command += ["--policies", str(args.output / "dev/policies.json")]
            subprocess.run(command, env=env, check=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "budget_router",
                    "verify",
                    "--output",
                    str(args.output / split),
                ],
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
