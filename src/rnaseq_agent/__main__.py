"""Command line: simulate data, ask a question, or run the evaluation."""

from __future__ import annotations

import argparse

from . import agent
from .evaluate import evaluate, save
from .simulate import simulate
from .tools import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(prog="rnaseq-agent", description=__doc__)
    parser.add_argument("--data", default="data/demo", help="workspace folder (default: data/demo)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("simulate", help="write a simulated dataset with a known answer")
    ask_parser = sub.add_parser("ask", help="ask one question")
    ask_parser.add_argument("question")
    sub.add_parser("eval", help="run every evaluation task and write a report")
    args = parser.parse_args()

    if args.command == "simulate":
        truth = simulate(args.data)
        print(f"Wrote {truth['n_genes']} genes x {truth['n_samples']} samples to {args.data}")
        return

    workspace = Workspace(args.data)
    if args.command == "ask":
        print(agent.ask(args.question, workspace).to_markdown())
    else:
        results = evaluate(workspace)
        path = save(results, agent.DEFAULT_MODEL, "results")
        passed = sum(r["passed"] for r in results)
        print(f"{passed} of {len(results)} tasks passed. Report: {path}")


if __name__ == "__main__":
    main()
