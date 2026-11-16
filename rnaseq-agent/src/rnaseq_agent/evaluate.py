"""Score the agent on questions whose answers we know.

Three checks per task:
- correct:  the answer contains the facts it must contain
- grounded: the answer contains none of the facts it must not contain
            (for example a p-value for a gene that is not in the dataset)
- used_tool: the expected tool was actually called

The expected values are computed from the simulated truth and from calling the
tools directly, never typed in by hand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .agent import ChatFn, Run, ask, ollama_chat
from .tools import Workspace


@dataclass
class Task:
    id: str
    question: str
    expected_tool: str
    must_match: list[str]  # regex patterns that must all appear (case-insensitive)
    must_not_match: list[str]  # regex patterns that must not appear


def build_tasks(workspace: Workspace) -> list[Task]:
    truth = json.loads((workspace.root / "truth.json").read_text())
    de = workspace.differential_expression()
    up_gene, down_gene = truth["up"][7], truth["down"][7]
    null_gene = "GENE_1500"
    return [
        Task(
            "dataset",
            "How many samples are in this dataset and which conditions do they belong to?",
            "describe_dataset",
            [rf"\b{truth['n_samples']}\b|\beight\b", "control", "treated"],
            [],
        ),
        Task(
            "qc",
            "Is there any sample I should worry about before running the analysis?",
            "qc_summary",
            [re.escape(truth["low_depth_sample"])],
            [],
        ),
        Task(
            "de_counts",
            "How many genes are significantly up and how many are down in treated versus control?",
            "differential_expression",
            [rf"\b{de['n_up']}\b", rf"\b{de['n_down']}\b"],
            [],
        ),
        Task(
            "gene_up",
            f"Is {up_gene} differentially expressed, and in which direction?",
            "gene_result",
            [r"\bup"],
            [r"\bnot (significant|differentially)"],
        ),
        Task(
            "gene_down",
            f"Is {down_gene} differentially expressed, and in which direction?",
            "gene_result",
            [r"\bdown"],
            [r"\bnot (significant|differentially)"],
        ),
        Task(
            "gene_null",
            f"Is {null_gene} differentially expressed?",
            "gene_result",
            [r"\b(not|no)\b"],
            [],
        ),
        Task(
            "missing_gene",
            "What is the adjusted p-value of BRCA1 in this dataset?",
            "gene_result",
            [r"\b(not|no)\b"],
            # A made-up p-value is the failure this task exists to catch.
            [r"\b\d+(\.\d+)?e-\d+\b", r"\b0\.\d+\b"],
        ),
    ]


def score(task: Task, run: Run) -> dict:
    answer = run.answer.lower()
    correct = all(re.search(p, answer) for p in task.must_match)
    grounded = not any(re.search(p, answer) for p in task.must_not_match)
    used_tool = any(call["name"] == task.expected_tool for call in run.tool_calls)
    return {
        "task": task.id,
        "correct": correct,
        "grounded": grounded,
        "used_tool": used_tool,
        "passed": correct and grounded and used_tool,
        "answer": run.answer,
        "tools_called": [call["name"] for call in run.tool_calls],
    }


def evaluate(workspace: Workspace, chat: ChatFn = ollama_chat) -> list[dict]:
    return [score(task, ask(task.question, workspace, chat)) for task in build_tasks(workspace)]


def to_markdown(results: list[dict], model: str) -> str:
    passed = sum(r["passed"] for r in results)
    lines = [
        f"# Evaluation: {model}",
        "",
        f"**{passed} of {len(results)} tasks passed.**",
        "",
        "| Task | Correct | Grounded | Used tool | Passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    mark = {True: "yes", False: "NO"}
    for r in results:
        lines.append(
            f"| {r['task']} | {mark[r['correct']]} | {mark[r['grounded']]} "
            f"| {mark[r['used_tool']]} | {mark[r['passed']]} |"
        )
    lines += ["", "## Answers", ""]
    for r in results:
        lines += [f"**{r['task']}** (tools: {', '.join(r['tools_called']) or 'none'})", "", r["answer"], ""]
    return "\n".join(lines)


def save(results: list[dict], model: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-")
    (out / f"{name}.json").write_text(json.dumps(results, indent=2))
    path = out / f"{name}.md"
    path.write_text(to_markdown(results, model))
    return path
