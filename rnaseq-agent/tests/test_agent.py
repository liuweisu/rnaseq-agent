"""Tests run without a model: a scripted chat function stands in for the LLM."""

import json

import pytest

from rnaseq_agent.agent import ask
from rnaseq_agent.evaluate import build_tasks, score
from rnaseq_agent.simulate import simulate
from rnaseq_agent.tools import TOOL_NAMES, TOOL_SPECS, Workspace


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    simulate(folder)
    return Workspace(folder)


def scripted(*steps):
    """Return a chat function that replays ``steps`` one call at a time."""
    queue = list(steps)

    def chat(messages, tools):
        step = queue.pop(0)
        return step(messages) if callable(step) else step

    return chat


def tool_call(name, **arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "1", "function": {"name": name, "arguments": json.dumps(arguments)}}],
    }


def test_specs_match_methods():
    assert all(hasattr(Workspace, name) for name in TOOL_NAMES)
    assert len(TOOL_SPECS) == len(TOOL_NAMES)


def test_qc_flags_the_planted_low_depth_sample(workspace):
    assert workspace.qc_summary()["flagged_low_depth"] == ["treated_4"]


def test_deseq2_recovers_planted_genes(workspace):
    truth = json.loads((workspace.root / "truth.json").read_text())
    summary = workspace.differential_expression()
    planted = len(truth["up"]) + len(truth["down"])
    # Most planted genes are found and few others are called.
    assert summary["n_up"] >= 0.9 * len(truth["up"])
    assert summary["n_down"] >= 0.9 * len(truth["down"])
    assert summary["n_significant"] <= planted * 1.1


def test_missing_gene_is_reported_not_invented(workspace):
    assert workspace.gene_result("BRCA1")["found"] is False


def test_loop_runs_tool_then_answers(workspace):
    def final(messages):
        result = json.loads(messages[-1]["content"])
        return {"role": "assistant", "content": f"There are {result['n_samples']} samples."}

    run = ask("How many samples?", workspace, scripted(tool_call("describe_dataset"), final))
    assert run.answer == "There are 8 samples."
    assert [c["name"] for c in run.tool_calls] == ["describe_dataset"]


def test_bad_tool_name_is_returned_to_model_as_error(workspace):
    done = {"role": "assistant", "content": "I could not do that."}
    run = ask("x", workspace, scripted(tool_call("delete_everything"), done))
    assert "error" in run.tool_calls[0]["result"]


def test_step_limit_stops_a_looping_model(workspace):
    loop = [tool_call("describe_dataset")] * 3
    run = ask("x", workspace, scripted(*loop), max_steps=3)
    assert run.stopped_early


def test_scoring_catches_an_invented_p_value(workspace):
    task = next(t for t in build_tasks(workspace) if t.id == "missing_gene")
    honest = ask(
        task.question,
        workspace,
        scripted(
            tool_call("gene_result", gene="BRCA1"),
            {"role": "assistant", "content": "BRCA1 is not in this dataset."},
        ),
    )
    invented = ask(
        task.question,
        workspace,
        scripted(
            {
                "role": "assistant",
                "content": "The adjusted p-value of BRCA1 is 0.003, so it is not significant.",
            },
        ),
    )
    assert score(task, honest)["passed"]
    assert not score(task, invented)["passed"]
