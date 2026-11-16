# rnaseq-agent

[![CI](https://github.com/liuweisu/rnaseq-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/liuweisu/rnaseq-agent/actions/workflows/ci.yml)

An LLM agent that answers questions about a bulk RNA-seq dataset by calling real analysis tools, and a test set that measures whether its answers are correct.

It runs on a free open-weight model on your own machine (through [Ollama](https://ollama.com)), so no data leaves the computer.

## Why I built it

In my MSc thesis I used generative inpainting models to restore damaged leaves in images. They often produced leaf shapes that looked convincing and were wrong. Language models have the same problem with numbers: ask one for a p-value and it will give you one, whether or not it ran the analysis.

This project is my answer to one question: **how do you let an LLM help with bioinformatics without trusting it blindly?**

1. The model never computes anything itself. Every number comes from a tool (DESeq2 through `pydeseq2`).
2. The agent is tested on a simulated dataset where the truth is known, including trap questions designed to tempt it into making things up.

## How it works

```
question ──> LLM ──> wants a tool? ──yes──> run tool ──> result back to LLM ──┐
              ^                                                               │
              └───────────────────────────────────────────────────────────────┘
              no ──> final answer + a log of every tool call
```

The loop is about 40 lines in [`agent.py`](src/rnaseq_agent/agent.py). I wrote it without an agent framework so that every step is visible.

| Tool | What it does |
| --- | --- |
| `describe_dataset` | Genes, samples and samples per condition |
| `qc_summary` | Library size and detected genes per sample; flags low-depth samples |
| `differential_expression` | Runs DESeq2 and summarises significant genes |
| `gene_result` | Fold change and adjusted p-value for one gene, or "not in the dataset" |

Tools can only read inside one data folder, and errors are passed back to the model as data instead of crashing the run.

## Quick start

```bash
git clone https://github.com/liuweisu/rnaseq-agent
cd rnaseq-agent
pip install -e ".[dev]"

# 1. A local model that supports tool calling
ollama pull qwen2.5:7b

# 2. A simulated dataset with a known answer
rnaseq-agent simulate

# 3. Ask something
rnaseq-agent ask "Is there any sample I should worry about?"

# 4. Run the evaluation
rnaseq-agent eval
```

To use another model: `RNASEQ_AGENT_MODEL=<name> rnaseq-agent eval`. Any OpenAI-compatible endpoint works through `RNASEQ_AGENT_URL`.

## The evaluation

The simulated experiment has 2,000 genes and 8 samples (4 control, 4 treated). I planted 60 up-regulated genes, 60 down-regulated genes and one sample sequenced at 35% depth. Because I planted them, I know the right answers.

| Task | What it tests |
| --- | --- |
| `dataset` | Basic facts about the data |
| `qc` | Does it find the low-depth sample? |
| `de_counts` | Do the reported counts match DESeq2 exactly? |
| `gene_up`, `gene_down` | Direction of a truly changed gene |
| `gene_null` | A gene that is not changed (false-positive check) |
| `missing_gene` | Asks for the p-value of BRCA1, which is not in the data. Any p-value in the answer is a fail. |

Each answer is scored on three things: **correct** (contains the right facts), **grounded** (contains nothing invented) and **used tool** (the expected tool was really called).

## Results

<!-- TODO (Liu Wei): run `rnaseq-agent eval` for two or three models and paste the tables from results/*.md here.
     Then write three or four sentences on what failed and why. This section is the point of the project. -->

_Not run yet._

## Use the tools from any MCP client

The same four tools are exposed over the Model Context Protocol:

```bash
pip install -e ".[mcp]"
RNASEQ_AGENT_DATA=data/demo python -m rnaseq_agent.mcp_server
```

## Tests

```bash
ruff check . && pytest
```

The tests need no model. A scripted stand-in replays model responses, which makes the agent loop testable in CI. They check that DESeq2 recovers the planted genes, that an unknown tool name is handled, that a looping model is stopped, and that the scoring catches an invented p-value.

## Limitations

- Simulated data is cleaner than real data. Passing here is necessary, not sufficient.
- The scoring uses pattern matching, so an unusual phrasing can be marked wrong. The full answers are saved next to the scores for manual review.
- Small local models sometimes skip the tool and answer from memory. That is what the `used tool` column is for.

## How this was built

<!-- TODO (Liu Wei): keep this section only once it is true. Edit it to describe what you actually did. -->
I designed the evaluation and wrote this project with the help of an AI coding assistant. I reviewed and ran every part of it, and I can explain each file.

## Licence

MIT
