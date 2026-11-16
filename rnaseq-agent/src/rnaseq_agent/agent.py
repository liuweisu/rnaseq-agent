"""A minimal tool-calling agent, written without an agent framework on purpose.

The loop is short enough to read in one sitting:
  1. send the conversation and the tool schemas to the model
  2. if the model asks for a tool, run it and append the result
  3. repeat until the model answers in plain text

It talks to any OpenAI-compatible chat endpoint. The default is a local
Ollama server, so no data leaves the machine.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from .tools import TOOL_NAMES, TOOL_SPECS, Workspace

SYSTEM_PROMPT = """You are a careful bioinformatics assistant for bulk RNA-seq data.
Rules:
- Use the tools to get every number, gene name and sample name. Never state a
  value that did not come from a tool result in this conversation.
- If a tool says something was not found, say so. Do not guess.
- Answer in two or three sentences and name the thresholds you used."""

DEFAULT_URL = os.environ.get("RNASEQ_AGENT_URL", "http://localhost:11434/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("RNASEQ_AGENT_MODEL", "qwen2.5:7b")

# A "chat function" takes (messages, tools) and returns the assistant message
# dict. Injecting it keeps the loop testable without a running model.
ChatFn = Callable[[list[dict], list[dict]], dict]


def ollama_chat(messages: list[dict], tools: list[dict]) -> dict:
    response = requests.post(
        DEFAULT_URL,
        json={"model": DEFAULT_MODEL, "messages": messages, "tools": tools, "temperature": 0},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


@dataclass
class Run:
    """What happened in one question: the answer plus every tool call made."""

    question: str
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    stopped_early: bool = False

    def to_markdown(self) -> str:
        lines = [f"## {self.question}", "", self.answer or "_No answer._", "", "**Tool calls**", ""]
        for call in self.tool_calls:
            lines.append(f"- `{call['name']}({json.dumps(call['arguments'])})`")
        if not self.tool_calls:
            lines.append("- none")
        return "\n".join(lines) + "\n"


def run_tool(workspace: Workspace, name: str, arguments: dict) -> dict:
    """Run one tool. Errors go back to the model as data, they do not crash the run."""
    if name not in TOOL_NAMES:
        return {"error": f"Unknown tool {name!r}. Available: {TOOL_NAMES}"}
    try:
        return getattr(workspace, name)(**arguments)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:  # the model sees bad arguments
        return {"error": f"{type(exc).__name__}: {exc}"}


def ask(question: str, workspace: Workspace, chat: ChatFn = ollama_chat, max_steps: int = 6) -> Run:
    run = Run(question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        message = chat(messages, TOOL_SPECS)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            run.answer = (message.get("content") or "").strip()
            return run
        for call in calls:
            name = call["function"]["name"]
            raw = call["function"].get("arguments") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                arguments = {}
            result = run_tool(workspace, name, arguments)
            run.tool_calls.append({"name": name, "arguments": arguments, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "content": json.dumps(result),
                }
            )
    run.stopped_early = True
    run.answer = "Stopped: the step limit was reached before the model gave an answer."
    return run
