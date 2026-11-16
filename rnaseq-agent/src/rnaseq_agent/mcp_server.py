"""Expose the same tools over the Model Context Protocol (MCP).

This lets any MCP client (a desktop assistant, an IDE) use the tools without
this repo's own agent loop. Run with:

    RNASEQ_AGENT_DATA=data/demo python -m rnaseq_agent.mcp_server
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from .tools import Workspace

mcp = MCPServer("rnaseq-agent")
_workspace: Workspace | None = None


def workspace() -> Workspace:
    global _workspace
    if _workspace is None:
        _workspace = Workspace(os.environ.get("RNASEQ_AGENT_DATA", "data/demo"))
    return _workspace


@mcp.tool()
def describe_dataset() -> dict:
    """Number of genes, samples and samples per condition."""
    return workspace().describe_dataset()


@mcp.tool()
def qc_summary() -> dict:
    """Library size and detected genes per sample; flags low-depth samples."""
    return workspace().qc_summary()


@mcp.tool()
def differential_expression(
    reference: str = "control", treated: str = "treated", alpha: float = 0.05, min_abs_log2fc: float = 1.0
) -> dict:
    """Run DESeq2 for treated vs reference; counts and top significant genes."""
    return workspace().differential_expression(reference, treated, alpha, min_abs_log2fc)


@mcp.tool()
def gene_result(gene: str) -> dict:
    """DESeq2 log2 fold change and adjusted p-value for one gene."""
    return workspace().gene_result(gene)


if __name__ == "__main__":
    mcp.run()
