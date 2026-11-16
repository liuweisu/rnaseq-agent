"""The analysis tools the agent is allowed to call.

Design rules:
- Every number the agent reports must come from one of these functions.
- Tools only read and write inside one workspace folder, so the agent cannot
  be talked into opening other files on the machine.
- Tools return small JSON-serialisable dicts, so the model sees a summary and
  the full tables stay on disk.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


class Workspace:
    """A folder holding counts.csv and metadata.csv, plus results we write."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / "counts.csv").exists():
            raise FileNotFoundError(f"No counts.csv in {self.root}")
        self._de_cache: dict[tuple, pd.DataFrame] = {}

    # -- data access -------------------------------------------------------
    def counts(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "counts.csv", index_col=0)

    def metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "metadata.csv").set_index("sample")

    # -- tools -------------------------------------------------------------
    def describe_dataset(self) -> dict:
        """Number of genes and samples, and how samples split over conditions."""
        counts, meta = self.counts(), self.metadata()
        return {
            "n_genes": int(counts.shape[0]),
            "n_samples": int(counts.shape[1]),
            "samples_per_condition": meta["condition"].value_counts().to_dict(),
            "samples": list(counts.columns),
        }

    def qc_summary(self) -> dict:
        """Library size and detected genes per sample, with outlier flags."""
        counts = self.counts()
        lib = counts.sum(axis=0)
        detected = (counts > 0).sum(axis=0)
        median = float(lib.median())
        flagged = [s for s in counts.columns if lib[s] < 0.5 * median]
        return {
            "median_library_size": int(median),
            "per_sample": {
                s: {
                    "library_size": int(lib[s]),
                    "fraction_of_median": round(float(lib[s] / median), 2),
                    "genes_detected": int(detected[s]),
                }
                for s in counts.columns
            },
            "flagged_low_depth": flagged,
            "rule": "flagged when library size is below 50% of the median",
        }

    def _run_deseq2(self, reference: str, treated: str) -> pd.DataFrame:
        key = (reference, treated)
        if key in self._de_cache:
            return self._de_cache[key]
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        counts, meta = self.counts(), self.metadata()
        levels = set(meta["condition"])
        for level in (reference, treated):
            if level not in levels:
                raise ValueError(f"Unknown condition {level!r}; available: {sorted(levels)}")
        keep = meta.index[meta["condition"].isin([reference, treated])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dds = DeseqDataSet(
                counts=counts[keep].T, metadata=meta.loc[keep], design="~condition", quiet=True
            )
            dds.deseq2()
            stats = DeseqStats(dds, contrast=["condition", treated, reference], quiet=True)
            stats.summary()
        res = stats.results_df.sort_values("padj")
        res.to_csv(self.root / f"deseq2_{treated}_vs_{reference}.csv")
        self._de_cache[key] = res
        return res

    def differential_expression(
        self,
        reference: str = "control",
        treated: str = "treated",
        alpha: float = 0.05,
        min_abs_log2fc: float = 1.0,
    ) -> dict:
        """Run DESeq2 (pydeseq2) and summarise significant genes."""
        res = self._run_deseq2(reference, treated)
        sig = res[(res["padj"] < alpha) & (res["log2FoldChange"].abs() >= min_abs_log2fc)]
        top = sig.head(10)
        return {
            "contrast": f"{treated} vs {reference}",
            "thresholds": {"padj_below": alpha, "abs_log2fc_at_least": min_abs_log2fc},
            "n_tested": int(res["padj"].notna().sum()),
            "n_significant": len(sig),
            "n_up": int((sig["log2FoldChange"] > 0).sum()),
            "n_down": int((sig["log2FoldChange"] < 0).sum()),
            "top_genes": [
                {"gene": g, "log2fc": round(float(r.log2FoldChange), 2), "padj": float(f"{r.padj:.2e}")}
                for g, r in top.iterrows()
            ],
            "results_file": f"deseq2_{treated}_vs_{reference}.csv",
        }

    def gene_result(self, gene: str, reference: str = "control", treated: str = "treated") -> dict:
        """DESeq2 result for one gene. Says so plainly when the gene is absent."""
        res = self._run_deseq2(reference, treated)
        if gene not in res.index:
            return {"gene": gene, "found": False, "message": "This gene is not in the dataset."}
        row = res.loc[gene]
        padj = None if np.isnan(row.padj) else float(f"{row.padj:.2e}")
        return {
            "gene": gene,
            "found": True,
            "log2fc": round(float(row.log2FoldChange), 2),
            "padj": padj,
            "direction": "up" if row.log2FoldChange > 0 else "down",
            "significant_at_padj_0.05_and_abs_log2fc_1": bool(
                padj is not None and padj < 0.05 and abs(row.log2FoldChange) >= 1
            ),
        }


# JSON schemas the model sees. Kept next to the functions so they stay in sync.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "describe_dataset",
            "description": "Number of genes, samples and samples per condition.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qc_summary",
            "description": "Library size and detected genes per sample; flags low-depth samples.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "differential_expression",
            "description": "Run DESeq2 for treated vs reference; counts and top significant genes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string"},
                    "treated": {"type": "string"},
                    "alpha": {"type": "number"},
                    "min_abs_log2fc": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gene_result",
            "description": "DESeq2 log2 fold change and adjusted p-value for one gene.",
            "parameters": {
                "type": "object",
                "properties": {"gene": {"type": "string"}},
                "required": ["gene"],
            },
        },
    },
]
TOOL_NAMES = [spec["function"]["name"] for spec in TOOL_SPECS]
