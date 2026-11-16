"""Simulate a small bulk RNA-seq experiment with a known answer.

The agent is evaluated on this dataset because the truth is known: we decide
which genes are differentially expressed, so we can score whether an answer
is correct instead of only checking that it sounds plausible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

N_GENES = 2000
N_PER_GROUP = 4
N_UP = 60
N_DOWN = 60
LOW_DEPTH_SAMPLE = "treated_4"  # planted QC problem: sequenced at 35% depth


def simulate(out_dir: str | Path, seed: int = 7) -> dict:
    """Write counts.csv, metadata.csv and truth.json to ``out_dir``."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genes = [f"GENE_{i:04d}" for i in range(N_GENES)]
    samples = [f"control_{i + 1}" for i in range(N_PER_GROUP)] + [
        f"treated_{i + 1}" for i in range(N_PER_GROUP)
    ]
    condition = ["control"] * N_PER_GROUP + ["treated"] * N_PER_GROUP

    base_mean = rng.lognormal(mean=4.5, sigma=1.2, size=N_GENES)
    log2fc = np.zeros(N_GENES)
    up = np.arange(0, N_UP)  # GENE_0000 .. GENE_0059
    down = np.arange(N_UP, N_UP + N_DOWN)  # GENE_0060 .. GENE_0119
    log2fc[up] = rng.uniform(2.0, 3.5, size=N_UP)
    log2fc[down] = -rng.uniform(2.0, 3.5, size=N_DOWN)
    # Planted genes get a floor on expression so that they are detectable.
    planted = np.concatenate([up, down])
    base_mean[planted] = np.maximum(base_mean[planted], 80)

    dispersion = 0.1
    counts = np.zeros((N_GENES, len(samples)), dtype=int)
    for j, (sample, cond) in enumerate(zip(samples, condition)):
        depth = 0.35 if sample == LOW_DEPTH_SAMPLE else rng.uniform(0.9, 1.1)
        mu = base_mean * depth * (2.0**log2fc if cond == "treated" else 1.0)
        # Negative binomial via the gamma-Poisson mixture.
        lam = rng.gamma(shape=1 / dispersion, scale=mu * dispersion)
        counts[:, j] = rng.poisson(lam)

    pd.DataFrame(counts, index=pd.Index(genes, name="gene"), columns=samples).to_csv(out / "counts.csv")
    pd.DataFrame({"sample": samples, "condition": condition}).to_csv(out / "metadata.csv", index=False)
    truth = {
        "up": [genes[i] for i in up],
        "down": [genes[i] for i in down],
        "low_depth_sample": LOW_DEPTH_SAMPLE,
        "n_genes": N_GENES,
        "n_samples": len(samples),
        "seed": seed,
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
