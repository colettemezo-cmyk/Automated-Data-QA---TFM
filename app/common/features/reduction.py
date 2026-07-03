"""Embedding dimensionality-reduction strategies (PC1, top-K, adaptive, raw).

The training pipeline computes a per-column reduction of the 768-d Arctic
embeddings before handing the features to LightGBM/XGBoost. Historically that
was a single scalar PC1 per column. This module generalises that step so the
training + QA pipelines can swap between:

  * `pc1`           — scalar PC1 per column (legacy default; K=1).
  * `topN`          — fixed top-N principal components per column (e.g. `top5`).
  * `adaptive_<v>`  — adaptive K per column to reach cumulative explained
                      variance >= v, capped at `K_MAX_DEFAULTS[v]`.
                      e.g. `adaptive_0.90`, `adaptive_0.95`, `adaptive_0.99`.
  * `raw`           — no projection; passes the full 768-d embedding through
                      as one feature per dimension (5,376 features for 7 cols).

The strategy controls both *fitting* (in `workflows/training/model_input.py`)
and *projection at inference* (in `common/features/inference_matrix.py`). The
fitted axes + per-column K are serialised into `model_1/reduction/` and the
manifest, so QA loads back exactly what training fit.

Backwards compatibility: legacy `model_1/pc1/{col}_pc1_axis.npy` artifacts are
still readable — `load_reduction_artifacts` falls back to the old layout when
the new `reduction/` directory is absent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bootstrap import ensure_app_on_path

ensure_app_on_path(__file__)

# Sensible per-target K caps. These are deliberately tight on the loose
# variance targets and looser on the strict ones: at 0.90 you mostly want
# the "knee" of the eigenspectrum; at 0.99 you have to honour the long tail
# (cardinality permitting).
K_MAX_DEFAULTS: dict[float, int] = {
    0.90: 32,
    0.95: 64,
    0.99: 128,
}

# Embedding dim hard-coded for the Arctic v2 model; used only to size raw-mode
# feature names. If you swap encoders this constant moves to common.config.
RAW_EMBED_DIM = 768

# Feature-name format. `_EMB_PC{i}` is 1-indexed so `_EMB_PC1` matches the
# legacy column name when k=1, preserving compatibility with existing
# model_1 manifests + the scored parquets they produced.
PC_FEATURE_FMT = "{col}_EMB_PC{i}"
RAW_FEATURE_FMT = "{col}_EMB_DIM{i:03d}"


@dataclass(frozen=True)
class ReductionStrategy:
    """How to reduce a 768-d embedding column to K features.

    Resolved from a string spec (`"pc1"`, `"top5"`, `"adaptive_0.90"`,
    `"raw"`) via `parse_strategy()`. Stored verbatim in the model_1 manifest
    so inference can reproduce the projection without re-parsing the spec.
    """

    name: str
    mode: str  # "fixed_k" | "adaptive" | "raw"
    fixed_k: int | None = None
    variance_target: float | None = None
    k_max: int | None = None
    k_min: int = 1

    @property
    def is_raw(self) -> bool:
        return self.mode == "raw"

    @property
    def is_adaptive(self) -> bool:
        return self.mode == "adaptive"

    @property
    def is_fixed_k(self) -> bool:
        return self.mode == "fixed_k"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "fixed_k": self.fixed_k,
            "variance_target": self.variance_target,
            "k_max": self.k_max,
            "k_min": self.k_min,
        }

    @classmethod
    def from_manifest_dict(cls, payload: dict[str, Any]) -> ReductionStrategy:
        return cls(
            name=str(payload["name"]),
            mode=str(payload["mode"]),
            fixed_k=payload.get("fixed_k"),
            variance_target=payload.get("variance_target"),
            k_max=payload.get("k_max"),
            k_min=int(payload.get("k_min", 1)),
        )


_ADAPTIVE_RE = re.compile(r"^adaptive_(?P<v>0(?:\.[0-9]+)?|1(?:\.0+)?)$")
_TOPK_RE = re.compile(r"^top(?P<k>[1-9][0-9]*)$")
_PCK_RE = re.compile(r"^pc(?P<k>[1-9][0-9]*)$")


def parse_strategy(
    spec: str,
    *,
    k_max_override: int | None = None,
) -> ReductionStrategy:
    """Resolve a string spec to a `ReductionStrategy`.

    Recognised specs:
      * `pc1`           — alias for `top1` (keeps its own canonical name for
                          backward-compat with existing model_1 artifacts).
      * `pcN`           — alias for `topN` (e.g. `pc5` == `top5`); canonicalises
                          to the `topN` name so manifests / caches are uniform.
      * `topN`          — fixed K = N (e.g. `top5`, `top10`).
      * `adaptive_v`    — adaptive K with `variance_target=v` and
                          `k_max=K_MAX_DEFAULTS.get(v, 128)`. `v` may be any
                          float in `(0, 1]` (e.g. `adaptive_0.9` and
                          `adaptive_0.90` both parse).
      * `raw`           — no projection, 768 features per column.

    `k_max_override` overrides the table lookup for adaptive specs. Ignored
    for the other modes.
    """
    spec = spec.strip().lower()
    if spec == "pc1":
        return ReductionStrategy(name="pc1", mode="fixed_k", fixed_k=1, k_max=1)

    # `pcN` (N >= 2) is a user-friendly alias for `topN`; normalise to the
    # canonical `topN` spec so downstream naming (cache keys, manifests,
    # reports) is identical whether the caller typed `pc5` or `top5`.
    m_pc = _PCK_RE.match(spec)
    if m_pc is not None:
        spec = f"top{int(m_pc.group('k'))}"

    m = _TOPK_RE.match(spec)
    if m is not None:
        k = int(m.group("k"))
        return ReductionStrategy(
            name=f"top{k}", mode="fixed_k", fixed_k=k, k_max=k
        )

    m = _ADAPTIVE_RE.match(spec)
    if m is not None:
        v = float(m.group("v"))
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"adaptive variance target must be in (0, 1]; got {v!r}"
            )
        # Round to 2 decimals for the manifest name so `adaptive_0.9` and
        # `adaptive_0.90` collapse to the same canonical id.
        v_rounded = round(v, 2)
        k_max = (
            k_max_override
            if k_max_override is not None
            else K_MAX_DEFAULTS.get(v_rounded, K_MAX_DEFAULTS.get(v, 128))
        )
        return ReductionStrategy(
            name=f"adaptive_{v_rounded:.2f}",
            mode="adaptive",
            variance_target=v_rounded,
            k_max=int(k_max),
        )

    if spec == "raw":
        return ReductionStrategy(
            name="raw", mode="raw", fixed_k=RAW_EMBED_DIM, k_max=RAW_EMBED_DIM
        )

    raise ValueError(
        f"Unknown reduction strategy spec {spec!r}. "
        "Expected one of: 'pc1', 'topN' (e.g. 'top5'), "
        "'adaptive_v' (e.g. 'adaptive_0.90'), or 'raw'."
    )


def feature_names_for_column(
    col: str, k: int, *, is_raw: bool = False
) -> list[str]:
    """Output feature column names produced by a reduction.

    For PC modes we use 1-indexed `_EMB_PC{i}` so K=1 yields the legacy
    `_EMB_PC1` name exactly (preserves model_1 manifest backward compat).
    Raw mode uses `_EMB_DIM{i:03d}` to make it obvious in feature
    importances that these are raw embedding dims, not PCs.
    """
    if is_raw:
        return [RAW_FEATURE_FMT.format(col=col, i=i + 1) for i in range(k)]
    return [PC_FEATURE_FMT.format(col=col, i=i + 1) for i in range(k)]


@dataclass
class ReductionArtifacts:
    """In-memory bundle of frozen per-column projection parameters.

    Loaded once at inference time (or model_2 training) from the artifacts
    that `save_reduction_artifacts` wrote during model_1 training. The
    `axes_per_col[col]` matrix has shape (k_col, dim); project a batch with
    `(vectors - mean) @ axes.T`. For raw strategies, `axes` is `None`
    (identity projection — caller must just copy the embedding through).
    """

    strategy: ReductionStrategy
    axes_per_col: dict[str, np.ndarray | None]
    mean_per_col: dict[str, np.ndarray]
    k_per_col: dict[str, int]
    achieved_cumvar_per_col: dict[str, float] = field(default_factory=dict)
    ratios_per_col: dict[str, np.ndarray] = field(default_factory=dict)
    legacy_pc1: bool = False  # true when loaded from the pre-strategy pc1/ layout

    def feature_names(self, embed_cols: list[str]) -> list[str]:
        names: list[str] = []
        for col in embed_cols:
            k = self.k_per_col[col]
            names.extend(
                feature_names_for_column(col, k, is_raw=self.strategy.is_raw)
            )
        return names


# ---------------------------------------------------------------------------
# Artifact (de)serialisation
# ---------------------------------------------------------------------------

REDUCTION_SUBDIR = "reduction"
LEGACY_PC1_SUBDIR = "pc1"
REDUCTION_MANIFEST = "reduction_manifest.json"


def reduction_dir(model_dir: Path) -> Path:
    return Path(model_dir) / REDUCTION_SUBDIR


def legacy_pc1_dir(model_dir: Path) -> Path:
    return Path(model_dir) / LEGACY_PC1_SUBDIR


def save_reduction_artifacts(
    reduction_dir_path: Path,
    strategy: ReductionStrategy,
    *,
    embed_cols: list[str],
    axes_per_col: dict[str, np.ndarray | None],
    mean_per_col: dict[str, np.ndarray],
    k_per_col: dict[str, int],
    achieved_cumvar_per_col: dict[str, float] | None = None,
    ratios_per_col: dict[str, np.ndarray] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist axes + mean + per-column metadata under `reduction_dir_path/`.

    `reduction_dir_path` is the FINAL directory (typically
    `<model_dir>/reduction/`) where the artifacts live, not its parent.

    Layout:
      reduction_dir_path/
        reduction_manifest.json   # strategy + per-column k/cum_var/ratios
        {col}_axes.npy            # (k, dim) — absent for raw mode
        {col}_mean.npy            # (dim,)
        {col}_ratios.npy          # (k,)   — absent for raw mode
    """
    rdir = Path(reduction_dir_path)
    rdir.mkdir(parents=True, exist_ok=True)
    achieved = dict(achieved_cumvar_per_col or {})
    ratios_in = dict(ratios_per_col or {})

    per_column: dict[str, dict[str, Any]] = {}
    for col in embed_cols:
        k = int(k_per_col[col])
        np.save(rdir / f"{col}_mean.npy", mean_per_col[col].astype(np.float32))
        if not strategy.is_raw:
            axes = axes_per_col[col]
            if axes is None:
                raise ValueError(
                    f"non-raw strategy {strategy.name!r} requires axes for {col!r}"
                )
            np.save(rdir / f"{col}_axes.npy", axes.astype(np.float32))
            ratios = ratios_in.get(col)
            if ratios is not None:
                np.save(rdir / f"{col}_ratios.npy", ratios.astype(np.float32))
        per_column[col] = {
            "k": k,
            "achieved_cumulative_variance": float(achieved.get(col, 0.0)),
            "ratios": (
                ratios_in[col].astype(np.float32).tolist()
                if col in ratios_in
                else []
            ),
        }

    manifest: dict[str, Any] = {
        "strategy": strategy.to_manifest_dict(),
        "embed_cols": list(embed_cols),
        "per_column": per_column,
    }
    if extra:
        manifest["extra"] = extra
    (rdir / REDUCTION_MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return rdir


def load_reduction_artifacts(
    model_dir: Path,
    embed_cols: list[str],
) -> ReductionArtifacts:
    """Load frozen reduction artifacts, with legacy PC1 fallback.

    `model_dir` is the model_1 (or comparison) directory. Subdirectories
    `reduction/` and `pc1/` are searched in that order. You may also pass
    one of those subdirectories directly — both layouts are detected.

    Resolution order:
      1. `model_dir/reduction/reduction_manifest.json` — new strategy-aware.
      2. `model_dir/pc1/{col}_pc1_axis.npy` — legacy PC1-only; promoted to a
         synthetic `pc1` strategy so downstream projection code is uniform.
    """
    model_dir = Path(model_dir)
    # Allow callers to pass `<model_dir>/reduction/` or `<model_dir>/pc1/` directly.
    if model_dir.name == REDUCTION_SUBDIR or model_dir.name == LEGACY_PC1_SUBDIR:
        model_dir = model_dir.parent

    rdir = reduction_dir(model_dir)
    manifest_path = rdir / REDUCTION_MANIFEST

    if manifest_path.exists():
        return _load_reduction_layout(rdir, manifest_path, embed_cols)

    pc1_dir = legacy_pc1_dir(model_dir)
    if any((pc1_dir / f"{col}_pc1_axis.npy").exists() for col in embed_cols):
        return _load_legacy_pc1_layout(pc1_dir, embed_cols)

    raise FileNotFoundError(
        f"No reduction artifacts found under {model_dir}. "
        f"Expected {manifest_path} (new layout) or {pc1_dir}/*_pc1_axis.npy "
        f"(legacy layout). Re-run training to export them."
    )


def _load_reduction_layout(
    rdir: Path,
    manifest_path: Path,
    embed_cols: list[str],
) -> ReductionArtifacts:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    strategy = ReductionStrategy.from_manifest_dict(payload["strategy"])
    per_column = payload.get("per_column", {})
    axes_per_col: dict[str, np.ndarray | None] = {}
    mean_per_col: dict[str, np.ndarray] = {}
    k_per_col: dict[str, int] = {}
    cumvar_per_col: dict[str, float] = {}
    ratios_per_col: dict[str, np.ndarray] = {}

    for col in embed_cols:
        col_meta = per_column.get(col, {})
        mean_path = rdir / f"{col}_mean.npy"
        if not mean_path.exists():
            raise FileNotFoundError(
                f"Reduction mean for {col!r} missing at {mean_path}."
            )
        mean_per_col[col] = np.load(mean_path).astype(np.float32)
        k = int(col_meta.get("k", 0))
        k_per_col[col] = k if k > 0 else (
            int(strategy.fixed_k) if strategy.fixed_k else 0
        )
        if strategy.is_raw:
            axes_per_col[col] = None
            # Raw mode produces 1 feature per embedding dim. The serialised k
            # may be 0 (no axes), so use the embedding dim from the mean.
            if k_per_col[col] == 0:
                k_per_col[col] = int(mean_per_col[col].shape[0])
        else:
            axes_path = rdir / f"{col}_axes.npy"
            if not axes_path.exists():
                raise FileNotFoundError(
                    f"Reduction axes for {col!r} missing at {axes_path}."
                )
            axes = np.load(axes_path).astype(np.float32)
            axes_per_col[col] = axes
            k_per_col[col] = int(axes.shape[0])
        cumvar_per_col[col] = float(col_meta.get("achieved_cumulative_variance", 0.0))
        ratios = col_meta.get("ratios") or []
        ratios_per_col[col] = np.asarray(ratios, dtype=np.float32)

    return ReductionArtifacts(
        strategy=strategy,
        axes_per_col=axes_per_col,
        mean_per_col=mean_per_col,
        k_per_col=k_per_col,
        achieved_cumvar_per_col=cumvar_per_col,
        ratios_per_col=ratios_per_col,
        legacy_pc1=False,
    )


def _load_legacy_pc1_layout(
    pc1_dir: Path,
    embed_cols: list[str],
) -> ReductionArtifacts:
    axes_per_col: dict[str, np.ndarray | None] = {}
    mean_per_col: dict[str, np.ndarray] = {}
    k_per_col: dict[str, int] = {}
    for col in embed_cols:
        axis = np.load(pc1_dir / f"{col}_pc1_axis.npy").astype(np.float32)
        mean = np.load(pc1_dir / f"{col}_pc1_mean.npy").astype(np.float32)
        # Promote (dim,) -> (1, dim) so downstream projection code is uniform.
        axes_per_col[col] = axis[None, :]
        mean_per_col[col] = mean
        k_per_col[col] = 1
    return ReductionArtifacts(
        strategy=parse_strategy("pc1"),
        axes_per_col=axes_per_col,
        mean_per_col=mean_per_col,
        k_per_col=k_per_col,
        achieved_cumvar_per_col={c: 0.0 for c in embed_cols},
        ratios_per_col={c: np.zeros(1, dtype=np.float32) for c in embed_cols},
        legacy_pc1=True,
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_with_artifacts(
    vectors: np.ndarray,
    col: str,
    artifacts: ReductionArtifacts,
) -> np.ndarray:
    """Project a (n, dim) batch of embeddings into the (n, k_col) feature space.

    Raw strategies short-circuit to a no-op copy (no matrix multiply).
    """
    if artifacts.strategy.is_raw or artifacts.axes_per_col.get(col) is None:
        return vectors.astype(np.float32, copy=False)
    axes = artifacts.axes_per_col[col]
    mean = artifacts.mean_per_col[col]
    return ((vectors - mean) @ axes.T).astype(np.float32)
