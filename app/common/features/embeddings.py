"""Stage 2: per-column streaming embeddings + PC1 scalar sidecar.

We talk to Snowflake's Arctic Embed v2.0 via its prebuilt ONNX weights
through onnxruntime. This bypasses `sentence-transformers`/`torch`
entirely - much faster on CPU (~2x via int8) and a much smaller
dependency footprint. See the `OnnxArcticEncoder` class below.
"""

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from bootstrap import ensure_app_on_path

ensure_app_on_path(__file__)

from common.config.columns import EMBED_COLS
from common.config.embedding import (
    EMBED_ROW_CHUNK,
    ENCODE_BATCH_SIZE_CPU,
    ENCODE_BATCH_SIZE_GPU,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    ONNX_FILE_CPU_INT8,
    ONNX_FILE_FALLBACK,
    ONNX_FILE_GPU_FP16,
    PASSAGE_PREFIX,
)
from common.features.pca import first_pc_axis


class OnnxArcticEncoder:
    """Thin wrapper around onnxruntime that exposes a `.encode(...)`
    method matching the slice of `SentenceTransformer.encode` that
    `_embed_one_column` actually calls.

    Why bypass `optimum.onnxruntime.ORTModelForFeatureExtraction`?
    Snowflake's ONNX export emits `sentence_embedding` (already pooled,
    shape [batch, 768]) - not the `last_hidden_state` that `optimum`'s
    feature-extraction wrapper expects. So we'd have to monkey-patch
    around it anyway. Going direct to `onnxruntime` is cleaner and
    sidesteps a bunch of `optimum`-specific failure modes.

    Pipeline per call:
      1. Tokenize each input via the model's HF fast tokenizer
         (Rust-backed; cheap even on 30k strings).
      2. Sort by token length so each batch pads tightly.
         The encoder runs over batches, then we scatter results back
         to the original input order.
      3. Run the ONNX session.
      4. (Optionally) L2-normalise. The ONNX `sentence_embedding`
         output is mean-pooled but NOT normalised, so we do that here.
    """

    def __init__(
        self,
        model_id: str = MODEL_NAME,
        max_seq_len: int = MAX_SEQ_LENGTH,
        prefer_gpu: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Pick provider + ONNX variant. We don't import torch just to
        # check for CUDA - onnxruntime tells us which providers are
        # actually loaded, which is what we really care about anyway.
        available = set(ort.get_available_providers())
        cuda_ok = prefer_gpu and "CUDAExecutionProvider" in available

        if cuda_ok:
            self.device = "cuda"
            self.providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",  # fallback for ops not on CUDA
            ]
            file_name = ONNX_FILE_GPU_FP16
        else:
            self.device = "cpu"
            self.providers = ["CPUExecutionProvider"]
            file_name = ONNX_FILE_CPU_INT8

        # Resolve the ONNX file via huggingface_hub (downloads +
        # caches). If the preferred variant isn't on the Hub for this
        # model, fall back to the fp32 model so we still produce
        # embeddings.
        try:
            onnx_path = hf_hub_download(model_id, file_name)
        except Exception as e:  # noqa: BLE001
            print(
                f"[onnx] failed to fetch {file_name!r} ({e}); "
                f"falling back to {ONNX_FILE_FALLBACK!r}."
            )
            onnx_path = hf_hub_download(model_id, ONNX_FILE_FALLBACK)
            file_name = ONNX_FILE_FALLBACK
        self.onnx_path = onnx_path
        self.onnx_file = file_name

        # Session knobs. `ORT_ENABLE_ALL` runs every onnxruntime graph
        # optimisation (constant folding, node fusion, ...). On CPU we
        # also pin to physical cores: SMT siblings fighting over the
        # same AVX/FMA ports almost always hurts transformer inference.
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.device == "cpu":
            try:
                physical = psutil.cpu_count(logical=False) or 1
            except Exception:
                physical = max(1, (os.cpu_count() or 2) // 2)
            sess_opts.intra_op_num_threads = physical
            sess_opts.inter_op_num_threads = 1
        self.sess_opts = sess_opts

        self.session = ort.InferenceSession(onnx_path, sess_opts, providers=self.providers)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]
        # We need a pooled `sentence_embedding` output from the ONNX
        # export. If a future export drops it, fall back to mean-pooling
        # `token_embeddings` ourselves.
        if "sentence_embedding" in self.output_names:
            self.pool_strategy = "onnx_pooled"
        elif "token_embeddings" in self.output_names:
            self.pool_strategy = "manual_mean"
        else:
            raise RuntimeError(
                f"ONNX model {file_name!r} has unexpected outputs {self.output_names!r}; "
                "expected 'sentence_embedding' or 'token_embeddings'."
            )

    # Threshold for tokenize_mode='auto'. Originally set to 10_000 on
    # the theory that the per-batch (all-Rust) path would beat the
    # single-pass path at high cardinality. End-to-end measurement on
    # the ODM-LATAM dataset disproved that:
    #   * RESTAURANT_BRAND_NAMES (33,868 uniques, ~1.2M tokens)
    #     single_pass: 847.1s   per_batch: 986.7s   (per_batch +16%)
    #   * 8k unique brand strings (earlier offline bench)
    #     single_pass: 167.7s   per_batch: 168.8s   (tied)
    # We never observed a cardinality at which per_batch actually wins,
    # so 'auto' now defaults to single_pass for every realistic input.
    # The per_batch path is still kept as an opt-in fallback via
    # `tokenize_mode='per_batch'` (or `tokenize_mode_overrides`) in
    # case a future dataset / model combination changes the trade-off.
    _AUTO_PER_BATCH_THRESHOLD = 10**18

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,  # kept for ST signature compat; we always return numpy
        normalize_embeddings: bool = True,
        tokenize_mode: str = "auto",
    ) -> np.ndarray:
        """Encode `sentences` to L2-normalised embedding vectors.

        Two tokenize/pad pipelines are available; both produce
        bit-identical outputs (verified) and only differ in throughput.

        Parameters
        ----------
        tokenize_mode : {"auto", "single_pass", "per_batch"}
            * "single_pass": tokenize ALL inputs in one batched Rust
              call, then pad each forward-pass batch in Python. This
              is the fast path on every input size we have measured on
              this dataset; ~10-20% faster overall than per_batch and
              ~16% faster on the worst-case column
              (RESTAURANT_BRAND_NAMES, 33,868 uniques).
            * "per_batch":   tokenize+truncate+pad inside one Rust call
              per batch (the original path). Kept as an opt-in fallback
              in case a future dataset/model combination flips the
              trade-off; never observed to win in current measurements.
            * "auto" (default): currently always picks "single_pass" on
              realistic input sizes. See `_AUTO_PER_BATCH_THRESHOLD`
              for the (very high) crossover that would trigger
              per_batch.
        """
        texts = list(sentences)
        n = len(texts)
        if n == 0:
            return np.empty((0, 768), dtype=np.float32)

        if tokenize_mode == "auto":
            tokenize_mode = (
                "per_batch" if n >= self._AUTO_PER_BATCH_THRESHOLD else "single_pass"
            )

        if tokenize_mode == "single_pass":
            return self._encode_single_pass(
                texts, batch_size, show_progress_bar, normalize_embeddings
            )
        elif tokenize_mode == "per_batch":
            return self._encode_per_batch(
                texts, batch_size, show_progress_bar, normalize_embeddings
            )
        else:
            raise ValueError(
                f"Unknown tokenize_mode {tokenize_mode!r}; expected one of "
                "'auto', 'single_pass', 'per_batch'."
            )

    def _encode_single_pass(
        self,
        texts: list,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Single-pass tokenize (one Rust call) + Python pad per batch.

        We tokenize every input up-front with `padding=False` so we
        get raw variable-length token lists, sort by length so each
        batch pads tightly, then build padded numpy arrays manually
        inside the per-batch loop. The key win vs `_encode_per_batch`
        is that we avoid 1 Python tokenize call PER input - which on
        small columns saves a measurable chunk of the per-batch
        overhead.
        """
        n = len(texts)
        want_token_type_ids = "token_type_ids" in self.input_names
        enc_all = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
            return_attention_mask=True,
            return_token_type_ids=want_token_type_ids,
        )
        ids_list = enc_all["input_ids"]
        masks_list = enc_all["attention_mask"]
        types_list = enc_all.get("token_type_ids") if want_token_type_ids else None

        lens = np.fromiter((len(x) for x in ids_list), dtype=np.int32, count=n)
        order = np.argsort(lens, kind="stable")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        out = None  # lazy-allocate; emb dim becomes known on the first run
        n_batches = (n + batch_size - 1) // batch_size
        bar = (
            tqdm(total=n_batches, desc="onnx batches", leave=False, unit="batch")
            if show_progress_bar
            else None
        )
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            b_ids = [ids_list[i] for i in idx]
            b_masks = [masks_list[i] for i in idx]
            bsz = len(b_ids)
            max_len = max(len(x) for x in b_ids)

            # Manual pad-right. We benchmarked this against
            # `tokenizer.pad()` (Rust) on 8000 RESTAURANT_BRAND_NAMES
            # unique strings:
            #   - this Python loop : 167.7s
            #   - tokenizer.pad()  : 226.5s   (+34% slower; HF's own
            #     warning even says pad() is slower than __call__ for
            #     fast tokenizers).
            # On VERY large + long-token columns this loop's
            # PyLong->int64 conversions add up though, which is why the
            # public `encode()` defaults to per_batch above 10k uniques.
            padded_ids = np.full((bsz, max_len), pad_id, dtype=np.int64)
            padded_masks = np.zeros((bsz, max_len), dtype=np.int64)
            for k, (ids, mask) in enumerate(zip(b_ids, b_masks)):
                L = len(ids)
                padded_ids[k, :L] = ids
                padded_masks[k, :L] = mask
            feeds = {"input_ids": padded_ids, "attention_mask": padded_masks}

            if types_list is not None:
                padded_types = np.zeros((bsz, max_len), dtype=np.int64)
                for k, t_ids in enumerate((types_list[i] for i in idx)):
                    padded_types[k, : len(t_ids)] = t_ids
                feeds["token_type_ids"] = padded_types

            emb = self._forward_and_pool(feeds, padded_masks)
            if normalize_embeddings:
                emb = self._l2_normalise(emb)

            if out is None:
                out = np.empty((n, emb.shape[-1]), dtype=np.float32)
            out[start : start + bsz] = emb.astype(np.float32, copy=False)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        return out[inv_order]

    def _encode_per_batch(
        self,
        texts: list,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Tokenize + truncate + pad per-batch, all inside one Rust call.

        This was the original path. Predictable, no Python pad
        overhead, wins at high cardinality + long tokens. The downside
        is paying one tokenize+pad invocation per batch instead of one
        global tokenize, which is a real-but-bounded constant per batch.

        Sort key here is the UNTRUNCATED token length (via
        `tokenizer.tokenize(t)`); above max_seq_len everything ends up
        the same length post-truncation anyway, so any ordering within
        the over-cap cluster is fine.
        """
        n = len(texts)
        lens = np.fromiter(
            (len(self.tokenizer.tokenize(t)) for t in texts),
            dtype=np.int32,
            count=n,
        )
        order = np.argsort(lens, kind="stable")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n)
        sorted_texts = [texts[i] for i in order]

        out = None
        n_batches = (n + batch_size - 1) // batch_size
        bar = (
            tqdm(total=n_batches, desc="onnx batches", leave=False, unit="batch")
            if show_progress_bar
            else None
        )
        for start in range(0, n, batch_size):
            chunk = sorted_texts[start : start + batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="np",
            )
            feeds = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in self.input_names and "token_type_ids" in enc:
                feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

            emb = self._forward_and_pool(feeds, feeds["attention_mask"])
            if normalize_embeddings:
                emb = self._l2_normalise(emb)

            if out is None:
                out = np.empty((n, emb.shape[-1]), dtype=np.float32)
            out[start : start + len(chunk)] = emb.astype(np.float32, copy=False)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        return out[inv_order]

    def _forward_and_pool(self, feeds: dict, attention_mask: np.ndarray) -> np.ndarray:
        """Run one ONNX forward pass and return per-input embeddings.

        Wraps the two output-shape variants we support
        (`sentence_embedding` for Snowflake's pre-pooled export,
        manual mean-pool for `token_embeddings`-only exports) so both
        encode strategies stay small.
        """
        if self.pool_strategy == "onnx_pooled":
            return self.session.run(["sentence_embedding"], feeds)[0]
        tok_emb = self.session.run(["token_embeddings"], feeds)[0]
        mask = attention_mask.astype(np.float32)[..., None]
        summed = (tok_emb * mask).sum(axis=1)
        count = np.clip(mask.sum(axis=1), 1.0, None)
        return summed / count

    @staticmethod
    def _l2_normalise(emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        return emb / np.clip(norms, 1e-12, None)


def _embed_one_column(
    dataset: pl.DataFrame,
    col: str,
    encoder: OnnxArcticEncoder,
    embed_dir: Path,
    row_chunk: int,
    encode_batch_size: int,
    tokenize_mode: str = "auto",
) -> None:
    """Stream-embed a single column and write two parquet sidecars:

      * {col}_EMBEDDING.parquet : FixedSizeList<float32, dim>, one row
        per input row. Full embedding, suitable for downstream ML.
      * {col}_EMB_PC1.parquet   : float32 scalar per row. Projection of
        each embedding onto the column's first principal component.
        This is what the correlation heatmaps use to represent the
        text column.
    """
    n_rows_total = dataset.height
    emb_path = embed_dir / f"{col}_EMBEDDING.parquet"
    pc1_path = embed_dir / f"{col}_EMB_PC1.parquet"

    # 1) Unique values via polars streaming engine + per-value counts.
    # Polars 1.25+ uses `engine="streaming"`; the old `streaming=True`
    # is deprecated.
    counts_df = (
        dataset.lazy()
        .select(pl.col(col).cast(pl.Utf8).fill_null("MISSING").alias("_t"))
        .group_by("_t")
        .agg(pl.len().alias("_n"))
        .sort("_t")
        .collect(engine="streaming")
    )
    unique_texts = counts_df["_t"].to_list()
    counts = counts_df["_n"].to_numpy()
    del counts_df

    # 2) Encode unique values once. Arctic embeds passages without any
    #    prefix; PASSAGE_PREFIX is "" by default but kept configurable
    #    so swapping back to a query-prefixed or E5-style model is a
    #    one-liner.
    to_encode = (
        [PASSAGE_PREFIX + t for t in unique_texts] if PASSAGE_PREFIX else unique_texts
    )
    unique_vectors = encoder.encode(
        to_encode,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        tokenize_mode=tokenize_mode,
    ).astype(np.float32)
    dim = unique_vectors.shape[1]
    del to_encode

    # 3) Compute the first principal component axis from the unique
    #    vectors, weighted by their row counts. This gives us a (dim,)
    #    vector that we can project any row onto in O(dim) time.
    pc1_axis, pc1_mean = first_pc_axis(unique_vectors, counts)
    del counts

    text_to_code = {t: i for i, t in enumerate(unique_texts)}
    del unique_texts

    # 4) Stream rows in chunks. For each chunk we write:
    #    - the full FixedSizeList<float32, dim> embedding row
    #    - the scalar PC1 projection
    emb_schema = pa.schema(
        [pa.field(f"{col}_EMBEDDING", pa.list_(pa.float32(), dim))]
    )
    pc1_schema = pa.schema([pa.field(f"{col}_EMB_PC1", pa.float32())])

    chunk_iter = tqdm(
        range(0, n_rows_total, row_chunk),
        desc=f"  {col} rows",
        leave=False,
        unit="chunk",
    )
    with pq.ParquetWriter(emb_path, emb_schema, compression="zstd") as emb_writer, \
         pq.ParquetWriter(pc1_path, pc1_schema, compression="zstd") as pc1_writer:
        for start in chunk_iter:
            length = min(row_chunk, n_rows_total - start)

            chunk_texts = (
                dataset.slice(start, length)
                .get_column(col)
                .cast(pl.Utf8)
                .fill_null("MISSING")
                .to_list()
            )
            codes = np.fromiter(
                (text_to_code[t] for t in chunk_texts),
                dtype=np.int32,
                count=length,
            )
            chunk_vectors = unique_vectors[codes]  # (length, dim)
            chunk_pc1 = ((chunk_vectors - pc1_mean) @ pc1_axis).astype(np.float32)

            # Full embedding -> FixedSizeList parquet append.
            flat = pa.array(chunk_vectors.reshape(-1), type=pa.float32())
            fsl = pa.FixedSizeListArray.from_arrays(flat, dim)
            emb_writer.write_table(
                pa.Table.from_arrays([fsl], names=[f"{col}_EMBEDDING"])
            )
            # PC1 scalar -> tiny parquet append.
            pc1_writer.write_table(
                pa.Table.from_arrays(
                    [pa.array(chunk_pc1, type=pa.float32())],
                    names=[f"{col}_EMB_PC1"],
                )
            )

            del chunk_texts, codes, chunk_vectors, chunk_pc1, flat, fsl

    del unique_vectors, text_to_code, pc1_axis, pc1_mean


def embed_text_columns(
    parquet_path: Path,
    embed_dir: Path,
    cols: list = None,
    model_id: str = MODEL_NAME,
    encode_batch_size: int = None,
    row_chunk: int = EMBED_ROW_CHUNK,
    max_seq_len: int = MAX_SEQ_LENGTH,
    prefer_gpu: bool = True,
    force: bool = False,
    tokenize_mode: str = "auto",
    tokenize_mode_overrides: dict = None,
    timer=None,
) -> None:
    """Compute streaming embeddings + PC1 scalars for each text column.

    Parameters
    ----------
    parquet_path      : source data (see Stage 1).
    embed_dir         : where the per-column sidecar parquets go.
    cols              : list of columns to embed. Defaults to
                        `common.config.columns.EMBED_COLS`. Filtered to
                        columns that actually exist in the parquet.
    model_id          : HF Hub model id. Default: Arctic Embed v2.0
                        medium.
    encode_batch_size : how many texts to forward through the model at
                        once. Defaults to ENCODE_BATCH_SIZE_GPU on CUDA,
                        else ENCODE_BATCH_SIZE_CPU.
    row_chunk         : row-streaming chunk size for the per-row sidecars.
    max_seq_len       : truncate inputs to this many tokens. Defaults to
                        MAX_SEQ_LENGTH (128). Lower = much faster on CPU,
                        as long as the truncation point is past your
                        data's 99th-percentile token count.
    prefer_gpu        : if True (default) and CUDAExecutionProvider is
                        available, use the fp16 ONNX weights on GPU. Set
                        to False to force CPU + int8.
    force             : if False (default) and BOTH sidecar files exist
                        for a column, skip that column. Set True to
                        re-embed.
    tokenize_mode     : default tokenize/pad strategy. One of
                        "auto", "single_pass", "per_batch". See
                        `OnnxArcticEncoder.encode` for what each does.
                        "auto" currently always picks "single_pass" on
                        realistic input sizes.
    tokenize_mode_overrides
                      : optional dict {col_name: mode} to override the
                        default for specific columns. Useful if you want
                        to pin RESTAURANT_BRAND_NAMES to "per_batch"
                        independently of the auto threshold, or force
                        every column into the same mode for benchmarking.
    """
    embed_dir.mkdir(parents=True, exist_ok=True)
    dataset = pl.read_parquet(parquet_path)
    cols = cols if cols is not None else EMBED_COLS
    cols = [c for c in cols if c in dataset.columns]
    if not cols:
        print("[stage2] no columns to embed (none of EMBED_COLS exist in parquet).")
        return

    overrides = tokenize_mode_overrides or {}

    print(f"[stage2] loading ONNX encoder for {model_id!r} ...")
    encoder = OnnxArcticEncoder(
        model_id=model_id,
        max_seq_len=max_seq_len,
        prefer_gpu=prefer_gpu,
    )
    print(
        f"[stage2] device={encoder.device!r} onnx_file={encoder.onnx_file!r} "
        f"max_seq_len={encoder.max_seq_len}"
    )

    if encode_batch_size is None:
        encode_batch_size = (
            ENCODE_BATCH_SIZE_GPU if encoder.device == "cuda" else ENCODE_BATCH_SIZE_CPU
        )

    n_expected = dataset.height

    def _sidecar_row_count(path: Path) -> int:
        return pq.ParquetFile(path).metadata.num_rows

    col_bar = tqdm(cols, desc="Embedding columns", unit="col")
    for col in col_bar:
        col_bar.set_postfix_str(col)
        emb_path = embed_dir / f"{col}_EMBEDDING.parquet"
        pc1_path = embed_dir / f"{col}_EMB_PC1.parquet"
        if not force and emb_path.exists() and pc1_path.exists():
            emb_rows = _sidecar_row_count(emb_path)
            if emb_rows == n_expected:
                tqdm.write(f"[stage2] {col}: sidecars already exist, skipping.")
                continue
            tqdm.write(
                f"[stage2] {col}: sidecars exist but rows {emb_rows:,} != "
                f"parquet {n_expected:,}; re-embedding."
            )
        col_mode = overrides.get(col, tokenize_mode)
        step_ctx = (
            timer.step(
                "2.embed.column",
                f"Embed column {col}",
                f"features.embeddings._embed_one_column({col!r})",
            )
            if timer is not None
            else nullcontext()
        )
        with step_ctx:
            t0 = time.perf_counter()
            _embed_one_column(
                dataset=dataset,
                col=col,
                encoder=encoder,
                embed_dir=embed_dir,
                row_chunk=row_chunk,
                encode_batch_size=encode_batch_size,
                tokenize_mode=col_mode,
            )
            tqdm.write(
                f"[stage2] {col}: done in {time.perf_counter() - t0:.1f}s "
                f"(tokenize_mode={col_mode!r})"
            )
            if timer is not None:
                timer.register_planned("2.embed.column", f"Embed column {col}", "features.embeddings._embed_one_column")

    print(
        f"[stage2] {len(cols)} columns processed -> {embed_dir}/ "
        f"(row order matches {parquet_path})."
    )


# ============================================================================
# How to speed up the RESTAURANT_NAME pass (high-cardinality column)
# ============================================================================
# In priority order. Already done = (DONE). Not yet wired = (TODO).
#
# 1. (DONE) Deduplicate before encoding. Already a 10-100x win for most cols.
# 2. (DONE) ONNX runtime + int8 (CPU) / fp16 (GPU) prequantized weights.
#    Provided by Snowflake on the HF Hub; Arctic v2.0's int8 export gives
#    ~2x throughput on CPU vs torch fp32 with cosine similarity ~0.95-0.98
#    to the fp32 outputs. Plumbed through `OnnxArcticEncoder`.
# 3. (DONE) Auto-detect CUDAExecutionProvider in `OnnxArcticEncoder` and
#    promote to fp16 weights when present. A CUDA GPU brings
#    RESTAURANT_NAME from minutes down to seconds; install
#    `onnxruntime-gpu` (replacing `onnxruntime`) to enable.
# 4. (TODO) Canonicalise text BEFORE deduping. Many restaurant names differ
#    only in casing/whitespace ("McDonald's" vs "MCDONALDS" vs "Mc Donalds").
#    A simple `" ".join(s.lower().split())` typically collapses uniques by
#    30-60% with no semantic loss. Drop in to `_embed_one_column` after the
#    `unique_texts = counts_df["_t"].to_list()` line and recompute counts.
# 5. (TODO) Persistent embedding cache. Hash each input string and look it
#    up in a SQLite/parquet keyed cache so the next run only embeds the
#    new strings. Big win if you re-run on slowly-changing data.
# 6. (TODO) Multi-process / multi-session ONNX encoding. Run several
#    `OnnxArcticEncoder` sessions in parallel (one per physical NUMA node
#    on big CPUs, or one per GPU). For a 1.4M-unique RESTAURANT_NAME pass
#    this is a near-linear speedup on CPU.
# 7. (TODO) Drop the long tail. For names with count==1 (which contribute
#    almost no statistical signal), substitute a fixed zero vector. Skip
#    encoding them entirely. Typically removes 30-70% of uniques on a
#    real-world restaurant-name distribution.
# 8. (TODO) Use a smaller model just for RESTAURANT_NAME. E.g.
#    "paraphrase-multilingual-MiniLM-L12-v2" is similar speed/quality.
# 9. (TODO) Pre-split RESTAURANT_NAME into franchise / suffix tokens and
#    embed only the franchise-token. Most predictive power comes from the
#    chain identifier ("Starbucks Av. 9 de Julio" -> "Starbucks").
