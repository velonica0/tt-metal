# Qwen3.5 / Qwen3.6 / Qwen3.8 on Blackhole (and Wormhole T3K)

This directory implements Tenstorrent inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6/3.8 family. The same code path serves these checkpoints:

| Model            | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.8-27B      | `Qwen/Qwen3.8-27B`     | P150x4 / P150x8      | 4- / 8-way tensor parallel |
| Qwen3.6/3.8-27B  | `Qwen/Qwen3.8-27B`     | Wormhole T3K — `T3K` | 8-way tensor parallel  |

Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`) has the same config and
tensor layout as Qwen3.6-27B, so it needs no model-side changes — only `HF_MODEL`.

- The **9B** runs on a **single Blackhole P150** device. It uses the validated
  single-device forward path (no collectives).
- The **27B** variants (both Qwen3.5-27B and Qwen3.6-27B) run on a **P150x4**
  (a `(1, 4)` Blackhole mesh) using **4-way tensor parallelism (TP)**. The TP
  path needs `FABRIC_1D` for the cross-device collectives (all-reduce /
  reduce-scatter) and a trace region for the captured chunk-outer prefill trace.
- **Qwen3.6-27B additionally runs at TP=8** on a `(1, 8)` mesh (`P150x8`).
  Because it has only **4 KV heads**, TP=8 cannot give each device its own head:
  each head is instead **replicated across the device pair holding its GQA query
  group** (devices 0-1 share KV head 0, 2-3 head 1, and so on), so
  `n_local_kv_heads` is 1 at both TP=4 and TP=8 and the whole runtime KV path is
  unchanged. See `tp_common.replicate_kv_weight` and
  `ModelArgs.SUPPORTS_KV_REPLICATION`.

Everything model-specific (hybrid layer dispatch, DeltaNet head/conv dims,
partial rotary factor, vocab, layer count) is read from the parsed HF config, so
the single code base adapts to each checkpoint. The device count alone
(`num_devices > 1`) switches between the single-device and TP code paths — see
`tt/model_config.py` and `tt/tp_common.py`.

## Architecture

Assembly: `tok_embeddings → N × Qwen36DecoderLayer → RMSNorm → LM Head`.

Each model interleaves two attention block types (read from the HF
`layer_types`): **Gated DeltaNet** (linear-attention, recurrent + causal conv
state) layers and **Gated Full Attention** (paged KV cache) layers. The 9B has
32 layers (24 DeltaNet + 8 full-attention). Qwen3.5 uses zero-centered RMSNorm
everywhere and **partial** RoPE (only a fraction of each head is rotated).

## Environment setup

Before running **any** test, export the two environment variables that select
the checkpoint and the device mesh.

**9B (single P150):**

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=P150
```

**27B (P150x4):**

```bash
# Qwen3.6-27B
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=P150x4

# …or Qwen3.5-27B
export HF_MODEL=Qwen/Qwen3.5-27B
export MESH_DEVICE=P150x4
```

**27B on a Wormhole T3K (8 chips, TP=8):**

```bash
export HF_MODEL=Qwen/Qwen3.8-27B   # or Qwen/Qwen3.6-27B
export MESH_DEVICE=T3K
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape (`P150` → `(1,1)`, `P150x4` → `(1,4)`,
`P150x8` / `T3K` → `(1,8)`, `N300` → `(1,2)`; see `tests/test_factory.py::MESH_SHAPES`).

### Wormhole notes

The model code reads the worker grid, DRAM bank count and ethernet link count from the device,
so the T3K needs no separate configs (`tt/tp_common.py` documents the arch differences). Two
things differ from Blackhole:

* T3K has 12 GB DRAM per chip, so the 27B needs the full 8-chip mesh (TP=8, KV heads replicated
  across device pairs exactly as on P150x8). `N300` (`(1,2)`) is mapped for completeness but the
  27B does not fit on it.
* The tuned prefill matmul outputs (attention `wo`, MLP `down`) are DRAM- rather than L1-resident
  on Wormhole (`tp_common.prefill_matmul_out_memcfg`): the 64-core grid leaves no L1 next to the
  matmul's circular buffers at S=2048. This is a correctness-first choice; nothing has been
  perf-tuned for Wormhole yet.

Measured on a T3K (2026-09-21, nothing perf-tuned for Wormhole yet):

| Checkpoint | `test_model_tp.py` | `text_demo.py` | vs HF fp32 (`test_model_tp_hf_reference.py`) |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 11/11 | 55 tok/s B=1, 359 tok/s B=8, TTFT 11.7 s @ 103k | per-step PCC 0.97-0.995 (bfp4 gate/up), 0.99-0.9994 (bfp8) |
| Qwen3.8-27B | 10/11 (see below) | 17 tok/s B=1, 106 tok/s B=8, TTFT 1.4 s @ 2.6k | per-step PCC 0.995-0.998; `test_generate_tp`: "The capital of France is **Paris**." |

Known issue (Wormhole T3K only): an **intermittent device hang** in the eager (non-traced) prefill
paths. Root-caused far enough to hand to tt-metal (2026-09-22):

* Mechanism: after a 2D-multicast matmul (`bmm_large_block_zm_fused_bias_activation`, the kernel behind
  the tuned prefill program configs and behind `ttnn.linear` auto configs at prefill sizes), a few Tensix
  cores are left with the unpack/math handshake desynchronised — the watcher shows the *next* compute
  kernel on those cores parked at `UABD`/`MWDD` (unpack "done", math still waiting for source data).
  Any following binary eltwise op (e.g. the residual `add`) deadlocks on those cores; unary ops still run.
  With dispatch to the remote chips going over the fabric, the stalled program fills the fetch queue and
  the host spins in `fetch_queue_reserve_back`. Recover with `tt-smi -r`.
* Isolated reproducer (`~/like/matmul_probe.py` on the box, no model): M=2048, K=768, N=5120, 8x8 grid,
  per-core 8x20, `in0_block_w=4`, subblock 1x4, `out_block_h=4`, fp32 dest acc — followed by
  `ttnn.add`. Hangs 5/5 on the 1x8 mesh with `packer_l1_acc=True`, 0/5 with it off; **never** on a
  single chip (all 8 tested individually), 1x1+fabric, 1x2 or 1x4 — only on the full 8-chip mesh
  (1x8 and 2x4), i.e. a launch-timing race. Changing any one of `in0_block_w`, `out_block_h` or
  `out_subblock_w` also changes the hit rate (4/5 → 1/5 → 0/10 over 5-10 cold runs each).
* Model-level mitigation applied: `tp_common.COMPUTE_HIFI2` drops `packer_l1_acc` on Wormhole (the MLP
  already ran without it). This removes the isolated reproducer but **does not eliminate the hang in the
  full model** — it moves between call sites/tests as configs or the test order change, and it was seen
  once in a fresh session. The traced, warmed-up served path (`text_demo.py`, vLLM) has never hung in
  any run; `test_model_tp.py` hangs in one of its eager cases most sessions.
* Not fixable in the model: needs an LLK/kernel-exit fix in tt-metal. Adjacent upstream reports:
  tenstorrent/tt-metal#53823 (WH eltwise-binary unpack-sync race), #55664 (FP32 dest-acc TRISC
  handshake ordering).

Component tests can run before the checkpoint is downloaded: point `HF_MODEL` at a directory
holding only `config.json` + tokenizer files (or set `QWEN36_SYNTHETIC_WEIGHTS=1`) and the TP
tests synthesize random weights of the right shapes for both the device path and the torch
reference (`tests/test_factory.py::synthetic_weights_enabled`). The full-model tests need the
real checkpoint.

Optional flags:

```bash
# Run SDPA in BF8 (faster; slightly lower precision).
export QWEN_SDPA_BF8=1
```


## End-to-end demo test (`demo/text_demo.py`)

The e2e text-generation test lives in `demo/text_demo.py`. It is a single
parametrized test (`test_demo_text`) covering a range of input sequence lengths
(ISLs): 128, 4k, 8k, 16k, 32k, 64k, 128k, and 256k tokens. Each ISL runs prefill
+ decode and validates output (non-degenerate generation) and per-ISL
performance gates (TTFT and decode tok/s).

Two execution variants exist per ISL, identified by the test id prefix:

- **`traced_*`** — captures the prefill (chunk-outer) and decode forward passes
  as device traces and replays them. This is the **preferred** path and the one
  vLLM serves; run these by default.
- **`paged_*`** — non-traced paged path, useful as an eager reference/fallback.

Run the preferred traced cases (the env vars above must already be exported):

```bash
# All traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"

# A single ISL, e.g. the short 128-token traced case
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"

# Medium / long traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_64k"
```

The **same command works for both 9B and 27B** — only the exported `HF_MODEL` /
`MESH_DEVICE` differ. On a single device the test takes the validated 9B path; on
the `(1,4)` mesh it routes through the TP chunk-outer traced prefill + paged
traced decode path automatically.

> Long-context cases (64k+) download a public-domain corpus (Frankenstein, War
> and Peace) on first run and cache it under `demo/sample_prompts/.context_cache`.

## Tests

There are two tiers of tests under `tests/`.

### Single-device unit / component tests — **9B**

These run on a single P150 against the 9B checkpoint (the conftests
`setdefault HF_MODEL=Qwen/Qwen3.5-9B`). They validate each component's forward
pass against a torch reference (PCC) — see `tests/pcc_thresholds.json` for the
per-test thresholds.

`tests/unit/` (component PCC vs torch):

| Test                       | Validates                                            |
| -------------------------- | ---------------------------------------------------- |
| `test_embedding.py`        | token embedding                                      |
| `test_rms_norm.py`         | zero-centered RMSNorm (the "+1" fold)                |
| `test_rope.py`             | partial-rotary RoPE (host freqs + on-device lookup)  |
| `test_mlp.py`              | single-device SwiGLU MLP (layer 0)                   |
| `test_attention.py`        | single-device gated full attention (layer 3)         |
| `test_gdn.py`              | single-device Gated DeltaNet (layer 0)               |
| `test_lm_head.py`          | LM head logits (bf8 vs bf16)                          |
| `test_layer.py`            | full decoder-block sanity (no NaN/Inf, non-constant) |
| `test_model.py`            | Generator decode contract: traced vs paged decode    |
| `test_substate.py`         | weight `substate` helper (pure CPU, no device)       |

`tests/` (single-device, also 9B):

| Test                     | Validates                                                  |
| ------------------------ | ---------------------------------------------------------- |
| `test_prefill.py`        | masked fixed-bucket + chunk-outer prefill vs `prefill_paged` |
| `test_weight_mapping.py` | HF → internal weight key remapping (pure CPU)              |

Run the 9B unit suite (with `HF_MODEL=Qwen/Qwen3.5-9B`, `MESH_DEVICE=P150`):

```bash
pytest models/demos/blackhole/qwen36/tests/unit/ -v -s
pytest models/demos/blackhole/qwen36/tests/test_prefill.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_weight_mapping.py -v -s
```

> `test_prefill.py` auto-skips cases longer than `--max-prefill` (default 8192).
> Raise it to exercise long-context prefill, e.g. `--max-prefill 131072`.

### Tensor-parallel tests — **27B (P150x4)**

The `*_tp` tests exercise the multi-device TP path and default to the 27B
checkpoint. They must run on the `(1,4)` mesh with `FABRIC_1D` (the
`parametrize_mesh_tp` helper wires this from `MESH_DEVICE`). PCC thresholds are in
`tests/pcc_thresholds.json`.

| Test                  | Validates                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `test_mlp_tp.py`      | TP SwiGLU MLP (column/row-parallel + reduce-scatter)                |
| `test_attention_tp.py`| TP gated full attention: decode / prefill / paged-KV contract       |
| `test_gdn_tp.py`      | TP Gated DeltaNet: decode + chunk-prefill                           |
| `test_model_tp.py`    | full-model TP contract: paged+traced path matches the bespoke oracle |
| `test_generate_tp.py` | full-model bespoke `generate_tp` on a real prompt (answer oracle)   |

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.6-27B` or `Qwen/Qwen3.5-27B`,
`MESH_DEVICE=P150x4`):

```bash
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
```

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.
