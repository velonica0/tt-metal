# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model TP path vs the HuggingFace reference (real checkpoint, any mesh).

The other TP model tests prove the served path matches the bespoke path *per-step*; this one
anchors the bespoke path (``prefill_tp`` + ``decode_tp``) to the HF ``Qwen3_5ForCausalLM`` run on
CPU with the same weights: teacher-forced per-step logits PCC and top-1 agreement over a short
factual prompt and a ~128-token passage, plus a greedy continuation sanity check.

Meant for the small checkpoints (0.8B / 2B / 4B / 9B run on CPU in fp32 in minutes) so a new
device or mesh can be validated end-to-end before pulling a 27B download. Run:

    HF_MODEL=Qwen/Qwen3.5-0.8B MESH_DEVICE=T3K \\
      pytest models/demos/blackhole/qwen36/tests/test_model_tp_hf_reference.py -v -s

Measured on a T3K 1x8 (2026-09-21):
  * Qwen3.5-0.8B: per-step PCC 0.972-0.995 with the default bfp4 gate/up weights (top-1 6/7 on both
    prompts); 0.990-0.9994 with QWEN36_MLP_GATEUP_DTYPE=bfp8 (7/7 short, 6/7 passage) -- the residual
    gap is MLP weight quantization, which a small model feels much more than the 27B.
  * Qwen3.8-27B: per-step PCC 0.995-0.998 (bfp4 gate/up), short 7/7, passage 5/7 top-1 / 7/7 in top-3.
The PCC threshold lives in pcc_thresholds.json.
"""
import math
import os

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PASSAGE = (
    "The Tenstorrent Wormhole processor is built around a grid of Tensix cores connected by a "
    "network on chip. Each core has its own local memory and a set of matrix and vector engines, "
    "and several chips can be linked together over Ethernet so that a single model is split "
    "across them. In this arrangement every device holds a slice of each weight matrix, the "
    "activations are exchanged with collective operations, and the result is"
)


def _hf_reference(ckpt_dir):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig.from_pretrained(ckpt_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(ckpt_dir, config=cfg, dtype=torch.float32)
    model.eval()
    return model


@torch.no_grad()
def _hf_step_logits(hf, ids, n_dec):
    """Prefill logits at the last prompt position, then n_dec teacher-forced greedy decode steps.
    Returns (per-step logits [n_dec+1, vocab], tokens fed at each decode step)."""
    ids_t = torch.tensor([ids], dtype=torch.long)
    out = hf(input_ids=ids_t, use_cache=True)
    logits = [out.logits[0, -1].float()]
    past = out.past_key_values
    fed = []
    for _ in range(n_dec):
        nxt = int(torch.argmax(logits[-1]))
        fed.append(nxt)
        out = hf(input_ids=torch.tensor([[nxt]], dtype=torch.long), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits.append(out.logits[0, -1].float())
    return torch.stack(logits), fed


@torch.no_grad()
def _tt_step_logits(model, ids, fed):
    """Same schedule on the TT bespoke path: prefill (padded to the 128-token GDN chunk), then
    decode the SAME tokens HF fed (teacher forcing), so every step sees identical inputs."""
    model.reset_tp()
    T = len(ids)
    T_pad = max(128, math.ceil(T / 128) * 128)
    padded = ids + [0] * (T_pad - T)
    logits = [model.prefill_tp(torch.tensor([padded], dtype=torch.long), valid_len=T).float().reshape(-1)]
    for i, tok in enumerate(fed):
        logits.append(model.decode_tp(tok, T + i).float().reshape(-1))
    return torch.stack(logits)


@torch.no_grad()
@parametrize_mesh_tp()
def test_model_tp_matches_hf(mesh_device, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=512)
    ckpt = model.args.CKPT_DIR
    vocab = model.args.vocab_size

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    hf = _hf_reference(ckpt)

    n_dec = 6
    thr = get_pcc_threshold(request, default=0.97)
    worst = 1.0
    short_ref_top3 = None
    for name, prompt in (("short", "The capital of France is"), ("passage", PASSAGE)):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        ref, fed = _hf_step_logits(hf, ids, n_dec)
        tt = _tt_step_logits(model, ids, fed)
        assert tt.shape[-1] >= vocab
        tt = tt[:, :vocab]
        pccs = [float(comp_pcc(ref[i], tt[i], 0.99)[1]) for i in range(n_dec + 1)]
        top1 = [int(torch.argmax(ref[i]) == torch.argmax(tt[i])) for i in range(n_dec + 1)]
        # Open text has near-ties (27B, teacher-forced after "a single": HF " logical accelerator",
        # TT ", model" at PCC 0.996), so exact top-1 is logged but the bar is "TT's greedy token is
        # in HF's top-3" at every step.
        in_top3 = [int(torch.argmax(tt[i]) in torch.topk(ref[i], 3).indices) for i in range(n_dec + 1)]
        hf_text = tok.decode(fed + [int(torch.argmax(ref[-1]))])
        tt_text = tok.decode([int(torch.argmax(tt[i])) for i in range(n_dec + 1)])
        logger.info(f"[{name}] T={len(ids)} per-step logits PCC vs HF: {[f'{p:.4f}' for p in pccs]}")
        logger.info(
            f"[{name}] top-1 agreement {sum(top1)}/{len(top1)} (in HF top-3: {sum(in_top3)}/{len(in_top3)}); "
            f"HF: {hf_text!r} | TT: {tt_text!r}"
        )
        worst = min(worst, min(pccs))
        if name == "short":
            short_ref_top3 = torch.topk(ref[0], 3).indices.tolist()
        assert min(pccs) >= thr, f"[{name}] per-step logits PCC vs HF below {thr}: {pccs}"
        assert all(in_top3), f"[{name}] TT greedy token outside HF top-3 at steps {in_top3}"

    # Free-running greedy continuation must be coherent (stateful KV + GDN carry over decode).
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    new_ids = model.generate_tp(ids, max_new_tokens=8)
    text = tok.decode(new_ids)
    logger.info(f"GENERATED: {tok.decode(ids)!r} -> {text!r}")
    assert len(set(new_ids)) > 1, f"degenerate continuation: {new_ids}"
    # bfp4 gate/up can flip a near-tie at the first token (' Paris' vs ' the' on the 0.8B); require
    # the greedy first token to be within HF's top-3, and exact 'Paris' only at bfp8.
    assert new_ids[0] in short_ref_top3, f"first token {tok.decode([new_ids[0]])!r} not in HF top-3"
    if os.environ.get("QWEN36_MLP_GATEUP_DTYPE") == "bfp8":
        assert "Paris" in text, f"expected 'Paris' in continuation, got {text!r}"
    logger.info(f"PASSED: TP model matches HF reference (worst per-step PCC {worst:.4f})")
