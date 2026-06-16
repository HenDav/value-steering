# SPDX-License-Identifier: Apache-2.0
"""
GPU smoke test for the vLLM-backed value-head training pipeline (NOT full parity).

Validates the throughput-efficient path: ONE vLLM pooling pass (scripts/vllm_extract.extract_all)
-> cached per-token features -> train_probe over FeatureDataset (feature_fn=identity, no
per-epoch re-forward) -> checkpoint+sidecar -> load_value_head round-trip -> head SEPARATES the
two label classes -> conformal calibration returns a threshold in (0, 1).

Needs CUDA + a small $VALUE_STEER_TEST_MODEL (e.g. facebook/opt-125m). Run:
    VALUE_STEER_TEST_MODEL=facebook/opt-125m pytest tests/test_gpu_smoke.py -q -m gpu
"""

import os
import sys
import tempfile

import pytest

pytestmark = pytest.mark.gpu

_MODEL = os.environ.get("VALUE_STEER_TEST_MODEL")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def _have_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


requires_gpu_model = pytest.mark.skipif(
    not (_have_gpu() and _MODEL),
    reason="needs CUDA + $VALUE_STEER_TEST_MODEL (small model)",
)


def _synthetic(tok, n_per_class=12):
    """Two separable classes: undesirable (label 1) responses contain 'XBADX', good (0) don't.
    Returns (token_ids, prompt_len, label) via plain tokenization (chat-template path is covered
    elsewhere; here we exercise extraction + cached training)."""
    items = []
    for i in range(n_per_class):
        for text, score in ((f"a calm helpful reply {i}", 0.0),
                            (f"XBADX harmful content {i}", 1.0)):
            ids = tok("question: " + text, add_special_tokens=True)["input_ids"]
            items.append((ids, 2, score))
    return items


@requires_gpu_model
def test_vllm_extract_cached_train_smoke():
    import torch
    import vllm_extract
    from calibrate_threshold import calibrate_from_dataset
    from torch.utils.data import DataLoader

    from value_steer.train_probe import (
        FeatureCacheDataset,
        FeatureCollator,
        save_probe_checkpoint,
        train_probe,
    )
    from value_steer.value_probe import DEFAULT_SPEC, ValueHead, load_value_head

    device = "cuda"
    llm = vllm_extract.build_pooling_llm(_MODEL, gpu_memory_utilization=0.3, max_model_len=512)
    tok = llm.get_tokenizer()
    hidden = vllm_extract.hidden_size(llm)

    items = _synthetic(tok)
    tr, va = items[:-8], items[-8:]
    with tempfile.TemporaryDirectory() as d:
        # extract -> disk cache (the real path), then train over the memmapped cache
        vllm_extract.write_feature_cache(llm, os.path.join(d, "train"), [it[0] for it in tr],
                                         [it[1] for it in tr], [it[2] for it in tr])
        vllm_extract.write_feature_cache(llm, os.path.join(d, "val"), [it[0] for it in va],
                                         [it[1] for it in va], [it[2] for it in va])
        train_ds = FeatureCacheDataset(os.path.join(d, "train"))
        val_ds = FeatureCacheDataset(os.path.join(d, "val"))
        assert train_ds.H == hidden and len(train_ds) == len(tr)
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=FeatureCollator())

        head = ValueHead(hidden)
        train_probe(None, head, loader, epochs=10, lr=1e-3, loss_name="focal", use_td=True,
                    coh_weight=0.1, device=device, feature_fn=lambda f, a: f)

        # 1. checkpoint + sidecar round-trip via the runner's loader
        path = os.path.join(d, "value_head.bin")
        w, sidecar = save_probe_checkpoint(path, head, feature_spec=DEFAULT_SPEC,
                                           threshold=0.5, meta={"domain": "smoke"})
        assert os.path.exists(w) and os.path.exists(sidecar)
        reloaded = load_value_head(path, hidden, device)

        # 2. trained head SEPARATES the classes on held-out rows (mean P over response tokens)
        @torch.no_grad()
        def mean_p(want):
            ps = []
            for i in range(len(val_ds)):
                it = val_ds[i]
                if it["label"] != want:
                    continue
                ps.append(float(reloaded.p(it["features"].to(device))[it["prompt_len"]:].mean()))
            return sum(ps) / len(ps)

        assert mean_p(1.0) > mean_p(0.0), "head did not separate undesirable from good"

        # 3. calibration returns a usable threshold in (0, 1)
        thr = calibrate_from_dataset(val_ds, reloaded, tau=0.2, device=device)
        assert 0.0 < thr < 1.0
