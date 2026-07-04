# SPDX-License-Identifier: Apache-2.0
"""
CPU unit tests for train_probe.py: losses, collator, a real train step on a stub
backbone, and the checkpoint roundtrip (incl. that the runner's loader reads it).

Run:  pytest test_train_probe.py -q
"""

import json
import math

import torch
import torch.nn as nn

from value_steer.train_probe import (
    ProbeCollator,
    ProbeDataset,
    focal_loss_with_logits,
    load_probe_meta,
    probe_loss,
    save_probe_checkpoint,
    td_coherence_loss,
    train_probe,
)
from value_steer.value_probe import FeatureSpec, ValueHead, load_value_head


# --------------------------------------------------------------------------- #
# focal loss                                                                  #
# --------------------------------------------------------------------------- #
def test_focal_known_value():
    # logit=0 (p=0.5), target=1, alpha_pos=0.7, gamma=1:
    # bce=-log(0.5)=0.6931, pt=0.5, (1-pt)^1=0.5, alpha_t=0.7 -> 0.7*0.5*0.6931
    out = focal_loss_with_logits(torch.zeros(1), torch.ones(1), alpha_pos=0.7, gamma=1.0)
    assert out.item() == pytest_approx(0.7 * 0.5 * math.log(2))


def test_focal_gamma_zero_is_weighted_bce():
    logit, target = torch.tensor([0.0]), torch.tensor([1.0])
    out = focal_loss_with_logits(logit, target, alpha_pos=0.7, gamma=0.0)
    assert out.item() == pytest_approx(0.7 * math.log(2))   # alpha_t * bce


def test_focal_downweights_easy_examples():
    # an easy positive (high logit, target 1) should incur less focal loss than a hard one
    easy = focal_loss_with_logits(torch.tensor([5.0]), torch.tensor([1.0]))
    hard = focal_loss_with_logits(torch.tensor([0.0]), torch.tensor([1.0]))
    assert easy.item() < hard.item()


# --------------------------------------------------------------------------- #
# TD coherence                                                                #
# --------------------------------------------------------------------------- #
def test_td_zero_for_constant_trajectory():
    z = torch.tensor([[1.0, 1.0, 1.0]])
    assert td_coherence_loss(z, torch.ones_like(z)).item() == pytest_approx(0.0)


def test_td_known_value():
    z = torch.tensor([[0.0, 2.0]])                       # one transition, diff 2 -> 4
    assert td_coherence_loss(z, torch.ones_like(z)).item() == pytest_approx(4.0)


def test_td_ignores_pad_transitions():
    z = torch.tensor([[0.0, 2.0, 9.0]])                  # third token is pad
    attn = torch.tensor([[1, 1, 0]])
    # only the (0->2) transition is valid: 4.0; the (2->9) jump is masked out
    assert td_coherence_loss(z, attn).item() == pytest_approx(4.0)


# --------------------------------------------------------------------------- #
# probe_loss                                                                  #
# --------------------------------------------------------------------------- #
def test_probe_loss_ignores_pad_logits():
    attn = torch.tensor([[1, 1, 0]])
    labels = torch.tensor([1.0])
    z1 = torch.tensor([[0.5, -0.5, 3.0]])
    z2 = torch.tensor([[0.5, -0.5, -7.0]])               # differs only at the pad slot
    assert probe_loss(z1, attn, labels).item() == pytest_approx(
        probe_loss(z2, attn, labels).item()
    )


def test_probe_loss_td_adds_nonneg_term():
    z = torch.tensor([[0.0, 3.0]])
    attn = torch.ones_like(z)
    labels = torch.tensor([1.0])
    with_td = probe_loss(z, attn, labels, use_td=True, coh_weight=0.1)
    without = probe_loss(z, attn, labels, use_td=False)
    assert with_td.item() > without.item()               # +0.1 * 9


# --------------------------------------------------------------------------- #
# collator                                                                    #
# --------------------------------------------------------------------------- #
class _FakeTok:
    """apply_chat_template returns deterministic token ids: user content as len(prompt)
    1-tokens after a 0 BOS, assistant content as len(response) 2-tokens."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [0]                                        # BOS
        ids += [1] * len(messages[0]["content"])         # user/prompt tokens
        if add_generation_prompt:
            ids += [1]                                   # assistant header
        if len(messages) > 1:
            ids += [2] * len(messages[1]["content"])     # response tokens
        return ids


def test_collator_pads_and_masks():
    ds = ProbeDataset(_FakeTok(), prompts=["ab", "abcd"], responses=["x", "yy"], labels=[0, 1])
    batch = [ds[0], ds[1]]
    ids, attn, plens, labels = ProbeCollator(pad_token_id=-1)(batch)
    assert ids.shape[0] == 2 and ids.shape[1] == max(len(ds[0]["input_ids"]), len(ds[1]["input_ids"]))
    # row 0 is shorter -> padded with -1 -> attention 0 in the tail
    assert attn[0].sum().item() == len(ds[0]["input_ids"])
    assert labels.tolist() == [0.0, 1.0]
    assert plens.tolist() == [ds[0]["prompt_len"], ds[1]["prompt_len"]]


# --------------------------------------------------------------------------- #
# end-to-end train step on a stub backbone                                    #
# --------------------------------------------------------------------------- #
class _StubBackbone(nn.Module):
    """Frozen 'backbone': token id -> fixed distinct hidden vector, so a sequence of
    token-1s and a sequence of token-2s are linearly separable and the head can learn."""
    def __init__(self, hidden):
        super().__init__()
        emb = torch.zeros(3, hidden)
        emb[1, 0] = 1.0           # token 1 -> +e0
        emb[2, 1] = 1.0           # token 2 -> +e1
        self.register_buffer("emb", emb)

    def forward(self, input_ids, attention_mask=None):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.emb[input_ids])


def _loader(input_ids, attn, labels):
    plen = torch.zeros(len(labels), dtype=torch.long)
    return [(input_ids, attn, plen, labels)]


def test_train_probe_reduces_loss():
    torch.manual_seed(0)
    H = 8
    backbone = _StubBackbone(H)
    head = ValueHead(H)
    # two examples: token-1 seq labeled 0, token-2 seq labeled 1
    ids = torch.tensor([[1, 1, 1], [2, 2, 2]])
    attn = torch.ones_like(ids)
    labels = torch.tensor([0.0, 1.0])
    loader = _loader(ids, attn, labels)

    with torch.no_grad():
        z0 = head.logit(backbone(ids).last_hidden_state)
        before = probe_loss(z0, attn, labels).item()

    train_probe(backbone, head, loader, epochs=80, lr=0.05, device="cpu")

    with torch.no_grad():
        z1 = head.logit(backbone(ids).last_hidden_state)
        after = probe_loss(z1, attn, labels).item()
        # head should have learned to separate: token-2 seq scores higher than token-1 seq
        p = head.p(backbone(ids).last_hidden_state).mean(dim=1)
    assert after < before * 0.5
    assert p[1] > p[0]


def test_train_probe_logs_loss_history(capsys):
    """verbose=True prints a per-epoch loss line and records head._train_history (the loss curve)."""
    torch.manual_seed(0)
    H = 8
    backbone = _StubBackbone(H)
    head = ValueHead(H)
    ids = torch.tensor([[1, 1, 1], [2, 2, 2]])
    attn = torch.ones_like(ids)
    labels = torch.tensor([0.0, 1.0])
    loader = _loader(ids, attn, labels)

    train_probe(backbone, head, loader, epochs=3, lr=0.05, val_loader=loader, device="cpu",
                verbose=True)
    hist = head._train_history
    assert len(hist) == 3 and all({"epoch", "train_loss", "val_loss"} <= h.keys() for h in hist)
    assert hist[0]["val_loss"] is not None
    out = capsys.readouterr().out
    assert "epoch 1/3" in out and "train_loss=" in out and "val_loss=" in out


def test_backbone_stays_frozen():
    backbone = _StubBackbone(8)
    head = ValueHead(8)
    ids = torch.tensor([[1, 2], [2, 1]])
    attn = torch.ones_like(ids)
    labels = torch.tensor([0.0, 1.0])
    train_probe(backbone, head, _loader(ids, attn, labels), epochs=2, device="cpu")
    assert all(not p.requires_grad for p in backbone.parameters())



# --------------------------------------------------------------------------- #
# checkpoint roundtrip                                                        #
# --------------------------------------------------------------------------- #
def test_checkpoint_loads_via_runner_loader(tmp_path):
    H = 16
    head = ValueHead(H)
    for p in head.parameters():
        nn.init.normal_(p)
    path = str(tmp_path / "vhead.pt")
    w, sidecar = save_probe_checkpoint(
        path, head, feature_spec=FeatureSpec(), threshold=0.42,
        meta={"model": "stub", "loss": "focal", "coh_weight": 0.1},
    )
    # primary file loads via the SAME loader the runners use
    loaded = load_value_head(w, H, device="cpu")
    x = torch.randn(3, H)
    assert torch.allclose(head.p(x), loaded.p(x), atol=1e-6)
    # sidecar carries spec + calibrated threshold for the operator
    meta = load_probe_meta(w)
    assert meta["threshold"] == 0.42
    assert meta["feature_spec"]["norm"] == "post"
    assert meta["meta"]["loss"] == "focal"


def test_feature_collator_pads_and_masks():
    from value_steer.train_probe import FeatureCollator
    H = 5
    feats = [torch.randn(3, H), torch.randn(1, H), torch.randn(2, H)]
    batch = [{"features": f, "prompt_len": pl, "label": lab}
             for f, pl, lab in zip(feats, [1, 0, 1], [1.0, 0.0, 1.0])]
    padded, attn, plens, labels = FeatureCollator()(batch)
    assert padded.shape == (3, 3, H)              # padded to Lmax=3
    assert attn.tolist() == [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
    assert torch.allclose(padded[1, 0], feats[1][0]) and float(padded[1, 1:].abs().sum()) == 0.0
    assert plens.tolist() == [1, 0, 1] and labels.tolist() == [1.0, 0.0, 1.0]


def test_feature_cache_dataset_roundtrip(tmp_path):
    """FeatureCacheDataset memmaps the on-disk cache format written by write_feature_cache."""
    import numpy as np

    from value_steer.train_probe import FeatureCacheDataset, FeatureCollator
    H = 4
    feats = [np.random.randn(3, H).astype(np.float16), np.random.randn(2, H).astype(np.float16)]
    d = tmp_path / "cache"
    d.mkdir()
    total = 0
    with open(d / "feats.f16", "wb") as fb, open(d / "index.jsonl", "w") as ix:
        for k, f in enumerate(feats):
            f.tofile(fb)
            ix.write(json.dumps({"offset": total, "length": f.shape[0],
                                 "prompt_len": k, "label": float(k)}) + "\n")
            total += f.shape[0]
    (d / "meta.json").write_text(json.dumps({"H": H, "total_rows": total, "count": 2}))

    ds = FeatureCacheDataset(str(d))
    assert len(ds) == 2 and ds.H == H
    assert ds[0]["features"].shape == (3, H) and ds[1]["prompt_len"] == 1
    assert np.allclose(ds[0]["features"].numpy(), feats[0])
    padded, attn, plens, labels = FeatureCollator()([ds[0], ds[1]])
    assert padded.shape == (2, 3, H) and attn.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_resolve_threshold_precedence(tmp_path):
    from value_steer.train_probe import resolve_threshold
    H = 8
    head = ValueHead(H)
    path = str(tmp_path / "vhead.pt")
    save_probe_checkpoint(path, head, feature_spec=FeatureSpec(), threshold=0.3, meta={})

    # explicit threshold wins over the sidecar
    assert resolve_threshold({"threshold": 0.7, "value_head_path": path}) == 0.7
    # no explicit threshold -> falls back to the calibrated sidecar value
    assert resolve_threshold({"value_head_path": path}) == 0.3
    # no sidecar / no path -> default
    assert resolve_threshold({}) == 0.5
    assert resolve_threshold({"value_head_path": str(tmp_path / "missing.pt")}, default=0.42) == 0.42
    # sidecar present but threshold null -> default
    save_probe_checkpoint(path, head, feature_spec=FeatureSpec(), threshold=None, meta={})
    assert resolve_threshold({"value_head_path": path}) == 0.5


# small local approx helper to avoid importing pytest.approx at module top
def pytest_approx(x, rel=1e-5, abs=1e-7):
    import pytest
    return pytest.approx(x, rel=rel, abs=abs)
