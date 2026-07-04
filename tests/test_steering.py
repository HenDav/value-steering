# SPDX-License-Identifier: Apache-2.0
"""
CPU unit tests for the value-steering plugin's pure logic.

These cover the decision logic that does NOT need a GPU, a model, or even vLLM:
  * value_probe.ValueHead   -- save/load roundtrip, fp32 contract
  * value_probe.request_threshold
  * steering_ops.warp_logits     (temperature + nucleus)
  * steering_ops.select_vfd       (first-safe / argmin / ARGS)
  * steering_ops.force_eos_rows   (EOS-forcing math + NaN-safety)

Run:  pytest test_steering.py -q
They complement compat_check.py: the harness checks vLLM CONTRACTS; these check OUR
LOGIC. Together: a version bump that keeps the contracts but our math regresses is
caught here; a bump that breaks a contract is caught there.
"""

import math
from types import SimpleNamespace

import torch

from value_steer.steering_ops import force_eos_rows, select_vfd, warp_logits
from value_steer.value_probe import FeatureSpec, ValueHead, load_value_head, request_threshold


# --------------------------------------------------------------------------- #
# value_probe.ValueHead                                                       #
# --------------------------------------------------------------------------- #
def test_value_head_roundtrip(tmp_path):
    torch.manual_seed(0)
    H = 32
    head = ValueHead(H)
    for p in head.parameters():           # randomize so defaults don't hide bugs
        torch.nn.init.normal_(p)
    path = tmp_path / "vhead.pt"
    torch.save(head.net.state_dict(), path)

    loaded = load_value_head(str(path), H, device="cpu")
    x = torch.randn(5, H)
    assert torch.allclose(head.p(x), loaded.p(x), atol=1e-6)


def test_value_head_is_fp32_and_probability():
    H = 16
    head = ValueHead(H)
    assert all(p.dtype == torch.float32 for p in head.parameters())
    # bf16 input must be accepted (cast to fp32 internally) and yield valid probs
    out = head.p(torch.randn(4, H, dtype=torch.bfloat16))
    assert out.dtype == torch.float32
    assert out.shape == (4,)
    assert torch.all((out > 0) & (out < 1))


def test_value_head_load_arch_mismatch_raises(tmp_path):
    # A checkpoint from a different architecture must fail loudly (strict load).
    path = tmp_path / "wrong.pt"
    torch.save({"0.weight": torch.randn(3, 3)}, path)
    try:
        load_value_head(str(path), 16, device="cpu")
        assert False, "expected a load error on architecture mismatch"
    except Exception:
        pass



# --------------------------------------------------------------------------- #
# value_probe.request_threshold                                               #
# --------------------------------------------------------------------------- #
def test_request_threshold_paths():
    sp = SimpleNamespace(extra_args={"abstain_threshold": 0.3})
    assert request_threshold(sp, "abstain_threshold", 0.5) == 0.3      # present
    assert request_threshold(sp, "missing", 0.5) == 0.5               # absent key
    assert request_threshold(SimpleNamespace(extra_args=None), "k", 0.5) == 0.5
    assert request_threshold(None, "k", 0.5) == 0.5                   # no sampling params


def test_feature_spec_defaults():
    s = FeatureSpec()
    assert (s.layer, s.norm, s.pooling, s.dtype) == ("final", "post", "none", "fp32")


# --------------------------------------------------------------------------- #
# steering_ops.warp_logits                                                    #
# --------------------------------------------------------------------------- #
def test_warp_temperature_scales():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(warp_logits(logits, 2.0, 1.0), logits / 2.0)
    assert torch.allclose(warp_logits(logits, 1.0, 1.0), logits)


def test_warp_top_p_keeps_nucleus():
    # softmax([3,2,1,0]) ~ [.643,.236,.087,.032]; cumsum ~ [.643,.879,.966,1]
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    out = warp_logits(logits, 1.0, 0.9)
    finite = torch.isfinite(out)
    # top two tokens (cum .879 < .9) plus the one crossing .9 are kept; tail dropped
    assert bool(finite[0]) and bool(finite[1]) and bool(finite[2])
    assert not bool(finite[3])
    # the highest-logit token is always kept
    assert torch.isfinite(out[logits.argmax()])


def test_warp_top_p_full_is_noop():
    logits = torch.randn(10)
    assert torch.allclose(warp_logits(logits, 1.0, 1.0), logits)


# --------------------------------------------------------------------------- #
# steering_ops.select_vfd                                                     #
# --------------------------------------------------------------------------- #
def test_select_first_safe_lowest_index():
    # row 0: cols 1 and 3 safe -> pick 1 (first). row 1: only col 2 safe -> 2.
    p = torch.tensor([[0.9, 0.1, 0.8, 0.2],
                      [0.9, 0.7, 0.3, 0.6]])
    c = torch.tensor([0.5, 0.5])
    out = select_vfd(p, c)
    assert out.tolist() == [1, 2]


def test_select_none_safe_argmin():
    p = torch.tensor([[0.9, 0.7, 0.95, 0.8]])   # none < 0.5 -> safest = argmin = col 1
    c = torch.tensor([0.5])
    assert select_vfd(p, c).tolist() == [1]


def test_select_args_fallback_changes_choice():
    # none safe. plain argmin(p) -> col 0 (0.80 vs 0.82). ARGS subtracts prob_weight*
    # exp(logp): col 1 is far more likely, flipping the choice to col 1.
    p = torch.tensor([[0.80, 0.82]])
    c = torch.tensor([0.5])
    logp = torch.log(torch.tensor([[0.01, 0.99]]))
    assert select_vfd(p, c).tolist() == [0]                       # plain argmin
    out = select_vfd(p, c, args_fallback=True, prob_weight=1.0, logp=logp)
    assert out.tolist() == [1]                                    # ARGS prefers likely


def test_select_args_fallback_requires_logp():
    p = torch.tensor([[0.9, 0.8]])
    c = torch.tensor([0.5])
    try:
        select_vfd(p, c, args_fallback=True)
        assert False, "expected ValueError without logp"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# steering_ops.force_eos_rows                                                 #
# --------------------------------------------------------------------------- #
def test_force_eos_rows_forces_only_masked():
    logits = torch.randn(3, 7).clone()
    before_row1 = logits[1].clone()
    eos = 4
    mask = torch.tensor([True, False, True])
    rows = force_eos_rows(logits, mask, eos)
    assert rows.tolist() == [0, 2]
    # forced rows: argmax is EOS, everything else -inf
    for r in (0, 2):
        assert int(logits[r].argmax()) == eos
        others = [j for j in range(7) if j != eos]
        assert torch.all(logits[r, others] == float("-inf"))
    # untouched row preserved
    assert torch.allclose(logits[1], before_row1)


def test_force_eos_rows_no_nan_under_softmax():
    logits = torch.randn(2, 5)
    force_eos_rows(logits, torch.tensor([True, False]), 2)
    probs = torch.softmax(logits, dim=-1)
    assert torch.isfinite(probs).all()
    assert math.isclose(float(probs[0, 2]), 1.0, abs_tol=1e-5)   # all mass on EOS


def test_force_eos_rows_empty_mask_noop():
    logits = torch.randn(3, 5).clone()
    snap = logits.clone()
    rows = force_eos_rows(logits, torch.zeros(3, dtype=torch.bool), 1)
    assert rows.numel() == 0
    assert torch.allclose(logits, snap)
