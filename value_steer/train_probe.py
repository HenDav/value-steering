# SPDX-License-Identifier: Apache-2.0
"""
Probe training for the value-steering plugin (shared infra).

Trains the shared value_probe.ValueHead on top of a FROZEN backbone, reproducing the
VFD paper's DenseValueModel scalar-head objective: per-token focal loss against the
sequence-level label, plus a temporal-difference coherence term that penalizes large
jumps in the value logit between adjacent tokens. The feature is the backbone's
final-layer POST-norm last_hidden_state (cast to fp32) -- the SAME tensor the runners
score at inference, so a probe trained here drops straight into abstention/VFD.

Label-agnostic: you pass per-example labels in [0, 1]; whether they mean P(unsafe)
(VFD) or P(should-quit) (abstention) lives in your data, not here. The objective and
feature are shared; only the labels differ.

Writing this is CPU work; the loss/feature/checkpoint pieces are unit-tested on CPU
with a stub backbone. Running it at scale needs a GPU (and, for multi-GPU, wrap the
head in DDP -- the backbone is frozen so only head grads sync). Calibration is a
SEPARATE step (calibration.py), run after training on held-out trajectories.

Checkpoint schema: `save_probe_checkpoint` writes the bare head state dict to `path`
(so value_probe.load_value_head loads it unchanged) plus a `path + ".meta.json"`
sidecar carrying the FeatureSpec, the calibrated threshold, and training metadata --
the operator reads the sidecar to set `threshold` in the runner's additional_config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .value_probe import DEFAULT_SPEC, FeatureSpec, ValueHead


# --------------------------------------------------------------------------- #
# Losses (pure; CPU-testable). Ported from training.py.                       #
# --------------------------------------------------------------------------- #
def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_pos: float = 0.7,
    gamma: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Binary focal loss (Lin et al.) for soft targets in [0, 1]. Same shape in/out
    (reduction='none'). Matches training.py:focal_loss_with_logits."""
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
    loss = alpha_t * (1 - pt).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def td_coherence_loss(token_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared difference of the value logit between adjacent valid tokens:
    mean over transitions of (z_{t+1} - z_t)^2. Encourages a smooth value trajectory."""
    z_prev = token_logits[:, :-1]
    z_next = token_logits[:, 1:]
    trans_mask = attention_mask[:, :-1].float() * attention_mask[:, 1:].float()
    return ((z_next - z_prev) ** 2 * trans_mask).sum() / trans_mask.sum().clamp_min(1.0)


def probe_loss(
    token_logits: torch.Tensor,      # [B, L]
    attention_mask: torch.Tensor,    # [B, L]
    labels: torch.Tensor,            # [B] in [0, 1]
    *,
    loss_name: str = "focal",
    use_td: bool = True,
    coh_weight: float = 0.1,
    alpha_pos: float = 0.7,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Per-token loss against the sequence label (broadcast to every token),
    masked-mean over real tokens then mean over the batch, plus coh_weight * TD term.
    Matches the scalar-head loss in training.py's loop."""
    B, L = token_logits.shape
    targets = labels.view(B, 1).expand(B, L).to(token_logits.dtype)
    mask = attention_mask.float()
    if loss_name == "focal":
        per_token = focal_loss_with_logits(token_logits, targets, alpha_pos, gamma)
    elif loss_name == "bce":
        per_token = F.binary_cross_entropy_with_logits(token_logits, targets, reduction="none")
    else:
        raise ValueError(f"unknown loss_name: {loss_name}")
    per_token = per_token * mask
    token_loss = (per_token.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean()
    if use_td:
        return token_loss + coh_weight * td_coherence_loss(token_logits, attention_mask)
    return token_loss


# --------------------------------------------------------------------------- #
# Feature extraction (frozen backbone -> post-norm hidden, fp32).             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(backbone, input_ids, attention_mask) -> torch.Tensor:
    """Run the frozen backbone and return the final POST-norm last_hidden_state,
    detached. `backbone` is the HF base model (e.g. AutoModelForCausalLM(...).model),
    matching FeatureSpec(layer='final', norm='post'). The head casts to fp32 itself."""
    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    return hs.detach()


def batch_token_logits(backbone, head: ValueHead, input_ids, attention_mask) -> torch.Tensor:
    """[B, L, H] post-norm features -> [B, L] value logits (grad flows into head only)."""
    hs = extract_features(backbone, input_ids, attention_mask)
    return head.logit(hs)


# --------------------------------------------------------------------------- #
# Dataset / collator (ported from training.py).                              #
# --------------------------------------------------------------------------- #
def _ids_list(out):
    """Normalize apply_chat_template(tokenize=True) output to a flat list[int]. transformers
    5.x returns a BatchEncoding/dict (input_ids under a key) where 4.x returned a bare list;
    a single conversation may also come back nested ([[ids]])."""
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"] if isinstance(out, dict) else out.input_ids
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


class ProbeDataset(Dataset):
    """(prompt, response, label) -> tokenized full conversation + prompt_len + label.
    `tokenizer` is duck-typed: needs apply_chat_template(messages, tokenize=True,
    add_generation_prompt=...) -> list[int]."""

    def __init__(self, tokenizer, prompts, responses, labels):
        assert len(prompts) == len(responses) == len(labels)
        self.tok = tokenizer
        self.prompts = prompts
        self.responses = responses
        self.labels = labels

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": self.prompts[idx]},
             {"role": "assistant", "content": self.responses[idx]}],
            tokenize=True, add_generation_prompt=False,
        )
        prompt_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": self.prompts[idx]}],
            tokenize=True, add_generation_prompt=True,
        )
        full, prompt_ids = _ids_list(full), _ids_list(prompt_ids)
        return {"input_ids": full, "prompt_len": len(prompt_ids), "label": float(self.labels[idx])}


class ProbeCollator:
    """Right-pad to a square batch. Returns (input_ids, attention_mask, prompt_lens,
    labels). Loss masks by attention_mask (all real tokens, prompt included -- the
    paper trains the value at every position); prompt_lens is provided if you want a
    response-only variant."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        padded = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        attn = (padded != self.pad_token_id).long()
        prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        return padded, attn, prompt_lens, labels


class FeatureCacheDataset(Dataset):
    """Streams pre-extracted per-token features from an on-disk cache (written by
    vllm_extract.write_feature_cache): a memmapped float16 blob of all examples' [Li,H] rows
    concatenated, plus an index. Because the backbone is FROZEN its features never change, so
    we extract ONCE and train the head over this cache for many epochs -- no per-epoch
    re-forward, no model in the training process, and O(1) RAM regardless of dataset size
    (the OS pages the memmap), so big datasets are supported natively. Yields
    {features[Li,H], prompt_len, label}."""

    def __init__(self, cache_dir: str):
        with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.H = int(meta["H"])
        self.meta = meta
        with open(os.path.join(cache_dir, "index.jsonl"), encoding="utf-8") as f:
            self.index = [json.loads(l) for l in f]
        self._mm = np.memmap(os.path.join(cache_dir, "feats.f16"), dtype=np.float16,
                             mode="r", shape=(int(meta["total_rows"]), self.H))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        r = self.index[idx]
        rows = np.array(self._mm[r["offset"]: r["offset"] + r["length"]])   # copy out of the memmap
        return {"features": torch.from_numpy(rows), "prompt_len": int(r["prompt_len"]),
                "label": float(r["label"])}


class FeatureCollator:
    """Right-pad cached [Li, H] features to [B, Lmax, H] + attention mask. Returns the SAME
    tuple shape as ProbeCollator (features in slot 0), so train_probe's loop consumes it
    unchanged with feature_fn=identity."""

    def __call__(self, batch):
        feats = [b["features"] for b in batch]
        lens = [f.shape[0] for f in feats]
        B, Lmax, H = len(feats), max(lens), feats[0].shape[1]
        padded = feats[0].new_zeros((B, Lmax, H))
        attn = torch.zeros((B, Lmax), dtype=torch.long)
        for i, (f, li) in enumerate(zip(feats, lens)):
            padded[i, :li] = f
            attn[i, :li] = 1
        prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        return padded, attn, prompt_lens, labels


# --------------------------------------------------------------------------- #
# Training loop (single device; DDP is a thin wrapper since backbone is frozen).
# --------------------------------------------------------------------------- #
def train_probe(
    backbone,
    head: ValueHead,
    train_loader,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    loss_name: str = "focal",
    use_td: bool = True,
    coh_weight: float = 0.1,
    val_loader=None,
    patience: int = 3,
    grad_clip: float = 1.0,
    device: str = "cpu",
    feature_fn=None,
    weight_decay: float = 1e-2,
    warmup_frac: float = 0.0,
    verbose: bool = False,
) -> ValueHead:
    """Train `head` on a frozen backbone. Early stops on val loss (patience) when a
    val_loader is given, restoring the best head. Returns the trained head.

    Feature source: pass an HF `backbone` (the post-norm last_hidden_state path), OR a
    `feature_fn(input_ids, attn) -> [B, L, H]` to source features from elsewhere (e.g. a
    vLLM pooling pass -- the exact inference forward). Exactly one is used; with feature_fn
    the backbone is ignored (may be None), since the backbone is frozen either way.

    `warmup_frac` > 0 enables a linear warmup-then-decay LR schedule (pure torch, no transformers)
    over `epochs * len(train_loader)` steps -- matching the reference recipe; 0.0 keeps a constant
    LR. `verbose` prints a per-epoch loss line; the per-epoch history is on `head._train_history`."""
    extract = feature_fn
    if extract is None:
        if backbone is None:
            raise ValueError("train_probe needs either a backbone or a feature_fn")
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        extract = lambda ids, attn: extract_features(backbone, ids, attn)  # noqa: E731
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    sched = None
    if warmup_frac > 0.0:
        total = max(1, epochs * len(train_loader))
        warm = max(1, int(warmup_frac * total))

        def _lr_lambda(step):                      # linear warmup -> linear decay to 0
            if step < warm:
                return step / warm
            return max(0.0, (total - step) / max(1, total - warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    history = []
    best, bad, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        head.train()
        run_loss, n_batches = 0.0, 0
        for input_ids, attn, _plen, labels in train_loader:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            opt.zero_grad()
            logits = head.logit(extract(input_ids, attn))
            loss = probe_loss(logits, attn, labels, loss_name=loss_name,
                              use_td=use_td, coh_weight=coh_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            run_loss += float(loss)
            n_batches += 1
        train_loss = run_loss / max(n_batches, 1)

        v, is_best = None, False
        if val_loader is not None:
            v = _eval_loss(extract, head, val_loader, device, loss_name, use_td, coh_weight)
            if v < best - 1e-4:
                best, bad, is_best = v, 0, True
                best_state = {k: t.detach().clone() for k, t in head.net.state_dict().items()}
            else:
                bad += 1
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": v})
        if verbose:
            msg = f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f}"
            if v is not None:
                msg += f" val_loss={v:.4f}" + (" *best" if is_best else f" (no-improve {bad}/{patience})")
            print(msg, flush=True)
        if val_loader is not None and bad >= patience:
            if verbose:
                print(f"early stop at epoch {epoch + 1} (val no-improve {patience})", flush=True)
            break
    if best_state is not None:
        head.net.load_state_dict(best_state)
    head._train_history = history
    return head


@torch.no_grad()
def _eval_loss(extract, head, loader, device, loss_name, use_td, coh_weight) -> float:
    head.eval()
    total, n = 0.0, 0
    for input_ids, attn, _plen, labels in loader:
        input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
        logits = head.logit(extract(input_ids, attn))
        total += float(probe_loss(logits, attn, labels, loss_name=loss_name,
                                  use_td=use_td, coh_weight=coh_weight))
        n += 1
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Checkpoint schema.                                                          #
# --------------------------------------------------------------------------- #
def save_probe_checkpoint(
    path: str,
    head: ValueHead,
    *,
    feature_spec: FeatureSpec = DEFAULT_SPEC,
    threshold: float | None = None,
    meta: dict | None = None,
) -> tuple[str, str]:
    """Write the bare head state dict to `path` (loadable by value_probe.load_value_head)
    and a `path + '.meta.json'` sidecar with the feature spec, calibrated threshold, and
    training metadata. Returns (weights_path, sidecar_path)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)   # create trained/<run>/ if absent (don't crash post-train)
    torch.save(head.net.state_dict(), path)
    sidecar = path + ".meta.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(
            {"feature_spec": asdict(feature_spec), "threshold": threshold, "meta": meta or {}},
            f, indent=2,
        )
    return path, sidecar


def load_probe_meta(weights_path: str) -> dict:
    """Read the sidecar written by save_probe_checkpoint (feature_spec, threshold, meta)."""
    with open(weights_path + ".meta.json", encoding="utf-8") as f:
        return json.load(f)


def resolve_threshold(cfg: dict, default: float = 0.5) -> float:
    """Threshold precedence for a runner: explicit cfg['threshold'] > the calibrated value
    written into the checkpoint's .meta.json sidecar (when cfg['value_head_path'] is set) >
    `default`. Pure (no vLLM) so it is CPU-testable and reusable by the runners."""
    t = cfg.get("threshold")
    if t is not None:
        return float(t)
    path = cfg.get("value_head_path")
    if path and os.path.exists(path + ".meta.json"):
        st = load_probe_meta(path).get("threshold")   # corrupt sidecar raises, not swallowed
        if st is not None:
            return float(st)
    return float(default)
