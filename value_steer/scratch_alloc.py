# SPDX-License-Identifier: Apache-2.0
"""
Fixed scratch-slot allocator for VFD's K-candidate forward (pure index math, no GPU).

The worker-side runner cannot allocate KV blocks from the scheduler's pool, so VFD
reserves a fixed pool of candidate token-slots at init and hands them out per step.
This is just the free-list managing that pool: allocate slots for a step's candidates,
free them once the winner's KV is committed. Indices only -- the backing KV tensor
lives in the runner. Pure and unit-testable on CPU.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence


def candidate_block_layout(
    positions: Sequence[int],
    prefix_blocks: Sequence[Sequence[int]],
    scratch_block_ids: Sequence[int],
    scratch_idx: Sequence[int],
    K: int,
    block_size: int,
) -> dict:
    """Pure paged-KV index math for VFD's K-candidate forward (no torch, no vLLM).

    For each candidate (request i, candidate k) at its request's generation position
    ``p = positions[i]``, FlashAttention will write the new token's K/V to
    ``block_table[p // block_size]`` at offset ``p % block_size`` and read positions
    0..p from that block table. So the candidate's new token must live in the block that
    serves ``tail_idx = p // block_size`` -- the SAME block index that holds the request's
    real prefix tail. We therefore set that one entry to the candidate's PRIVATE scratch
    block (a copy of the real tail block; see _copy_real_tail_to_scratch) and keep the
    earlier prefix entries pointing at the request's real blocks. Block-aligned positions
    (offset 0) start a fresh block -> no copy needed.

    Args:
        positions: per-request generation position p (len R).
        prefix_blocks: per-request real block ids (len R; each a list of physical blocks).
        scratch_block_ids: the reserved scratch block ids (index space for scratch_idx).
        scratch_idx: flattened per-candidate scratch indices (len R*K).
        K: candidates per request. block_size: tokens per block.

    Returns a dict of lists keyed by the flattened row index ``row = i*K + k`` (len R*K):
        block_table  list[list[int]]  paged block table per row (width = max prefix + 1)
        slot_mapping list[int]        flat slot the new-token K/V is written to
        scratch_blk  list[int]        the candidate's scratch block
        real_tail_blk list[int]       the request's real tail block (copy src / commit dst)
        offset       list[int]        p % block_size
        needs_copy   list[bool]       offset > 0 (tail shares a block with real prefix)
        width        int              block-table width
    """
    R = len(positions)
    if len(prefix_blocks) != R:
        raise ValueError("positions and prefix_blocks must have the same length (R)")
    if len(scratch_idx) != R * K:
        raise ValueError(f"scratch_idx must have R*K={R * K} entries, got {len(scratch_idx)}")
    width = max((len(pb) for pb in prefix_blocks), default=0) + 1

    block_table: list[list[int]] = []
    slot_mapping: list[int] = []
    scratch_blk: list[int] = []
    real_tail_blk: list[int] = []
    offset: list[int] = []
    needs_copy: list[bool] = []
    for i in range(R):
        p = int(positions[i])
        tail_idx, off = divmod(p, block_size)
        prefix = list(prefix_blocks[i])
        rtb = int(prefix[tail_idx]) if tail_idx < len(prefix) else 0
        for k in range(K):
            row = i * K + k
            sblk = int(scratch_block_ids[scratch_idx[row]])
            r = [0] * width
            r[:tail_idx] = [int(b) for b in prefix[:tail_idx]]   # real full prefix blocks
            r[tail_idx] = sblk                                   # candidate's private tail
            block_table.append(r)
            slot_mapping.append(sblk * block_size + off)
            scratch_blk.append(sblk)
            real_tail_blk.append(rtb)
            offset.append(off)
            needs_copy.append(off > 0)
    return {
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "scratch_blk": scratch_blk,
        "real_tail_blk": real_tail_blk,
        "offset": offset,
        "needs_copy": needs_copy,
        "width": width,
    }


class ScratchAllocator:
    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self._free: deque[int] = deque(range(num_slots))
        self._allocated: set[int] = set()

    @property
    def available(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        """Hand out n free slots (FIFO). Raises if the pool can't satisfy the request."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > len(self._free):
            raise RuntimeError(f"scratch exhausted: need {n}, have {len(self._free)}")
        out = [self._free.popleft() for _ in range(n)]
        self._allocated.update(out)
        return out

    def free(self, slots: Iterable[int]) -> None:
        """Return slots to the pool. Raises on freeing a slot that isn't allocated
        (catches double-free / foreign indices)."""
        for s in slots:
            if s not in self._allocated:
                raise ValueError(f"slot {s} is not currently allocated")
            self._allocated.discard(s)
            self._free.append(s)

    def reset(self) -> None:
        self._free = deque(range(self.num_slots))
        self._allocated.clear()
