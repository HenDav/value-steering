# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for value_steer.scratch_alloc (pure index math).

Run:  pytest tests/test_scratch_alloc.py -q
"""

import pytest

from value_steer.scratch_alloc import ScratchAllocator, candidate_block_layout


def test_allocate_and_free_roundtrip():
    a = ScratchAllocator(8)
    assert a.available == 8
    s = a.allocate(3)
    assert len(s) == 3 and a.available == 5
    a.free(s)
    assert a.available == 8


def test_allocate_is_fifo_and_distinct():
    a = ScratchAllocator(4)
    assert a.allocate(2) == [0, 1]
    assert a.allocate(2) == [2, 3]


def test_exhaustion_raises_and_leaves_pool_intact():
    a = ScratchAllocator(2)
    a.allocate(2)
    with pytest.raises(RuntimeError):
        a.allocate(1)
    assert a.available == 0


def test_double_free_raises():
    a = ScratchAllocator(4)
    s = a.allocate(2)
    a.free(s)
    with pytest.raises(ValueError):
        a.free(s)                       # already returned


def test_free_foreign_slot_raises():
    a = ScratchAllocator(4)
    with pytest.raises(ValueError):
        a.free([0])                     # never allocated


def test_reset_restores_full_pool():
    a = ScratchAllocator(4)
    a.allocate(3)
    a.reset()
    assert a.available == 4
    assert a.allocate(4) == [0, 1, 2, 3]


def test_zero_size_rejected():
    with pytest.raises(ValueError):
        ScratchAllocator(0)


def test_allocate_zero_is_noop():
    a = ScratchAllocator(4)
    assert a.allocate(0) == []
    assert a.available == 4


# --------------------------------------------------------------------------- #
# candidate_block_layout -- the high-risk paged-KV slot/block math (CPU guard  #
# for the bugs that only the GPU greedy-equivalence test caught otherwise).    #
# --------------------------------------------------------------------------- #
def test_layout_midblock_puts_scratch_at_tail_idx_not_end():
    # prompt len 6, block_size 16 -> first generated token at position 6: tail_idx 0,
    # offset 6. The candidate's tail block (logical index 0) MUST be the scratch block,
    # with the real prefix preserved before it. (The original bug appended scratch at the
    # END of the row instead of at tail_idx -> caught here.)
    out = candidate_block_layout(
        positions=[6], prefix_blocks=[[1, 0, 0]],  # request occupies physical block 1
        scratch_block_ids=[100, 101], scratch_idx=[0], K=1, block_size=16,
    )
    assert out["block_table"] == [[100, 0, 0, 0]]   # logical block 0 -> scratch; width = 3+1
    assert out["slot_mapping"] == [100 * 16 + 6]
    assert out["offset"] == [6] and out["needs_copy"] == [True]
    assert out["real_tail_blk"] == [1]              # copy src / commit dst = the real block


def test_layout_block_aligned_starts_fresh_block_no_copy():
    # position 16 (block-aligned): tail_idx 1, offset 0 -> a brand-new block, no tail copy.
    out = candidate_block_layout(
        positions=[16], prefix_blocks=[[1, 0]], scratch_block_ids=[100],
        scratch_idx=[0], K=1, block_size=16,
    )
    assert out["offset"] == [0] and out["needs_copy"] == [False]
    assert out["block_table"][0][1] == 100          # scratch at logical block index 1
    assert out["block_table"][0][0] == 1            # real prefix block kept at index 0
    assert out["slot_mapping"] == [100 * 16 + 0]


def test_layout_multi_request_multi_candidate_distinct_scratch():
    # 2 requests x K=2 candidates: distinct scratch blocks per row, per-request tail/offset.
    out = candidate_block_layout(
        positions=[6, 20], prefix_blocks=[[1, 0], [2, 3]],
        scratch_block_ids=[50, 51, 52, 53], scratch_idx=[0, 1, 2, 3], K=2, block_size=16,
    )
    assert out["scratch_blk"] == [50, 51, 52, 53]               # one private block per candidate
    # req0 p=6: tail_idx0 off6 -> scratch at idx0; req1 p=20: tail_idx1 off4 -> scratch at idx1
    assert out["block_table"][0][0] == 50 and out["offset"][0] == 6
    assert out["block_table"][2] == [2, 52, 0] and out["offset"][2] == 4  # req1 keeps real blk 2
    assert out["real_tail_blk"] == [1, 1, 3, 3]                  # req1's tail (logical 1) = blk 3
    assert out["slot_mapping"] == [50 * 16 + 6, 51 * 16 + 6, 52 * 16 + 4, 53 * 16 + 4]


def test_layout_position_anchoring_consecutive_steps():
    # Anchoring invariant (the _next_pos fix): a prompt of length L generates at positions
    # L, L+1, L+2, ... -> offsets L%bs, (L+1)%bs, ... NOT starting at 0. Reproduces the
    # bootstrap/steady positions and asserts offsets advance by one (block_size 16, L=6).
    L, bs = 6, 16
    offs = [
        candidate_block_layout([L + t], [[1] * 8], [100], [0], 1, bs)["offset"][0]
        for t in range(5)
    ]
    assert offs == [6, 7, 8, 9, 10]      # anchored at L=6 (a regression to 0 would give 0..4)


def test_layout_validates_shapes():
    with pytest.raises(ValueError):
        candidate_block_layout([6], [[1], [2]], [0], [0], 1, 16)        # R mismatch
    with pytest.raises(ValueError):
        candidate_block_layout([6], [[1]], [0, 1], [0], 2, 16)         # scratch_idx len != R*K=2
