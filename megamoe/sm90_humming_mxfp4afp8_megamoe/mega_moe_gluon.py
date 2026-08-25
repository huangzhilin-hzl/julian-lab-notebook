################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Canonical SM90 Gluon MegaMoE implementation.

This file owns every Gluon partition used by the Gluon path, including
dispatch, context management, grouped GEMM, scatter, and combine.

The only GEMM ABI owned by this module is the persistent compact 1D-by-2D
path.  Exactly one CTA is launched per SM and each CTA walks a deterministic
sequence of logical M/N tiles over a BLOCK_M-padded activation pool and its
MN-major scale pool.

All A, SFA, and B tiles use Hopper TMA.  Every scheduler acquires packed
per-expert states and derives compact offsets locally.  The A producer owns
both readiness edges: dispatch-to-FC1 and FC1-publication-to-FC2.  FC1 applies
the granularity-8 gate/up SwiGLU epilogue, route weight, per-row/per-64 FP8
quantization, and release publication before FC2 consumes the compact L2
pool.  FC2 scatters every valid route directly back to its source rank and
reduces the source-local top-k slots into the final token-major BF16 output.
"""

from dataclasses import dataclass
import math

from packaging.version import InvalidVersion, Version
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)

try:
    from triton.experimental.gluon.language import (
        barrier as _gluon_partition_barrier,
    )
except ImportError:
    # Gluon 3.6 exported the same CTA synchronization builtin under this name.
    from triton.experimental.gluon.language import (
        thread_barrier as _gluon_partition_barrier,
    )


_MINIMUM_TRITON_VERSION = Version("3.6.0")
_SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE = 128
_SM90_MEGA_MOE_PRE_DISPATCH_GROUPS_PER_CTA = 64
_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS = 32
_SM90_MEGA_MOE_PRE_DISPATCH_THREADS = (
    _SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS * 32
)
_SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS = 8
_SM90_MEGA_MOE_FUSED_RESET_BLOCK_SIZE = (
    _SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS * 32
)


# Keep every partition used by the Gluon path in this canonical module.


@dataclass(frozen=True)
class SM90MegaMoESingleRankPool:
    """Local form of DeepGEMM's padded expert-pool contract.

    The multi-rank implementation uses the same fields; only the source route
    table and the final scatter destinations become peer-addressed.
    """

    acts: torch.Tensor
    acts_sf_mn_major: torch.Tensor
    topk_weights: torch.Tensor
    token_src_metadata: torch.Tensor
    expert_state: torch.Tensor
    source_routes: torch.Tensor

    @property
    def expert_recv_count(self) -> torch.Tensor:
        """Return the low-32 count view of the packed expert state.

        This is a derived compatibility/debug view, not a dispatch workspace
        buffer.  Kernels consume ``expert_state`` directly.
        """
        return (self.expert_state & 0xFFFFFFFF).to(torch.int32)

    @property
    def expert_pool_block_offsets(self) -> torch.Tensor:
        """Derive the former block-offset table from packed expert counts."""
        counts = self.expert_recv_count.to(torch.int64)
        blocks = torch.div(
            counts + _GLUON_BLOCK_M_VALUE - 1,
            _GLUON_BLOCK_M_VALUE,
            rounding_mode="floor",
        )
        return torch.cat((blocks.new_zeros(1), torch.cumsum(blocks, dim=0)))


@dataclass(frozen=True)
class SM90MegaMoEFusedFC1Workspace:
    """Reusable storage for the complete fused MegaMoE path."""

    pool: SM90MegaMoESingleRankPool
    l2_acts: torch.Tensor
    l2_acts_sf_mn_major: torch.Tensor
    output: torch.Tensor
    l1_arrival: torch.Tensor
    l2_arrival: torch.Tensor
    fc2_scatter_grid_counter: torch.Tensor
    combine_cross_rank_ready: torch.Tensor
    actual_num_pool_rows: torch.Tensor
    dispatch_counter: torch.Tensor
    expert_send_state: torch.Tensor
    l1_a_desc: object
    l1_sfa_desc: object
    l1_b_desc: object
    l2_store_desc: object
    l2_a_desc: object
    l2_sfa_desc: object
    l2_b_desc: object
    dispatch_descs: tuple[object, ...]
    l1_weight_data_ptr: int
    l2_weight_data_ptr: int
    max_pool_blocks: int
    num_pool_rows: int
    num_padded_sf_pool_tokens: int


@dataclass(frozen=True)
class SM90MegaMoEPreDispatchResult:
    """Inputs registered in symmetric memory for one dispatch invocation.

    Only the first ``num_tokens`` activation and scale rows are valid.  Top-k
    metadata is valid through the context's full ``max_tokens`` capacity;
    padded rows contain ``-1`` indices and zero weights, matching DeepGEMM's
    SM90 pre-dispatch contract.
    """

    input_acts_fp8: torch.Tensor
    input_acts_sf: torch.Tensor
    input_topk_idx: torch.Tensor
    input_topk_weights: torch.Tensor
    num_tokens: int
    hidden: int
    source_dtype: torch.dtype
    routed_scaling_factor: float
    compiled: object


@dataclass(frozen=True)
class SM90MegaMoEFusedFC1Result:
    """Artifacts from the complete registered MegaMoE chain.

    ``output`` is the observable token-major BF16 result after peer FC2 scatter
    and source-local top-k reduction.
    In ``combine_buffer``, only slots whose registered top-k index is valid
    are defined; invalid slots are deliberately not cleared or reduced.
    """

    pool: SM90MegaMoESingleRankPool
    l2_acts: torch.Tensor
    l2_acts_sf_mn_major: torch.Tensor
    l2_arrival: torch.Tensor
    output: torch.Tensor
    combine_buffer: torch.Tensor
    l1_arrival: torch.Tensor
    fc2_scatter_grid_counter: torch.Tensor
    combine_cross_rank_ready: torch.Tensor
    actual_num_pool_rows: torch.Tensor
    dispatch_counter: torch.Tensor
    workspace: SM90MegaMoEFusedFC1Workspace
    pre_dispatch: SM90MegaMoEPreDispatchResult
    compiled: object


class SM90MegaMoESymmetricContext:
    """Single-node symmetric buffers used by the SM90 pull/scatter path.

    Only PyTorch's CUDA symmetric-memory backend is used.  No runtime API from
    ``triton_dist`` participates in allocation, rendezvous, or peer mapping.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        hidden: int,
        num_experts: int,
        topk: int,
        world_size: int,
        rank: int,
        device: torch.device,
        group_name: str,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 < topk <= 32:
            raise ValueError("topk must be in [1, 32]")
        if num_experts % world_size:
            raise ValueError("num_experts must be divisible by world_size")
        if hidden % 128:
            raise ValueError("hidden must be divisible by 128")
        self.max_tokens = max_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.topk = topk
        self.world_size = world_size
        self.rank = rank
        self.experts_per_rank = num_experts // world_size
        self.max_routes = max_tokens * topk
        self.device = device

        def create(shape, dtype):
            storage_dtype = torch.int8 if dtype == torch.float8_e4m3fn else dtype
            tensor = symm_mem.empty(*shape, dtype=storage_dtype, device=device)
            if dtype == torch.float8_e4m3fn:
                tensor = tensor.view(dtype)
            handle = symm_mem.rendezvous(tensor, group=group_name)
            return tensor, handle, storage_dtype

        self.input_acts, acts_handle, acts_storage_dtype = create(
            (max_tokens, hidden),
            torch.float8_e4m3fn,
        )
        self.input_sf, sf_handle, sf_storage_dtype = create(
            (max_tokens, hidden // 128),
            torch.float32,
        )
        self.input_topk_weights, weight_handle, weight_storage_dtype = create(
            (max_tokens, topk),
            torch.float32,
        )
        self.input_topk_idx, index_handle, _ = create(
            (max_tokens, topk),
            torch.int64,
        )
        self.source_routes, route_handle, route_storage_dtype = create(
            (world_size, self.experts_per_rank, self.max_routes),
            torch.int32,
        )
        self.recv_count, count_handle, count_storage_dtype = create(
            (world_size, self.experts_per_rank),
            torch.int32,
        )
        self.expert_state, expert_state_handle, expert_state_storage_dtype = create(
            (self.experts_per_rank, ),
            torch.int64,
        )
        # Gluon exposes a real contiguous PyTorch [token, topk, hidden] tensor.
        # DeepGEMM's workspace ``Buffer(..., num_ranks=topk, ...)`` is instead
        # physically slot-major; its address formula must not be reused here.
        self.combine_buffer, combine_handle, combine_storage_dtype = create(
            (max_tokens, topk, hidden),
            torch.bfloat16,
        )
        self.dispatch_barrier, dispatch_barrier_handle, dispatch_barrier_storage_dtype = create(
            (world_size, ),
            torch.int32,
        )
        self.fused_barrier, fused_barrier_handle, fused_barrier_storage_dtype = create(
            (world_size, ),
            torch.int32,
        )

        def peer_ptrs(handle, shape, storage_dtype):
            return torch.tensor(
                [
                    handle.get_buffer(peer, shape, storage_dtype).data_ptr()
                    for peer in range(world_size)
                ],
                dtype=torch.int64,
                device=device,
            )

        self.peer_input_acts = tuple(
            acts_handle.get_buffer(
                peer,
                (max_tokens, hidden),
                acts_storage_dtype,
            ).view(torch.float8_e4m3fn)
            for peer in range(world_size)
        )
        self.peer_input_sf_ptrs = peer_ptrs(
            sf_handle,
            (max_tokens, hidden // 128),
            sf_storage_dtype,
        )
        self.peer_input_topk_weights_ptrs = peer_ptrs(
            weight_handle,
            (max_tokens, topk),
            weight_storage_dtype,
        )
        self.peer_source_routes_ptrs = peer_ptrs(
            route_handle,
            (world_size, self.experts_per_rank, self.max_routes),
            route_storage_dtype,
        )
        self.peer_recv_count_ptrs = peer_ptrs(
            count_handle,
            (world_size, self.experts_per_rank),
            count_storage_dtype,
        )
        self.peer_expert_state_ptrs = peer_ptrs(
            expert_state_handle,
            (self.experts_per_rank, ),
            expert_state_storage_dtype,
        )
        self.peer_combine_buffer_ptrs = peer_ptrs(
            combine_handle,
            (max_tokens, topk, hidden),
            combine_storage_dtype,
        )
        self.peer_dispatch_barrier_ptrs = peer_ptrs(
            dispatch_barrier_handle,
            (world_size, ),
            dispatch_barrier_storage_dtype,
        )
        self.peer_fused_barrier_ptrs = peer_ptrs(
            fused_barrier_handle,
            (world_size, ),
            fused_barrier_storage_dtype,
        )
        self._handles = (
            acts_handle,
            sf_handle,
            weight_handle,
            index_handle,
            route_handle,
            count_handle,
            expert_state_handle,
            combine_handle,
            dispatch_barrier_handle,
            fused_barrier_handle,
        )
        self._barrier_handle = acts_handle

    def barrier(self) -> None:
        self._barrier_handle.barrier()


@gluon.jit
def _sm90_mega_moe_fused_control_reset_kernel(
    l1_arrival,
    l2_arrival,
    actual_num_pool_rows,
    dispatch_counter,
    dispatch_barrier,
    fused_barrier,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    expert_state,
    expert_send_state,
    world_size: gl.constexpr,
    num_local_experts: gl.constexpr,
    num_global_experts: gl.constexpr,
    num_pool_blocks: gl.constexpr,
    reset_block_size: gl.constexpr,
    reset_layout: gl.constexpr,
):
    """Reset only fixed-size per-launch control state.

    Pool payload, scale-factor padding, and metadata are deliberately left
    untouched.  Every observable epilogue is guarded by ``valid_m`` and every
    valid row is overwritten before its arrival is published.  The compact
    arrival arrays are reset here instead of coupling dispatch completion to
    the combine partition through an in-kernel tail-cleanup rendezvous.
    """
    offsets = (
        gl.program_id(0) * reset_block_size
        + gl.arange(0, reset_block_size, layout=reset_layout)
    )
    gl.store(
        l1_arrival + offsets,
        0,
        mask=offsets < num_pool_blocks,
    )
    gl.store(
        l2_arrival + offsets,
        0,
        mask=offsets < num_pool_blocks,
    )
    gl.store(
        actual_num_pool_rows + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        dispatch_counter + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        dispatch_barrier + offsets,
        0,
        mask=offsets < world_size,
    )
    gl.store(
        fused_barrier + offsets,
        0,
        mask=offsets < world_size,
    )
    gl.store(
        fc2_scatter_grid_counter + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        combine_cross_rank_ready + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        expert_state + offsets,
        0,
        mask=offsets < num_local_experts,
    )
    gl.store(
        expert_send_state + offsets,
        0,
        mask=offsets < num_global_experts,
    )


@gluon.jit
def _sm90_mega_moe_pre_dispatch_kernel(
    x,
    x_sf,
    topk_idx,
    topk_weights,
    registered_x,
    registered_x_sf,
    registered_topk_idx,
    registered_topk_weights,
    num_tokens,
    padded_max,
    hidden: gl.constexpr,
    num_groups: gl.constexpr,
    topk: gl.constexpr,
    routed_scaling_factor,
    input_is_bf16: gl.constexpr,
    quant_layout: gl.constexpr,
    route_layout: gl.constexpr,
):
    """Register BF16 or block-scaled FP8 inputs in symmetric memory.

    The 64x128 register tile maps one half warp to each per-128 group.  With
    32 warps this is the same 1024-thread, one-token-per-CTA decomposition as
    DeepGEMM's SM90 pre-dispatch kernel for hidden sizes up to 8192.
    """
    bid = gl.program_id(0)
    route_offsets = gl.arange(
        0,
        1024,
        layout=route_layout,
    )
    if bid < num_tokens:
        group_offsets = gl.arange(
            0,
            64,
            layout=gl.SliceLayout(1, quant_layout),
        )
        columns = gl.arange(
            0,
            128,
            layout=gl.SliceLayout(0, quant_layout),
        )
        element_offsets = (
            group_offsets[:, None] * 128
            + columns[None, :]
        )
        valid_groups = group_offsets < num_groups
        valid_elements = valid_groups[:, None]
        if input_is_bf16:
            values = gl.load(
                x + bid * hidden + element_offsets,
                mask=valid_elements,
                other=0.0,
            ).to(gl.float32)
            absmax = gl.max(gl.abs(values), axis=1)
            scale = gl.maximum(absmax, 1.0e-10) * (1.0 / 448.0)
            quantized = (values * gl.fdiv(1.0, scale[:, None])).to(
                gl.float8e4nv
            )
        else:
            quantized = gl.load(
                x + bid * hidden + element_offsets,
                mask=valid_elements,
                other=0.0,
            )
            scale = gl.load(
                x_sf + bid * num_groups + group_offsets,
                mask=valid_groups,
                other=0.0,
            )
        gl.store(
            registered_x + bid * hidden + element_offsets,
            quantized,
            mask=valid_elements,
        )
        gl.store(
            registered_x_sf + bid * num_groups + group_offsets,
            scale,
            mask=valid_groups,
        )

        valid_slots = route_offsets < topk
        route = bid * topk + route_offsets
        indices = gl.load(
            topk_idx + route,
            mask=valid_slots,
            other=-1,
        ).to(gl.int64)
        weights = gl.load(
            topk_weights + route,
            mask=valid_slots,
            other=0.0,
        )
        gl.store(
            registered_topk_idx + route,
            indices,
            mask=valid_slots,
        )
        gl.store(
            registered_topk_weights + route,
            weights * routed_scaling_factor,
            mask=valid_slots,
        )
    else:
        pad_block = bid - num_tokens
        padded_route = (
            num_tokens * topk
            + pad_block * 1024
            + route_offsets
        )
        valid_padding = padded_route < padded_max * topk
        gl.store(
            registered_topk_idx + padded_route,
            -1,
            mask=valid_padding,
        )
        gl.store(
            registered_topk_weights + padded_route,
            0.0,
            mask=valid_padding,
        )


def create_sm90_symmetric_dispatch_tma_oob_3d_descriptors(
    ctx: SM90MegaMoESymmetricContext,
    pool_acts: torch.Tensor,
):
    """Create one 3D TMA tile per token, padding K with hardware OOB.

    The logical tensor is ``[token, K // 128, 128]`` while the TMA box is
    ``[1, next_power_of_2(K // 128), 128]``.  For DSV4Pro K=7168 this maps a
    56-group token into a 64-group (8 KiB) transaction.  TMA zero-fills the
    final eight groups on load and suppresses them on store, so the padded box
    never aliases the following token or pool row.
    """
    if ctx.world_size != 8:
        raise ValueError("the Gluon 3D OOB TMA dispatch specialization requires EP8")
    if ctx.hidden <= 0 or ctx.hidden % 128:
        raise ValueError("3D OOB TMA dispatch requires K divisible by 128")
    sf_groups = ctx.hidden // 128
    padded_sf_groups = triton.next_power_of_2(sf_groups)
    if padded_sf_groups > 256:
        raise ValueError("3D OOB TMA dispatch supports at most 256 K groups")
    block_shape = [1, padded_sf_groups, 128]
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=8,
        rank=3,
    )
    peer_descs = tuple(
        TensorDescriptor.from_tensor(
            peer.view(ctx.max_tokens, sf_groups, 128),
            block_shape,
            layout,
        )
        for peer in ctx.peer_input_acts
    )
    pool_desc = TensorDescriptor.from_tensor(
        pool_acts.view(pool_acts.shape[0], sf_groups, 128),
        block_shape,
        layout,
    )
    return (*peer_descs, pool_desc)


@gluon.jit
def _store_contiguous_bf16_fragment(
    ptrs,
    values,
    mask,
    width: gl.constexpr,
):
    """Store one lane-owned row fragment as one aligned global transaction."""
    packed_values = values.to(gl.uint16, bitcast=True)
    packed_mask = mask.to(gl.int8)
    if width == 8:
        return gl.inline_asm_elementwise(
            """
            {
            .reg .pred store_pred;
            setp.ne.u32 store_pred, $14, 0;
            @store_pred st.global.v4.b32 [$2], {$10, $11, $12, $13};
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            }
            """,
            "=r,=r,l,l,l,l,l,l,l,l,r,r,r,r,r,r",
            [ptrs, packed_values, packed_mask],
            dtype=gl.int8,
            is_pure=False,
            pack=8,
        )
    gl.static_assert(width == 4, "BF16 vector width must be 4 or 8")
    return gl.inline_asm_elementwise(
        """
        {
        .reg .pred store_pred;
        setp.ne.u32 store_pred, $7, 0;
        @store_pred st.global.v2.b32 [$1], {$5, $6};
        mov.u32 $0, 0;
        }
        """,
        "=r,l,l,l,l,r,r,r",
        [ptrs, packed_values, packed_mask],
        dtype=gl.int8,
        is_pure=False,
        pack=4,
    )


@gluon.jit
def _load_contiguous_bf16_fragment(ptrs, mask, width: gl.constexpr):
    """Load one lane-owned combine fragment as one aligned transaction."""
    packed_mask = mask.to(gl.int8)
    if width == 8:
        payload = gl.inline_asm_elementwise(
            """
            {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $12, 0;
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            mov.u32 $2, 0;
            mov.u32 $3, 0;
            @load_pred ld.global.v4.b32 {$0, $1, $2, $3}, [$4];
            }
            """,
            "=r,=r,=r,=r,l,l,l,l,l,l,l,l,r,r",
            [ptrs, packed_mask],
            dtype=gl.uint16,
            is_pure=False,
            pack=8,
        )
    else:
        gl.static_assert(width == 4, "BF16 vector width must be 4 or 8")
        payload = gl.inline_asm_elementwise(
            """
            {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $6, 0;
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            @load_pred ld.global.v2.b32 {$0, $1}, [$2];
            }
            """,
            "=r,=r,l,l,l,l,r",
            [ptrs, packed_mask],
            dtype=gl.uint16,
            is_pure=False,
            pack=4,
        )
    return payload.to(gl.bfloat16, bitcast=True)


@gluon.jit
def _load_packed_expert_state_acquire(expert_state_ptr):
    """Issue a non-RMW system-scope acquire load for packed expert state."""
    return gl.inline_asm_elementwise(
        "ld.global.sys.acquire.b64 $0, [$1];",
        "=l,l",
        [expert_state_ptr],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_i32_acquire_gpu(ptr):
    """Issue a non-RMW GPU-scope acquire load for a local counter."""
    return gl.inline_asm_elementwise(
        "ld.global.gpu.acquire.b32 $0, [$1];",
        "=r,l",
        [ptr],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_i32_acquire_sys_if(ptr, load_mask):
    """Issue a predicated system-acquire load for peer-visible counters."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $2, 0;
            mov.b32 $0, 0;
            @load_pred ld.global.sys.acquire.b32 $0, [$1];
        }
        """,
        "=r,l,r",
        [ptr, load_mask.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _red_i32_release_sys_if(ptr, value, store_mask):
    """Issue DeepGEMM's predicated system-release reduction add."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred store_pred;
            setp.ne.u32 store_pred, $3, 0;
            @store_pred red.release.sys.global.add.s32 [$1], $2;
            mov.b32 $0, 0;
        }
        """,
        "=r,l,r,r",
        [ptr, value, store_mask.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_packed_expert_state_acquire_if(expert_state_ptr, load_mask):
    """Issue a predicated system-acquire load without touching masked state."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $2, 0;
            mov.b64 $0, 0;
            @load_pred ld.global.sys.acquire.b64 $0, [$1];
        }
        """,
        "=l,l,r",
        [expert_state_ptr, load_mask.to(gl.int32)],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _peer_barrier_arrive_and_wait(
    peer_barrier_ptrs,
    barrier,
    world_size: gl.constexpr,
    expected: gl.constexpr,
    layout: gl.constexpr,
    participant_capacity: gl.constexpr,
):
    """Run DeepGEMM's single-counter NVLink barrier protocol.

    One lane release-reduces into each peer's counter.  Lane zero then waits
    until this rank's scalar has received ``world_size`` contributions for the
    requested phase.  The remaining vector slots stay unused so existing
    symmetric allocations remain ABI-compatible.
    """
    peer_offsets = gl.arange(0, participant_capacity, layout=layout)
    valid_peer = peer_offsets < world_size
    safe_peer_offsets = gl.where(valid_peer, peer_offsets, 0)
    remote_barriers = gl.load(
        peer_barrier_ptrs + safe_peer_offsets,
        mask=valid_peer,
        other=0,
    ).to(gl.pointer_type(gl.int32))
    _red_i32_release_sys_if(
        remote_barriers,
        gl.full([participant_capacity], 1, gl.int32, layout=layout),
        valid_peer,
    )

    leader = peer_offsets == 0
    local_counter_ptrs = barrier + peer_offsets * 0
    ready = _load_i32_acquire_sys_if(
        local_counter_ptrs,
        leader,
    )
    target: gl.constexpr = expected * world_size
    pending_mask = leader & (ready < target)
    pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)
    while pending != 0:
        fresh = _load_i32_acquire_sys_if(
            local_counter_ptrs,
            pending_mask,
        )
        ready = gl.where(pending_mask, fresh, ready)
        pending_mask = leader & (ready < target)
        pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)


@gluon.jit
def _packed_expert_count_from_cache(stored_counts, count_offsets, expert):
    """Select one expert count from a lane-distributed register snapshot."""
    return gl.max(
        gl.where(count_offsets == expert, stored_counts, 0),
        axis=0,
    )


@gluon.jit
def _load_packed_expert_counts(
    expert_state,
    count_offsets,
    num_experts: gl.constexpr,
    world_size: gl.constexpr,
):
    """Acquire packed states once, reloading only entries that remain pending."""
    valid = count_offsets < num_experts
    safe_offsets = gl.where(valid, count_offsets, 0)
    packed = _load_packed_expert_state_acquire_if(
        expert_state + safe_offsets,
        valid,
    )
    contributors = packed // 4294967296
    pending_mask = valid & (contributors < world_size)
    pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)
    while pending != 0:
        fresh = _load_packed_expert_state_acquire_if(
            expert_state + safe_offsets,
            pending_mask,
        )
        packed = gl.where(pending_mask, fresh, packed)
        contributors = packed // 4294967296
        pending_mask = valid & (contributors < world_size)
        pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)
    return gl.where(
        valid,
        packed - contributors * 4294967296,
        0,
    ).to(gl.int32)


@gluon.jit
def _fused_pool_dispatch_partition(
    task_state,
    dispatch_state,
    symmetric_state,
    descs,
    barriers,
    buffers,
    expert_send_state,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    block_m: gl.constexpr,
    num_sms: gl.constexpr,
    num_experts: gl.constexpr,
    num_global_experts: gl.constexpr,
    num_routes: gl.constexpr,
    experts_per_rank: gl.constexpr,
    max_routes: gl.constexpr,
    rank: gl.constexpr,
    world_size: gl.constexpr,
    dispatch_worker: gl.constexpr,
    num_dispatch_workers: gl.constexpr,
):
    """Run one DeepGEMM-style independent dispatch-warp token stream."""
    dispatch_counter = task_state[2]
    (
        pool_acts,
        pool_acts_sf,
        pool_topk_weights,
        token_src_metadata,
        num_padded_sf_pool_tokens,
        input_topk_idx,
        actual_num_pool_rows,
    ) = dispatch_state
    (
        peer_input_sf_ptrs,
        peer_input_topk_weights_ptrs,
        symmetric_source_routes,
        symmetric_recv_count,
        peer_source_routes_ptrs,
        peer_recv_count_ptrs,
        peer_expert_state_ptrs,
        dispatch_barrier,
        peer_dispatch_barrier_ptrs,
    ) = symmetric_state
    # DeepGEMM assigns one token stream to each dispatch warp.  Model the
    # two streams as separate one-warp specialized partitions.
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    peer_acts_descs, pool_acts_desc = descs
    tma_load_barriers = barriers
    pull_buffer = buffers
    sf_groups: gl.constexpr = hidden // 128
    sf_per_lane: gl.constexpr = (sf_groups + 31) // 32
    sf_capacity: gl.constexpr = sf_per_lane * 32
    sf_layout: gl.constexpr = gl.BlockedLayout(
        [sf_per_lane],
        [32],
        [1],
        [0],
    )
    sf_offsets = gl.arange(0, sf_capacity, layout=sf_layout)
    dispatch_pid = (
        gl.program_id(0) * num_dispatch_workers + dispatch_worker
    )
    num_global_dispatch_workers: gl.constexpr = (
        num_sms * num_dispatch_workers
    )

    # DeepGEMM first reserves source-local per-expert slots, then uses
    # ordinary remote stores.  Gluon's shared-memory descriptor does
    # not expose block-scope atomics, so reserve directly in this
    # rank's local L2 rather than issuing one system-scope atomic per
    # route on the destination rank.  The old value is the unique
    # source-rank slot later consumed by the remote expert.
    route_offsets = gl.arange(0, 32, layout=layout)
    route_ones = gl.full([32], 1, gl.int64, layout=layout)
    route_base = dispatch_pid * 32
    while route_base < num_routes:
        route = route_base + route_offsets
        valid_route = route < num_routes
        routed_expert = gl.load(
            input_topk_idx + route,
            mask=valid_route,
            other=-1,
        )
        valid_route = valid_route & (routed_expert >= 0)
        safe_routed_expert = gl.where(valid_route, routed_expert, 0)
        target_rank = safe_routed_expert // experts_per_rank
        local_expert = (
            safe_routed_expert - target_rank * experts_per_rank
        )
        remote_routes = gl.load(
            peer_source_routes_ptrs + target_rank,
            mask=valid_route,
            other=0,
        ).to(gl.pointer_type(gl.int32))
        slot = gl.atomic_add(
            expert_send_state + safe_routed_expert,
            route_ones,
            mask=valid_route,
            sem="relaxed",
            scope="gpu",
        )
        gl.store(
            remote_routes
            + (rank * experts_per_rank + local_expert) * max_routes
            + slot,
            route,
            mask=valid_route,
        )
        route_base += num_global_dispatch_workers * 32

    gl.atomic_add(
        dispatch_counter,
        1,
        sem="release",
        scope="gpu",
    )
    local_routes_ready = _load_i32_acquire_gpu(dispatch_counter)
    while local_routes_ready < num_global_dispatch_workers:
        local_routes_ready = _load_i32_acquire_gpu(dispatch_counter)

    # Once every local worker has finished its remote route stores,
    # publish this rank's send count with one ordinary remote count
    # store and one system-release packed-state atomic per expert.
    # The acquire above plus this release forms the visibility chain
    # from every route store to the remote expert-state consumer.
    global_expert = dispatch_pid
    while global_expert < num_global_experts:
        target_rank = global_expert // experts_per_rank
        local_expert = global_expert - target_rank * experts_per_rank
        source_count = gl.load(
            expert_send_state + global_expert
        ).to(gl.int32)
        remote_counts = gl.load(peer_recv_count_ptrs + target_rank).to(
            gl.pointer_type(gl.int32)
        )
        gl.store(
            remote_counts + rank * experts_per_rank + local_expert,
            source_count,
        )
        remote_expert_state = gl.load(
            peer_expert_state_ptrs + target_rank
        ).to(gl.pointer_type(gl.int64))
        packed_contribution = source_count.to(gl.int64) + 4294967296
        gl.atomic_add(
            remote_expert_state + local_expert,
            packed_contribution,
            sem="release",
            scope="sys",
        )
        global_expert += num_global_dispatch_workers

    # The peer signal is the publication-complete notification.  It
    # must not race a late worker's packed state atomic.
    gl.atomic_add(
        dispatch_counter,
        1,
        sem="release",
        scope="gpu",
    )
    if dispatch_worker == 0 and gl.program_id(0) == 0:
        local_publication_ready = _load_i32_acquire_gpu(
            dispatch_counter
        )
        while local_publication_ready < 2 * num_global_dispatch_workers:
            local_publication_ready = _load_i32_acquire_gpu(
                dispatch_counter
            )
        _peer_barrier_arrive_and_wait(
            peer_dispatch_barrier_ptrs,
            dispatch_barrier,
            world_size,
            1,
            layout,
            32,
        )

    # Snapshot every expert count once per dispatch warp.  The one-warp
    # layout lets lanes fetch experts in parallel; later token traversal
    # selects counts from registers instead of issuing a sequential chain of
    # system-scope loads for every worker.
    dispatch_count_capacity: gl.constexpr = triton.next_power_of_2(
        num_experts
    )
    dispatch_counts_per_lane: gl.constexpr = max(
        dispatch_count_capacity // 32,
        1,
    )
    dispatch_count_layout: gl.constexpr = gl.BlockedLayout(
        [dispatch_counts_per_lane],
        [32],
        [1],
        [0],
    )
    dispatch_count_offsets = gl.arange(
        0,
        dispatch_count_capacity,
        layout=dispatch_count_layout,
    )
    dispatch_stored_counts = _load_packed_expert_counts(
        task_state[0],
        dispatch_count_offsets,
        num_experts,
        world_size,
    )

    if dispatch_worker == 0 and gl.program_id(0) == 0:
        num_pool_blocks = gl.sum(
            gl.where(
                dispatch_count_offsets < num_experts,
                (dispatch_stored_counts + block_m - 1) // block_m,
                0,
            ),
            axis=0,
        )
        gl.store(actual_num_pool_rows, num_pool_blocks * block_m)

    sf_block_m: gl.constexpr = (block_m + 127) // 128 * 128

    # DeepGEMM stripes only valid routed tokens over the persistent grid.
    # Pool storage remains expert/block padded for TMA, but padding rows do
    # not need to be materialized: GEMM may read stale padding because all
    # observable epilogues are guarded by valid_m.  Besides removing the
    # useless input/SF traffic, publishing only valid arrivals lets the A
    # loader start a partial expert block as soon as its real tokens land.
    dispatch_token = dispatch_pid
    current_expert = 0
    expert_token_begin = 0
    expert_token_end = 0
    expert_pool_block = 0
    current_expert_count = _packed_expert_count_from_cache(
        dispatch_stored_counts,
        dispatch_count_offsets,
        0,
    )
    expert_token_end = current_expert_count
    while dispatch_token >= expert_token_end and current_expert < num_experts:
        expert_pool_block += (
            current_expert_count + block_m - 1
        ) // block_m
        current_expert += 1
        expert_token_begin = expert_token_end
        if current_expert < num_experts:
            current_expert_count = _packed_expert_count_from_cache(
                dispatch_stored_counts,
                dispatch_count_offsets,
                current_expert,
            )
        else:
            current_expert_count = 0
        expert_token_end += current_expert_count

    while current_expert < num_experts:
        local_row = dispatch_token - expert_token_begin
        pool_row = expert_pool_block * block_m + local_row
        pool_block = pool_row // block_m

        selected_rank = 0
        slot = local_row
        round_offset = 0
        token_idx_in_rank = 0
        found_route = False
        while not found_route:
            num_active = 0
            round_length = 0x7fffffff
            for source_rank in gl.static_range(world_size):
                rank_count = gl.load(
                    symmetric_recv_count
                    + source_rank * experts_per_rank
                    + current_expert
                )
                remaining = gl.maximum(rank_count - round_offset, 0)
                is_active = remaining > 0
                num_active += is_active.to(gl.int32)
                round_length = gl.where(
                    is_active,
                    gl.minimum(round_length, remaining),
                    round_length,
                )
            round_tokens = round_length * num_active
            if slot < round_tokens:
                desired_rank = slot - (slot // num_active) * num_active
                num_seen = 0
                for source_rank in gl.static_range(world_size):
                    rank_count = gl.load(
                        symmetric_recv_count
                        + source_rank * experts_per_rank
                        + current_expert
                    )
                    is_active = rank_count > round_offset
                    choose = is_active and num_seen == desired_rank
                    selected_rank = gl.where(
                        choose,
                        source_rank,
                        selected_rank,
                    )
                    num_seen += is_active.to(gl.int32)
                token_idx_in_rank = round_offset + slot // num_active
                found_route = True
            else:
                slot -= round_tokens
                round_offset += round_length
        route = gl.load(
            symmetric_source_routes
            + (selected_rank * experts_per_rank + current_expert)
            * max_routes
            + token_idx_in_rank,
        )
        remote_sf = gl.load(peer_input_sf_ptrs + selected_rank).to(
            gl.pointer_type(gl.float32)
        )
        remote_weights = gl.load(
            peer_input_topk_weights_ptrs + selected_rank
        ).to(gl.pointer_type(gl.float32))
        source_token = route // topk
        source_slot = route - source_token * topk
        tma_phase = (
            dispatch_token // num_global_dispatch_workers
        ) & 1
        mbarrier.expect(
            tma_load_barriers,
            pool_acts_desc.block_type.nbytes,
        )
        for source_rank in gl.static_range(world_size):
            if selected_rank == source_rank:
                tma.async_copy_global_to_shared(
                    peer_acts_descs[source_rank],
                    [source_token, 0, 0],
                    tma_load_barriers,
                    pull_buffer,
                )
        pool_block = pool_row // block_m
        row_in_block = pool_row - pool_block * block_m
        sf_pool_row = pool_block * sf_block_m + row_in_block
        valid_sf = sf_offsets < sf_groups
        scales = gl.load(
            remote_sf + source_token * sf_groups + sf_offsets,
            mask=valid_sf,
            other=0.0,
        )
        gl.store(
            pool_acts_sf
            + sf_offsets * num_padded_sf_pool_tokens
            + sf_pool_row,
            scales,
            mask=valid_sf,
        )
        weight = gl.load(
            remote_weights + route,
        )
        gl.store(pool_topk_weights + pool_row, weight)
        mbarrier.wait(tma_load_barriers, tma_phase)
        tma.async_copy_shared_to_global(
            pool_acts_desc,
            [pool_row, 0, 0],
            pull_buffer,
        )
        tma.store_wait(0)
        gl.store(
            token_src_metadata + pool_row * 3,
            selected_rank,
        )
        gl.store(
            token_src_metadata + pool_row * 3 + 1,
            source_token,
        )
        gl.store(
            token_src_metadata + pool_row * 3 + 2,
            source_slot,
        )
        gl.atomic_add(
            task_state[1] + pool_block,
            1,
            sem="release",
            scope="gpu",
        )
        dispatch_token += num_global_dispatch_workers
        while (
            dispatch_token >= expert_token_end
            and current_expert < num_experts
        ):
            expert_pool_block += (
                current_expert_count + block_m - 1
            ) // block_m
            current_expert += 1
            expert_token_begin = expert_token_end
            if current_expert < num_experts:
                current_expert_count = _packed_expert_count_from_cache(
                    dispatch_stored_counts,
                    dispatch_count_offsets,
                    current_expert,
                )
            else:
                current_expert_count = 0
            expert_token_end += current_expert_count


def create_sm90_mega_moe_symmetric_context(
    *,
    max_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
    group_name: str | None = None,
) -> SM90MegaMoESymmetricContext:
    """Collectively create the single-node peer buffers for MegaMoE."""
    import torch.distributed as dist

    _require_sm90_and_gluon()
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before rendezvous")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    return SM90MegaMoESymmetricContext(
        max_tokens=max_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
        world_size=world_size,
        rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
        group_name=group_name,
    )


def _require_sm90_and_gluon() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the Gluon MegaMoE capability gate requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise RuntimeError(f"the capability gate requires SM90, got sm{major}{minor}")

    try:
        installed_version = Version(triton.__version__.split("+", 1)[0])
    except InvalidVersion as exc:
        raise RuntimeError(
            f"cannot parse the installed Triton version {triton.__version__!r}"
        ) from exc
    if installed_version < _MINIMUM_TRITON_VERSION:
        raise RuntimeError(
            "the Gluon MegaMoE path requires Triton >= "
            f"{_MINIMUM_TRITON_VERSION}, got {triton.__version__}"
        )


_GLUON_BLOCK_M_VALUE = 64
_GLUON_BLOCK_N_VALUE = 128
_GLUON_BLOCK_K_VALUE = 128
_GLUON_SF_BLOCK_M_VALUE = 128
_GLUON_GROUP_M_VALUE = 8
# Match DeepGEMM's scale-domain WGMMA accumulation boundaries.  FC1 chains the
# four native K32 instructions inside one K128 scale block, while FC2 chains
# two native K32 instructions inside each independently scaled K64 half.  The
# scaled K128/K64 partials are still accumulated into ``final`` in FP32 across
# scale blocks.
# Seven 24-KiB A/B stages consume about 168 KiB per persistent CTA.  On SM90
# this both deepens the pipeline and prevents two such CTAs from being resident
# on one SM, which is required before adding grid-coupled dispatch warps.
_GLUON_1D2D_NUM_STAGES = 7
# Sparse/decode tiles use two WGMMA warpgroups inside one eight-warp math
# partition.  The selected distributed layout decides whether they split the
# normal accumulator along N or the swapAB accumulator along its channel axis.
_GLUON_1D2D_MATH_WARP_GROUPS = 2
_GLUON_1D2D_PRODUCERS = 2
_GLUON_1D2D_MATH_WARPS_PER_GROUP = 4
_GLUON_1D2D_MATH_WARPS = (
    _GLUON_1D2D_MATH_WARP_GROUPS
    * _GLUON_1D2D_MATH_WARPS_PER_GROUP
)
_GLUON_TMA_REGS_VALUE = 24
# Match DeepGEMM's 256-thread decode math/epilogue role.  Gluon's launch
# ``maxnreg`` is a CTA-wide average budget rather than the final per-thread
# budget of the default math partition.  The fused kernel's A/B/dispatch
# workers share one four-warp group capped at 48 registers/thread, so its
# corresponding launch average is 128:
#   floor8((128 * 8 + (128 - 48) * 4) / 8) = 168.
_GLUON_DECODE_MATH_REGS_VALUE = 168
_GLUON_FUSED_MAXNREG = 128

_GLUON_BLOCK_M = gl.constexpr(_GLUON_BLOCK_M_VALUE)
_GLUON_BLOCK_N = gl.constexpr(_GLUON_BLOCK_N_VALUE)
_GLUON_BLOCK_K = gl.constexpr(_GLUON_BLOCK_K_VALUE)
_GLUON_SF_BLOCK_M = gl.constexpr(_GLUON_SF_BLOCK_M_VALUE)
_GLUON_GROUP_M = gl.constexpr(_GLUON_GROUP_M_VALUE)
_GLUON_1D2D_PRODUCERS_CONSTEXPR = gl.constexpr(_GLUON_1D2D_PRODUCERS)
_GLUON_1D2D_MATH_WARPS_CONSTEXPR = gl.constexpr(_GLUON_1D2D_MATH_WARPS)
_GLUON_TMA_REGS = gl.constexpr(_GLUON_TMA_REGS_VALUE)


def run_sm90_mega_moe_pre_dispatch(
    ctx: SM90MegaMoESymmetricContext,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    x_sf: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> SM90MegaMoEPreDispatchResult:
    """Register dispatch inputs using the narrow DeepGEMM-style boundary.

    BF16 input is quantized to FP8 E4M3 with one FP32 scale per token/per-128
    group.  Already-quantized FP8 E4M3 input is copied together with its
    required FP32 ``x_sf`` tensor.  Both specializations copy top-k indices,
    multiply top-k weights by ``routed_scaling_factor``, and initialize padded
    top-k rows to ``-1``/zero.  Route counting, peer publication, prefix sums,
    pool materialization, and arrival signaling deliberately remain in the
    dispatch partition.
    """
    _require_sm90_and_gluon()
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional BF16 or FP8 E4M3 tensor")
    if x.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("x must have dtype bfloat16 or float8_e4m3fn")
    num_tokens, hidden = x.shape
    if ctx.max_tokens <= 0:
        raise ValueError("the symmetric context must reserve at least one token")
    if not 0 < ctx.topk <= _SM90_MEGA_MOE_PRE_DISPATCH_THREADS:
        raise ValueError("SM90 pre-dispatch requires topk in [1, 1024]")
    if num_tokens > ctx.max_tokens or hidden != ctx.hidden:
        raise ValueError("input exceeds the symmetric context capacity")
    if hidden <= 0:
        raise ValueError("SM90 pre-dispatch requires a positive hidden dimension")
    if hidden > (
        _SM90_MEGA_MOE_PRE_DISPATCH_GROUPS_PER_CTA
        * _SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE
    ):
        raise ValueError("SM90 pre-dispatch supports hidden dimensions up to 8192")
    if hidden % _SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE:
        raise ValueError("SM90 pre-dispatch requires hidden divisible by 128")
    if x.dtype == torch.bfloat16:
        if x_sf is not None:
            raise ValueError("x_sf must be None when pre-dispatch input is BF16")
    else:
        if x_sf is None:
            raise ValueError("x_sf is required when pre-dispatch input is FP8")
        if x_sf.shape != (num_tokens, hidden // 128):
            raise ValueError("x_sf must be [num_tokens, hidden // 128]")
        if x_sf.dtype != torch.float32:
            raise ValueError("x_sf must have dtype float32")
    if topk_idx.shape != (num_tokens, ctx.topk):
        raise ValueError("topk_idx does not match the symmetric context")
    if topk_weights.shape != topk_idx.shape:
        raise ValueError("topk_weights must match topk_idx")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_idx must have dtype int32 or int64")
    if topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must have dtype float32")
    tensors = (topk_idx, topk_weights)
    if x_sf is not None:
        tensors = (*tensors, x_sf)
    if x.device != ctx.device or any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all pre-dispatch tensors must use the context device")
    if not x.is_contiguous() or any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all pre-dispatch tensors must be contiguous")
    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")

    quant_layout = gl.BlockedLayout(
        [1, 8],
        [2, 16],
        [_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS, 1],
        [1, 0],
    )
    route_layout = gl.BlockedLayout(
        [1],
        [32],
        [_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS],
        [0],
    )
    num_padding_routes = (ctx.max_tokens - num_tokens) * ctx.topk
    grid = (
        num_tokens
        + triton.cdiv(
            num_padding_routes,
            _SM90_MEGA_MOE_PRE_DISPATCH_THREADS,
        ),
    )
    source_sf = ctx.input_sf if x_sf is None else x_sf
    compiled = _sm90_mega_moe_pre_dispatch_kernel[grid](
        x,
        source_sf,
        topk_idx,
        topk_weights,
        ctx.input_acts,
        ctx.input_sf,
        ctx.input_topk_idx,
        ctx.input_topk_weights,
        num_tokens,
        ctx.max_tokens,
        hidden,
        hidden // _SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE,
        ctx.topk,
        routed_scaling_factor,
        x.dtype == torch.bfloat16,
        quant_layout,
        route_layout,
        num_warps=_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS,
    )
    return SM90MegaMoEPreDispatchResult(
        input_acts_fp8=ctx.input_acts,
        input_acts_sf=ctx.input_sf,
        input_topk_idx=ctx.input_topk_idx,
        input_topk_weights=ctx.input_topk_weights,
        num_tokens=num_tokens,
        hidden=hidden,
        source_dtype=x.dtype,
        routed_scaling_factor=routed_scaling_factor,
        compiled=compiled,
    )


def _validate_sm90_mega_moe_pre_dispatch_result(
    ctx: SM90MegaMoESymmetricContext,
    result: SM90MegaMoEPreDispatchResult,
    *,
    num_tokens: int,
    hidden: int,
    source_dtype: torch.dtype,
    routed_scaling_factor: float,
) -> None:
    if not isinstance(result, SM90MegaMoEPreDispatchResult):
        raise TypeError("pre_dispatch_result must be a Gluon pre-dispatch result")
    if result.num_tokens != num_tokens or result.hidden != hidden:
        raise ValueError("pre_dispatch_result does not match this input shape")
    if result.source_dtype != source_dtype:
        raise ValueError("pre_dispatch_result does not match this input dtype")
    if result.routed_scaling_factor != routed_scaling_factor:
        raise ValueError(
            "pre_dispatch_result used a different routed_scaling_factor"
        )
    expected = (
        ctx.input_acts,
        ctx.input_sf,
        ctx.input_topk_idx,
        ctx.input_topk_weights,
    )
    actual = (
        result.input_acts_fp8,
        result.input_acts_sf,
        result.input_topk_idx,
        result.input_topk_weights,
    )
    if any(lhs.data_ptr() != rhs.data_ptr() for lhs, rhs in zip(actual, expected)):
        raise ValueError("pre_dispatch_result belongs to a different context")


def _host_cdiv(a, b):
    return (a + b - 1) // b


def _host_next_power_of_2(value):
    return 1 << max(value - 1, 0).bit_length()


@gluon.jit
def _groupgemm_scheduler_get_count(stored_counts, count_offsets, expert):
    """Select one cached expert count from a distributed register tensor."""
    return _packed_expert_count_from_cache(
        stored_counts,
        count_offsets,
        expert,
    )


@gluon.jit
def _groupgemm_phase_pool_block_offset(
    stored_counts,
    count_offsets,
    expert,
):
    """Return the BLOCK_M-padded pool prefix for ``expert``.

    This is the Gluon form of DeepGEMM's scheduler-local prefix reduction.
    It operates only on the cached count tensor and never reloads a host-built
    expert-offset table.
    """
    blocks = (
        stored_counts + _GLUON_BLOCK_M - 1
    ) // _GLUON_BLOCK_M
    return gl.sum(
        gl.where(count_offsets < expert, blocks, 0),
        axis=0,
    )


# The two-linear scheduler, producers, math loop, and FC1 epilogue below are
# maintained from DeepGEMM's SM90 contracts in
# deep_gemm/impls/sm90_fp8_mega_moe.cuh and scheduler/mega_moe.cuh.  They are
# not adapters around another Python MegaMoE implementation.
@gluon.jit
def _groupgemm_phase_scheduler_next(
    stored_counts,
    count_offsets,
    block_idx,
    expert,
    phase,
    current_count,
    pool_block_offset,
    num_experts: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
):
    """Assign one FC1/FC2 tile using DeepGEMM's expert-wave state machine.

    Phase 1 walks every FC1 tile in the current expert wave.  The same
    persistent CTA cursor is then rewound to the wave beginning for phase 2,
    so FC2 can consume any pool block whose FC1 publication counter is ready.
    All A, B, and math partitions call this helper with identical state.
    """
    found = False
    task_phase = 0
    task_expert = 0
    task_m_block = 0
    task_n_block = 0
    task_pool_block = 0
    task_valid_count = 0

    while expert < num_experts and not found:
        wave_end = gl.minimum(
            (
                (expert + 1 + num_experts_per_wave - 1)
                // num_experts_per_wave
            ) * num_experts_per_wave,
            num_experts,
        )
        if phase == 1:
            while expert < wave_end and not found:
                num_m_blocks = (
                    current_count + _GLUON_BLOCK_M - 1
                ) // _GLUON_BLOCK_M
                m_block = block_idx // l1_n_blocks
                if m_block < num_m_blocks:
                    task_phase = 1
                    task_expert = expert
                    task_m_block = m_block
                    task_n_block = block_idx - m_block * l1_n_blocks
                    task_pool_block = pool_block_offset + m_block
                    task_valid_count = current_count
                    block_idx += num_sms
                    found = True
                else:
                    block_idx -= num_m_blocks * l1_n_blocks
                    pool_block_offset += num_m_blocks
                    expert += 1
                    current_count = _groupgemm_scheduler_get_count(
                        stored_counts,
                        count_offsets,
                        expert,
                    )
            if not found:
                phase = 2
                expert = (
                    (expert - 1) // num_experts_per_wave
                ) * num_experts_per_wave
                current_count = _groupgemm_scheduler_get_count(
                    stored_counts,
                    count_offsets,
                    expert,
                )
                pool_block_offset = _groupgemm_phase_pool_block_offset(
                    stored_counts,
                    count_offsets,
                    expert,
                )
        else:
            while expert < wave_end and not found:
                num_m_blocks = (
                    current_count + _GLUON_BLOCK_M - 1
                ) // _GLUON_BLOCK_M
                expert_tasks = num_m_blocks * l2_n_blocks
                if block_idx < expert_tasks:
                    task_phase = 2
                    task_expert = expert
                    task_m_block = block_idx // l2_n_blocks
                    task_n_block = block_idx - task_m_block * l2_n_blocks
                    task_pool_block = pool_block_offset + task_m_block
                    task_valid_count = current_count
                    block_idx += num_sms
                    found = True
                else:
                    block_idx -= expert_tasks
                    pool_block_offset += num_m_blocks
                    expert += 1
                    current_count = _groupgemm_scheduler_get_count(
                        stored_counts,
                        count_offsets,
                        expert,
                    )
            if not found:
                phase = 1

    return (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        task_valid_count,
        block_idx,
        expert,
        phase,
        current_count,
        pool_block_offset,
    )


@gluon.jit
def _groupgemm_phase_a_sfa_tma_partition(
    l1_a_desc,
    l1_sfa_desc,
    l2_a_desc,
    l2_sfa_desc,
    expert_state,
    l1_arrival,
    l2_arrival,
    barriers,
    buffers,
    E: gl.constexpr,
    l1_k: gl.constexpr,
    l2_k: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    world_size: gl.constexpr,
):
    """Phase-aware A producer for FC1 per-128 and FC2 per-64 SFA."""
    stage_empty, stage_ready = barriers
    a_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_m: gl.constexpr = a_buffers.type.shape[1]
    block_k: gl.constexpr = a_buffers.type.shape[2]

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane],
        [32],
        [1],
        [0],
    )
    count_offsets = gl.arange(
        0,
        scheduler_count_capacity,
        layout=scheduler_layout,
    )
    stored_counts = _load_packed_expert_counts(
        expert_state,
        count_offsets,
        E,
        world_size,
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = _groupgemm_scheduler_get_count(
        stored_counts,
        count_offsets,
        scheduler_expert,
    )
    current_pool_block_offset = 0
    pipeline_tile = 0
    (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        valid_count,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
    ) = _groupgemm_phase_scheduler_next(
        stored_counts,
        count_offsets,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
        E,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
    )

    while task_phase != 0:
        local_row = task_m_block * block_m
        flat_row_start = task_pool_block * block_m
        scale_row_start = task_pool_block * _GLUON_SF_BLOCK_M
        if task_phase == 1:
            expected = gl.minimum(block_m, valid_count - local_row)
            ready = _load_i32_acquire_gpu(l1_arrival + task_pool_block)
            while ready < expected:
                ready = _load_i32_acquire_gpu(l1_arrival + task_pool_block)
        if task_phase == 2:
            ready = _load_i32_acquire_gpu(l2_arrival + task_pool_block)
            while ready < l1_n_blocks:
                ready = _load_i32_acquire_gpu(l2_arrival + task_pool_block)

        num_k_tiles = gl.where(
            task_phase == 1,
            l1_k // block_k,
            l2_k // block_k,
        )
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_empty.index(stage), pipe_phase ^ 1)

            if task_phase == 1:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l1_a_desc.block_type.nbytes
                    + l1_sfa_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l1_a_desc,
                    [flat_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    a_buffers.index(stage),
                )
                tma.async_copy_global_to_shared(
                    l1_sfa_desc,
                    [
                        k_tile * num_padded_sf_pool_tokens
                        + scale_row_start
                    ],
                    stage_ready.index(stage),
                    sfa_lo_buffers.index(stage),
                )
            else:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l2_a_desc.block_type.nbytes
                    + 2 * l2_sfa_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l2_a_desc,
                    [flat_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    a_buffers.index(stage),
                )
                tma.async_copy_global_to_shared(
                    l2_sfa_desc,
                    [
                        (k_tile * 2) * num_padded_sf_pool_tokens
                        + scale_row_start
                    ],
                    stage_ready.index(stage),
                    sfa_lo_buffers.index(stage),
                )
                tma.async_copy_global_to_shared(
                    l2_sfa_desc,
                    [
                        (k_tile * 2 + 1) * num_padded_sf_pool_tokens
                        + scale_row_start
                    ],
                    stage_ready.index(stage),
                    sfa_hi_buffers.index(stage),
                )
            pipeline_tile += 1
            k_tile += 1

        (
            task_phase,
            task_expert,
            task_m_block,
            task_n_block,
            task_pool_block,
            valid_count,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
        ) = _groupgemm_phase_scheduler_next(
            stored_counts,
            count_offsets,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
            E,
            l1_n_blocks,
            l2_n_blocks,
            num_experts_per_wave,
            num_sms,
        )


@gluon.jit
def _groupgemm_phase_b_tma_partition(
    l1_b_desc,
    l2_b_desc,
    expert_state,
    barriers,
    b_buffers,
    E: gl.constexpr,
    l1_n: gl.constexpr,
    l1_k: gl.constexpr,
    l2_n: gl.constexpr,
    l2_k: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    world_size: gl.constexpr,
):
    """Phase-aware B producer selecting the FC1 or FC2 descriptor."""
    stage_empty, stage_ready = barriers
    num_stages: gl.constexpr = b_buffers.type.shape[0]
    block_n: gl.constexpr = b_buffers.type.shape[1]
    block_k: gl.constexpr = b_buffers.type.shape[2]

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane],
        [32],
        [1],
        [0],
    )
    count_offsets = gl.arange(
        0,
        scheduler_count_capacity,
        layout=scheduler_layout,
    )
    stored_counts = _load_packed_expert_counts(
        expert_state,
        count_offsets,
        E,
        world_size,
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = _groupgemm_scheduler_get_count(
        stored_counts,
        count_offsets,
        scheduler_expert,
    )
    current_pool_block_offset = 0
    pipeline_tile = 0
    (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        valid_count,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
    ) = _groupgemm_phase_scheduler_next(
        stored_counts,
        count_offsets,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
        E,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
    )

    while task_phase != 0:
        phase_n = gl.where(task_phase == 1, l1_n, l2_n)
        phase_k = gl.where(task_phase == 1, l1_k, l2_k)
        flat_b_row_start = (
            task_expert * phase_n + task_n_block * block_n
        )
        num_k_tiles = phase_k // block_k
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_empty.index(stage), pipe_phase ^ 1)
            if task_phase == 1:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l1_b_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l1_b_desc,
                    [flat_b_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
                )
            else:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l2_b_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l2_b_desc,
                    [flat_b_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
                )
            pipeline_tile += 1
            k_tile += 1

        (
            task_phase,
            task_expert,
            task_m_block,
            task_n_block,
            task_pool_block,
            valid_count,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
        ) = _groupgemm_phase_scheduler_next(
            stored_counts,
            count_offsets,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
            E,
            l1_n_blocks,
            l2_n_blocks,
            num_experts_per_wave,
            num_sms,
        )


@gluon.jit
def _sm90_silu(value, fast_math: gl.constexpr):
    """SM90 SwiGLU sigmoid factor used by the Gluon FC1 epilogue."""
    exponent = gl.exp(-value)
    if fast_math:
        reciprocal = gl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [1.0 + exponent],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        reciprocal = gl.fdiv(1.0, 1.0 + exponent)
    return value * reciprocal


@gluon.jit
def _groupgemm_fc1_swap_mainloop(
    barriers,
    buffers,
    l1_weight_scales,
    task_expert,
    task_n_block,
    pipeline_tile,
    l1_k: gl.constexpr,
    intermediate_hidden: gl.constexpr,
    l1_ws_stride_e: gl.constexpr,
    l1_ws_stride_n: gl.constexpr,
    l1_ws_stride_k: gl.constexpr,
    n_swap: gl.constexpr,
):
    """FC1 swapAB mainloop with independent gate/up scale domains."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l1_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[8, 1],
        instr_shape=[16, n_swap, 32],
    )
    channel_offsets = gl.arange(
        0,
        _GLUON_BLOCK_N,
        layout=gl.SliceLayout(1, swap_mma_layout),
    )
    use_up_scale = ((channel_offsets // 8) % 2) == 1
    final = gl.zeros(
        (_GLUON_BLOCK_N, n_swap),
        dtype=gl.float32,
        layout=swap_mma_layout,
    )
    gate_scale_block = task_n_block // 2
    up_scale_block = intermediate_hidden // 128 + gate_scale_block
    for k_tile in range(num_k_tiles):
        tile = pipeline_tile + k_tile
        stage = tile % num_stages
        phase = tile // num_stages & 1
        mbarrier.wait(stage_ready.index(stage), phase)

        partial = gl.zeros(
            (_GLUON_BLOCK_N, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial = warpgroup_mma(
            b_buffers.index(stage),
            a_buffers.index(stage).slice(0, n_swap, dim=0).permute((1, 0)),
            partial,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=128,
        )
        token_scale = sfa_lo_buffers.index(stage).slice(
            0,
            n_swap,
            dim=0,
        ).load(gl.SliceLayout(0, swap_mma_layout))
        gate_scale = gl.load(
            l1_weight_scales
            + task_expert.to(gl.int64) * l1_ws_stride_e
            + gate_scale_block * l1_ws_stride_n
            + k_tile * l1_ws_stride_k
        )
        up_scale = gl.load(
            l1_weight_scales
            + task_expert.to(gl.int64) * l1_ws_stride_e
            + up_scale_block * l1_ws_stride_n
            + k_tile * l1_ws_stride_k
        )
        channel_scale = gl.where(use_up_scale, up_scale, gate_scale)
        partial = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial, ),
        )
        final += partial * (
            channel_scale[:, None] * token_scale[None, :]
        )

        _gluon_partition_barrier()
        mbarrier.arrive(stage_empty.index(stage), count=1)

    return final


@gluon.jit
def _groupgemm_fc2_swap_mainloop(
    barriers,
    buffers,
    l2_weight_scales,
    task_expert,
    task_n_block,
    pipeline_tile,
    l2_k: gl.constexpr,
    l2_ws_stride_e: gl.constexpr,
    l2_ws_stride_n: gl.constexpr,
    l2_ws_stride_k: gl.constexpr,
    n_swap: gl.constexpr,
):
    """FC2 swapAB mainloop with independent low/high K64 A scales."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l2_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[8, 1],
        instr_shape=[16, n_swap, 32],
    )
    final = gl.zeros(
        (_GLUON_BLOCK_N, n_swap),
        dtype=gl.float32,
        layout=swap_mma_layout,
    )
    for k_tile in range(num_k_tiles):
        tile = pipeline_tile + k_tile
        stage = tile % num_stages
        phase = tile // num_stages & 1
        mbarrier.wait(stage_ready.index(stage), phase)

        channel_stage = b_buffers.index(stage)
        token_stage = a_buffers.index(stage).slice(0, n_swap, dim=0)
        l2_weight_scale = gl.load(
            l2_weight_scales
            + task_expert.to(gl.int64) * l2_ws_stride_e
            + task_n_block * l2_ws_stride_n
            + k_tile * l2_ws_stride_k
        )

        partial_lo = gl.zeros(
            (_GLUON_BLOCK_N, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial_lo = warpgroup_mma(
            channel_stage.slice(0, 64, dim=1),
            token_stage.slice(0, 64, dim=1).permute((1, 0)),
            partial_lo,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=64,
        )
        partial_lo = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial_lo, ),
        )
        token_scale_lo = sfa_lo_buffers.index(stage).slice(
            0,
            n_swap,
            dim=0,
        ).load(gl.SliceLayout(0, swap_mma_layout))
        final += partial_lo * (
            l2_weight_scale * token_scale_lo[None, :]
        )

        partial_hi = gl.zeros(
            (_GLUON_BLOCK_N, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial_hi = warpgroup_mma(
            channel_stage.slice(64, 64, dim=1),
            token_stage.slice(64, 64, dim=1).permute((1, 0)),
            partial_hi,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=64,
        )
        partial_hi = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial_hi, ),
        )
        token_scale_hi = sfa_hi_buffers.index(stage).slice(
            0,
            n_swap,
            dim=0,
        ).load(gl.SliceLayout(0, swap_mma_layout))
        final += partial_hi * (
            l2_weight_scale * token_scale_hi[None, :]
        )

        _gluon_partition_barrier()
        mbarrier.arrive(stage_empty.index(stage), count=1)

    return final


@gluon.jit
def _groupgemm_fc2_swap_bf16_epilogue(
    final,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    task_pool_block,
    task_n_block,
    valid_m,
    l2_n: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    n_swap: gl.constexpr,
):
    """Scatter fused swapAB output directly into source-rank slots."""
    swap_mma_layout: gl.constexpr = final.type.layout
    channel_offsets = gl.arange(
        0,
        _GLUON_BLOCK_N,
        layout=gl.SliceLayout(1, swap_mma_layout),
    )
    token_offsets = gl.arange(
        0,
        n_swap,
        layout=gl.SliceLayout(0, swap_mma_layout),
    )
    output_cols = task_n_block * _GLUON_BLOCK_N + channel_offsets
    output_rows = task_pool_block * _GLUON_BLOCK_M + token_offsets
    converted = final.to(gl.bfloat16)
    valid_tokens = token_offsets < valid_m
    source_rank = gl.load(
        token_src_metadata + output_rows * 3,
        mask=valid_tokens,
        other=-1,
    )
    source_token = gl.load(
        token_src_metadata + output_rows * 3 + 1,
        mask=valid_tokens,
        other=0,
    )
    source_slot = gl.load(
        token_src_metadata + output_rows * 3 + 2,
        mask=valid_tokens,
        other=0,
    )
    safe_source_rank = gl.maximum(
        gl.minimum(source_rank, world_size - 1),
        0,
    )
    remote_combine = gl.load(
        peer_combine_buffer_ptrs + safe_source_rank
    ).to(gl.pointer_type(gl.bfloat16))
    scatter_ptrs = (
        remote_combine[None, :]
        + (source_token[None, :] * topk + source_slot[None, :]) * l2_n
        + output_cols[:, None]
    )
    gl.store(
        scatter_ptrs,
        converted,
        mask=(
            valid_tokens[None, :]
            & (source_rank[None, :] >= 0)
            & (source_rank[None, :] < world_size)
            & (source_token[None, :] >= 0)
            & (source_token[None, :] < max_tokens)
            & (source_slot[None, :] >= 0)
            & (source_slot[None, :] < topk)
            & (output_cols[:, None] < l2_n)
        ),
    )


@gluon.jit
def _groupgemm_fc2_bf16_scatter_epilogue(
    output,
    row_offsets,
    col_offsets,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    task_pool_block,
    task_n_block,
    valid_m,
    l2_n: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
):
    """Vector-scatter a normal-layout FC2 BF16 tile to source ranks."""
    output_rows = task_pool_block * _GLUON_BLOCK_M + row_offsets[:, None]
    output_cols = task_n_block * _GLUON_BLOCK_N + col_offsets[None, :]
    row_mask = row_offsets[:, None] < valid_m
    source_rank = gl.load(
        token_src_metadata + output_rows * 3,
        mask=row_mask,
        other=-1,
    )
    source_token = gl.load(
        token_src_metadata + output_rows * 3 + 1,
        mask=row_mask,
        other=0,
    )
    source_slot = gl.load(
        token_src_metadata + output_rows * 3 + 2,
        mask=row_mask,
        other=0,
    )
    safe_source_rank = gl.maximum(
        gl.minimum(source_rank, world_size - 1),
        0,
    )
    remote_combine = gl.load(
        peer_combine_buffer_ptrs + safe_source_rank
    ).to(gl.pointer_type(gl.bfloat16))
    scatter_ptrs = (
        remote_combine
        + (source_token * topk + source_slot) * l2_n
        + output_cols
    )
    _store_contiguous_bf16_fragment(
        scatter_ptrs,
        output,
        (
            row_mask
            & (source_rank >= 0)
            & (source_rank < world_size)
            & (source_token >= 0)
            & (source_token < max_tokens)
            & (source_slot >= 0)
            & (source_slot < topk)
            & (output_cols < l2_n)
        ),
        4,
    )


@gluon.jit
def _groupgemm_fc2_combine_partition(
    output,
    combine_buffer,
    topk_idx,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    l2_n: gl.constexpr,
    num_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    num_sms: gl.constexpr,
):
    """Publish peer FC2 stores, then reduce source-local top-k slots."""
    # Every route row is striped across the complete eight-warp math
    # partition.  Match DeepGEMM's epilogue ``sync_scope`` before publishing
    # this CTA into the grid barrier, so the leader's release transitively
    # covers ordinary peer stores issued by every participating warp.
    _gluon_partition_barrier()
    gl.atomic_add(
        fc2_scatter_grid_counter,
        1,
        sem="release",
        scope="gpu",
    )
    grid_arrived = _load_i32_acquire_gpu(fc2_scatter_grid_counter)
    while grid_arrived < num_sms:
        grid_arrived = _load_i32_acquire_gpu(fc2_scatter_grid_counter)

    # The local grid counter only orders this rank's ordinary FC2 scatter
    # stores before its system-scope peer signal.  Dispatch cleanup waits for
    # combine_cross_rank_ready below instead, matching DeepGEMM's epilogue
    # rendezvous after cross-rank completion.

    if gl.program_id(0) == 0:
        barrier_participant_capacity: gl.constexpr = (
            _GLUON_1D2D_MATH_WARPS_CONSTEXPR * 32
        )
        barrier_layout: gl.constexpr = gl.BlockedLayout(
            [1],
            [32],
            [_GLUON_1D2D_MATH_WARPS_CONSTEXPR],
            [0],
        )
        _peer_barrier_arrive_and_wait(
            peer_fused_barrier_ptrs,
            fused_barrier,
            world_size,
            1,
            barrier_layout,
            barrier_participant_capacity,
        )
        gl.atomic_add(
            combine_cross_rank_ready,
            1,
            sem="release",
            scope="gpu",
        )

    cross_rank_ready = _load_i32_acquire_gpu(combine_cross_rank_ready)
    while cross_rank_ready < 1:
        cross_rank_ready = _load_i32_acquire_gpu(combine_cross_rank_ready)
    # This is the epilogue side of DeepGEMM's second grid sync: the local
    # leader has acquired every peer's system-scope completion chain, then
    # makes that visibility available to every combine warp before BF16 loads.
    _gluon_partition_barrier()

    # One math warp owns one token.  The N128 loop bounds the live accumulator
    # to four FP32 values per lane under the fused kernel's 128-register limit.
    combine_rows: gl.constexpr = _GLUON_1D2D_MATH_WARPS_CONSTEXPR
    combine_width: gl.constexpr = _GLUON_BLOCK_N // 32
    combine_layout: gl.constexpr = gl.BlockedLayout(
        [1, combine_width],
        [1, 32],
        [_GLUON_1D2D_MATH_WARPS_CONSTEXPR, 1],
        [1, 0],
    )
    combine_row_vector = gl.arange(
        0,
        combine_rows,
        layout=gl.SliceLayout(1, combine_layout),
    )
    combine_col_vector = gl.arange(
        0,
        _GLUON_BLOCK_N,
        layout=gl.SliceLayout(0, combine_layout),
    )
    num_combine_n_blocks: gl.constexpr = (
        l2_n + _GLUON_BLOCK_N - 1
    ) // _GLUON_BLOCK_N
    output_m_block = gl.program_id(0)
    num_output_m_blocks = (num_tokens + combine_rows - 1) // combine_rows
    while output_m_block < num_output_m_blocks:
        output_rows = (
            output_m_block * combine_rows + combine_row_vector[:, None]
        )
        output_row_mask = output_rows < num_tokens
        valid_slot_mask = output_rows * 0
        for slot in gl.static_range(topk):
            routed_expert = gl.load(
                topk_idx + output_rows * topk + slot,
                mask=output_row_mask,
                other=-1,
            )
            valid_slot_mask = valid_slot_mask | gl.where(
                routed_expert >= 0,
                1 << slot,
                0,
            )
        output_n_block = 0
        while output_n_block < num_combine_n_blocks:
            output_cols = (
                output_n_block * _GLUON_BLOCK_N
                + combine_col_vector[None, :]
            )
            mask = output_row_mask & (output_cols < l2_n)
            reduced = gl.zeros(
                (combine_rows, _GLUON_BLOCK_N),
                dtype=gl.float32,
                layout=combine_layout,
            )
            for slot in gl.static_range(topk):
                combine_ptrs = (
                    combine_buffer
                    + (output_rows * topk + slot) * l2_n
                    + output_cols
                )
                values = _load_contiguous_bf16_fragment(
                    combine_ptrs,
                    mask & ((valid_slot_mask & (1 << slot)) != 0),
                    combine_width,
                ).to(gl.float32)
                reduced += values
            output_ptrs = output + output_rows * l2_n + output_cols
            _store_contiguous_bf16_fragment(
                output_ptrs,
                reduced.to(gl.bfloat16),
                mask,
                combine_width,
            )
            output_n_block += 1
        output_m_block += num_sms


@gluon.jit
def _groupgemm_fc1_epilogue(
    final,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    route_weights,
    l2_arrival,
    task_pool_block,
    task_n_block,
    valid_m,
    num_padded_sf_pool_tokens: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    use_swap_ab: gl.constexpr,
    n_swap: gl.constexpr,
):
    """Publish one FC1 N128 tile as FP8 N64 plus per-row FP32 SF.

    The complete 8-warp partition owns one output-64 scale domain.  Reducing
    the reshaped tensor along that domain therefore includes both N-split
    warpgroups and produces exactly one publisher for this FC1 N block.
    """
    if use_swap_ab:
        paired = final.reshape(
            (_GLUON_BLOCK_N // 16, 2, 8, n_swap)
        ).permute((3, 0, 2, 1))
        gate, up = gl.split(paired)
        gate = gate.reshape((n_swap, _GLUON_BLOCK_N // 2))
        up = up.reshape((n_swap, _GLUON_BLOCK_N // 2))
        # ``final`` is [N, M] in swapAB, but ``gate`` has already been
        # permuted back to logical [M, N/2].  Derive the row vector from the
        # post-permute layout so ``weight[:, None]`` can restore axis 1.
        row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
        row_offsets = gl.arange(0, n_swap, layout=row_layout)
    else:
        paired = final.reshape(
            (_GLUON_BLOCK_M, _GLUON_BLOCK_N // 16, 2, 8)
        ).permute((0, 1, 3, 2))
        gate, up = gl.split(paired)
        gate = gate.reshape(
            (_GLUON_BLOCK_M, _GLUON_BLOCK_N // 2)
        )
        up = up.reshape(
            (_GLUON_BLOCK_M, _GLUON_BLOCK_N // 2)
        )
        row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
        row_offsets = gl.arange(
            0,
            _GLUON_BLOCK_M,
            layout=row_layout,
        )

    if has_activation_clamp:
        gate = gl.minimum(gate, activation_clamp)
        up = gl.minimum(gl.maximum(up, -activation_clamp), activation_clamp)
    swiglu = _sm90_silu(gate, fast_math) * up
    valid_rows = row_offsets < valid_m
    weight = gl.load(
        route_weights
        + task_pool_block * _GLUON_BLOCK_M
        + row_offsets,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    amax = gl.max(gl.abs(activation), axis=1)
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (
        activation * gl.fdiv(1.0, scale[:, None])
    ).to(gl.float8e4nv)

    # SF is MN-major: [I/64, pool_blocks * SF_BLOCK_M].  The block stride
    # intentionally uses 128 rather than BLOCK_M=64.
    sf_pool_rows = (
        task_pool_block * _GLUON_SF_BLOCK_M + row_offsets
    )
    gl.store(
        l2_acts_sf
        + task_n_block * num_padded_sf_pool_tokens
        + sf_pool_rows,
        scale,
        mask=valid_rows,
    )

    if use_swap_ab:
        l2_epilogue_buffer.slice(0, n_swap, dim=0).store(quantized)
    else:
        l2_epilogue_buffer.store(quantized)
    _gluon_partition_barrier()
    fence_async_shared()
    tma.async_copy_shared_to_global(
        l2_store_desc,
        [
            task_pool_block * _GLUON_BLOCK_M,
            task_n_block * (_GLUON_BLOCK_N // 2),
        ],
        l2_epilogue_buffer,
    )
    tma.store_wait(0)
    # No math warp may reuse the joint shared tile until its single combined
    # TMA store has drained.  The release counter orders both payload and SF.
    _gluon_partition_barrier()
    gl.atomic_add(
        l2_arrival + task_pool_block,
        1,
        sem="release",
        scope="gpu",
    )
    _gluon_partition_barrier()


@gluon.jit
def _groupgemm_phase_math_partition(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    output,
    combine_buffer,
    topk_idx,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    route_weights,
    l2_arrival,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    l1_n: gl.constexpr,
    l1_k: gl.constexpr,
    l2_n: gl.constexpr,
    l2_k: gl.constexpr,
    E: gl.constexpr,
    l1_ws_stride_e: gl.constexpr,
    l1_ws_stride_n: gl.constexpr,
    l1_ws_stride_k: gl.constexpr,
    l2_ws_stride_e: gl.constexpr,
    l2_ws_stride_n: gl.constexpr,
    l2_ws_stride_k: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    num_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    use_swap_ab: gl.constexpr,
):
    """Fused 8-warp math partition executing FC1, FC2, and combine."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane],
        [32],
        [_GLUON_1D2D_MATH_WARPS_CONSTEXPR],
        [0],
    )
    count_offsets = gl.arange(
        0,
        scheduler_count_capacity,
        layout=scheduler_layout,
    )
    stored_counts = _load_packed_expert_counts(
        expert_state,
        count_offsets,
        E,
        world_size,
    )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 2],
        instr_shape=[16, 64, 32],
    )
    mma_rows = gl.arange(
        0,
        _GLUON_BLOCK_M,
        layout=gl.SliceLayout(1, mma_layout),
    )
    mma_cols = gl.arange(
        0,
        _GLUON_BLOCK_N,
        layout=gl.SliceLayout(0, mma_layout),
    )
    l1_use_up_scale = ((mma_cols // 8) % 2) == 1
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [2, 16],
        [4, 2],
        [1, 0],
    )
    store_rows = gl.arange(
        0,
        _GLUON_BLOCK_M,
        layout=gl.SliceLayout(1, store_layout),
    )
    store_cols = gl.arange(
        0,
        _GLUON_BLOCK_N,
        layout=gl.SliceLayout(0, store_layout),
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = _groupgemm_scheduler_get_count(
        stored_counts,
        count_offsets,
        scheduler_expert,
    )
    current_pool_block_offset = 0
    pipeline_tile = 0
    (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        valid_count,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
    ) = _groupgemm_phase_scheduler_next(
        stored_counts,
        count_offsets,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
        E,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
    )

    while task_phase != 0:
        local_row = task_m_block * _GLUON_BLOCK_M
        valid_m = gl.minimum(
            _GLUON_BLOCK_M,
            valid_count - local_row,
        )

        if task_phase == 1 and use_swap_ab:
            if valid_m <= 8:
                final_swap_8 = _groupgemm_fc1_swap_mainloop(
                    barriers, buffers, l1_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l1_k, l2_k,
                    l1_ws_stride_e, l1_ws_stride_n, l1_ws_stride_k,
                    8,
                )
                _groupgemm_fc1_epilogue(
                    final_swap_8, l2_store_desc, l2_epilogue_buffer,
                    l2_acts_sf, route_weights, l2_arrival,
                    task_pool_block, task_n_block,
                    valid_m, num_padded_sf_pool_tokens,
                    activation_clamp, has_activation_clamp, fast_math,
                    True, 8,
                )
            elif valid_m <= 16:
                final_swap_16 = _groupgemm_fc1_swap_mainloop(
                    barriers, buffers, l1_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l1_k, l2_k,
                    l1_ws_stride_e, l1_ws_stride_n, l1_ws_stride_k,
                    16,
                )
                _groupgemm_fc1_epilogue(
                    final_swap_16, l2_store_desc, l2_epilogue_buffer,
                    l2_acts_sf, route_weights, l2_arrival,
                    task_pool_block, task_n_block,
                    valid_m, num_padded_sf_pool_tokens,
                    activation_clamp, has_activation_clamp, fast_math,
                    True, 16,
                )
            elif valid_m <= 32:
                final_swap_32 = _groupgemm_fc1_swap_mainloop(
                    barriers, buffers, l1_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l1_k, l2_k,
                    l1_ws_stride_e, l1_ws_stride_n, l1_ws_stride_k,
                    32,
                )
                _groupgemm_fc1_epilogue(
                    final_swap_32, l2_store_desc, l2_epilogue_buffer,
                    l2_acts_sf, route_weights, l2_arrival,
                    task_pool_block, task_n_block,
                    valid_m, num_padded_sf_pool_tokens,
                    activation_clamp, has_activation_clamp, fast_math,
                    True, 32,
                )
            else:
                final_swap_64 = _groupgemm_fc1_swap_mainloop(
                    barriers, buffers, l1_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l1_k, l2_k,
                    l1_ws_stride_e, l1_ws_stride_n, l1_ws_stride_k,
                    64,
                )
                _groupgemm_fc1_epilogue(
                    final_swap_64, l2_store_desc, l2_epilogue_buffer,
                    l2_acts_sf, route_weights, l2_arrival,
                    task_pool_block, task_n_block,
                    valid_m, num_padded_sf_pool_tokens,
                    activation_clamp, has_activation_clamp, fast_math,
                    True, 64,
                )
            pipeline_tile += l1_k // block_k
        elif task_phase == 2 and use_swap_ab:
            # Gluon register tensors require power-of-two element counts, so
            # both linear phases use the same 8/16/32/64 token buckets.
            if valid_m <= 8:
                final_swap_8 = _groupgemm_fc2_swap_mainloop(
                    barriers, buffers, l2_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l2_k,
                    l2_ws_stride_e, l2_ws_stride_n, l2_ws_stride_k,
                    8,
                )
                _groupgemm_fc2_swap_bf16_epilogue(
                    final_swap_8, token_src_metadata,
                    peer_combine_buffer_ptrs, task_pool_block, task_n_block,
                    valid_m, l2_n, max_tokens, topk, world_size, 8,
                )
            elif valid_m <= 16:
                final_swap_16 = _groupgemm_fc2_swap_mainloop(
                    barriers, buffers, l2_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l2_k,
                    l2_ws_stride_e, l2_ws_stride_n, l2_ws_stride_k,
                    16,
                )
                _groupgemm_fc2_swap_bf16_epilogue(
                    final_swap_16, token_src_metadata,
                    peer_combine_buffer_ptrs, task_pool_block, task_n_block,
                    valid_m, l2_n, max_tokens, topk, world_size, 16,
                )
            elif valid_m <= 32:
                final_swap_32 = _groupgemm_fc2_swap_mainloop(
                    barriers, buffers, l2_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l2_k,
                    l2_ws_stride_e, l2_ws_stride_n, l2_ws_stride_k,
                    32,
                )
                _groupgemm_fc2_swap_bf16_epilogue(
                    final_swap_32, token_src_metadata,
                    peer_combine_buffer_ptrs, task_pool_block, task_n_block,
                    valid_m, l2_n, max_tokens, topk, world_size, 32,
                )
            else:
                final_swap_64 = _groupgemm_fc2_swap_mainloop(
                    barriers, buffers, l2_weight_scales, task_expert,
                    task_n_block, pipeline_tile, l2_k,
                    l2_ws_stride_e, l2_ws_stride_n, l2_ws_stride_k,
                    64,
                )
                _groupgemm_fc2_swap_bf16_epilogue(
                    final_swap_64, token_src_metadata,
                    peer_combine_buffer_ptrs, task_pool_block, task_n_block,
                    valid_m, l2_n, max_tokens, topk, world_size, 64,
                )
            pipeline_tile += l2_k // block_k
        else:
            final = gl.zeros(
                (_GLUON_BLOCK_M, _GLUON_BLOCK_N),
                dtype=gl.float32,
                layout=mma_layout,
            )
            num_k_tiles = gl.where(
                task_phase == 1,
                l1_k // block_k,
                l2_k // block_k,
            )
            k_tile = 0
            while k_tile < num_k_tiles:
                stage = pipeline_tile % num_stages
                pipe_phase = pipeline_tile // num_stages & 1
                mbarrier.wait(stage_ready.index(stage), pipe_phase)
                b_stage = b_buffers.index(stage)

                if task_phase == 1:
                    partial = gl.zeros(
                        (_GLUON_BLOCK_M, _GLUON_BLOCK_N),
                        dtype=gl.float32,
                        layout=mma_layout,
                    )
                    partial = warpgroup_mma(
                        a_buffers.index(stage),
                        b_stage.permute((1, 0)),
                        partial,
                        is_async=True,
                        use_acc=False,
                        max_num_imprecise_acc=128,
                    )
                    a_scale = sfa_lo_buffers.index(stage).load(
                        gl.SliceLayout(1, mma_layout)
                    )
                    gate_scale_block = task_n_block // 2
                    up_scale_block = (
                        l2_k // 128 + gate_scale_block
                    )
                    gate_scale = gl.load(
                        l1_weight_scales
                        + task_expert.to(gl.int64) * l1_ws_stride_e
                        + gate_scale_block * l1_ws_stride_n
                        + k_tile * l1_ws_stride_k
                    )
                    up_scale = gl.load(
                        l1_weight_scales
                        + task_expert.to(gl.int64) * l1_ws_stride_e
                        + up_scale_block * l1_ws_stride_n
                        + k_tile * l1_ws_stride_k
                    )
                    column_scale = gl.where(
                        l1_use_up_scale,
                        up_scale,
                        gate_scale,
                    )
                    partial = warpgroup_mma_wait(
                        num_outstanding=0,
                        deps=(partial, ),
                    )
                    final += partial * (
                        a_scale[:, None] * column_scale[None, :]
                    )
                else:
                    l2_weight_scale = gl.load(
                        l2_weight_scales
                        + task_expert.to(gl.int64) * l2_ws_stride_e
                        + task_n_block * l2_ws_stride_n
                        + k_tile * l2_ws_stride_k
                    )
                    a_stage = a_buffers.index(stage)
                    partial_lo = gl.zeros(
                        (_GLUON_BLOCK_M, _GLUON_BLOCK_N),
                        dtype=gl.float32,
                        layout=mma_layout,
                    )
                    partial_lo = warpgroup_mma(
                        a_stage.slice(0, 64, dim=1),
                        b_stage.slice(0, 64, dim=1).permute((1, 0)),
                        partial_lo,
                        is_async=True,
                        use_acc=False,
                        max_num_imprecise_acc=64,
                    )
                    partial_lo = warpgroup_mma_wait(
                        num_outstanding=0,
                        deps=(partial_lo, ),
                    )
                    a_scale_lo = sfa_lo_buffers.index(stage).load(
                        gl.SliceLayout(1, mma_layout)
                    )
                    final += partial_lo * (
                        a_scale_lo[:, None] * l2_weight_scale
                    )

                    partial_hi = gl.zeros(
                        (_GLUON_BLOCK_M, _GLUON_BLOCK_N),
                        dtype=gl.float32,
                        layout=mma_layout,
                    )
                    partial_hi = warpgroup_mma(
                        a_stage.slice(64, 64, dim=1),
                        b_stage.slice(64, 64, dim=1).permute((1, 0)),
                        partial_hi,
                        is_async=True,
                        use_acc=False,
                        max_num_imprecise_acc=64,
                    )
                    partial_hi = warpgroup_mma_wait(
                        num_outstanding=0,
                        deps=(partial_hi, ),
                    )
                    a_scale_hi = sfa_hi_buffers.index(stage).load(
                        gl.SliceLayout(1, mma_layout)
                    )
                    final += partial_hi * (
                        a_scale_hi[:, None] * l2_weight_scale
                    )

                _gluon_partition_barrier()
                mbarrier.arrive(stage_empty.index(stage), count=1)
                pipeline_tile += 1
                k_tile += 1

            if task_phase == 1:
                _groupgemm_fc1_epilogue(
                    final, l2_store_desc, l2_epilogue_buffer, l2_acts_sf,
                    route_weights, l2_arrival, task_pool_block, task_n_block,
                    valid_m,
                    num_padded_sf_pool_tokens,
                    activation_clamp, has_activation_clamp, fast_math,
                    False, _GLUON_BLOCK_M,
                )
            else:
                fc2_tile = gl.convert_layout(final.to(gl.bfloat16), store_layout)
                _groupgemm_fc2_bf16_scatter_epilogue(
                    fc2_tile, store_rows, store_cols, token_src_metadata,
                    peer_combine_buffer_ptrs, task_pool_block, task_n_block,
                    valid_m, l2_n, max_tokens, topk, world_size,
                )

        (
            task_phase,
            task_expert,
            task_m_block,
            task_n_block,
            task_pool_block,
            valid_count,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
        ) = _groupgemm_phase_scheduler_next(
            stored_counts,
            count_offsets,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
            E,
            l1_n_blocks,
            l2_n_blocks,
            num_experts_per_wave,
            num_sms,
        )

    _groupgemm_fc2_combine_partition(
        output,
        combine_buffer,
        topk_idx,
        fused_barrier,
        peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        l2_n,
        num_tokens,
        topk,
        world_size,
        num_sms,
    )


@gluon.jit
def _sm90_fused_dispatch_1d2d_compact_3d_oob_tma_kernel(
    pool_acts,
    l1_a_desc,
    l1_sfa_desc,
    l1_b_desc,
    l2_store_desc,
    l2_a_desc,
    l2_sfa_desc,
    l2_b_desc,
    dispatch_acts_desc_0,
    dispatch_acts_desc_1,
    dispatch_acts_desc_2,
    dispatch_acts_desc_3,
    dispatch_acts_desc_4,
    dispatch_acts_desc_5,
    dispatch_acts_desc_6,
    dispatch_acts_desc_7,
    dispatch_pool_desc,
    pool_acts_sf,
    l2_acts_sf,
    output,
    combine_buffer,
    peer_combine_buffer_ptrs,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    expert_send_state,
    l1_arrival,
    l2_arrival,
    actual_num_pool_rows,
    dispatch_counter,
    input_topk_idx,
    pool_topk_weights,
    token_src_metadata,
    peer_input_sf_ptrs,
    peer_input_topk_weights_ptrs,
    symmetric_source_routes,
    symmetric_recv_count,
    peer_source_routes_ptrs,
    peer_recv_count_ptrs,
    peer_expert_state_ptrs,
    dispatch_barrier,
    peer_dispatch_barrier_ptrs,
    l1_n: gl.constexpr,
    l1_k: gl.constexpr,
    l2_n: gl.constexpr,
    l2_k: gl.constexpr,
    E: gl.constexpr,
    l1_ws_stride_e: gl.constexpr,
    l1_ws_stride_n: gl.constexpr,
    l1_ws_stride_k: gl.constexpr,
    l2_ws_stride_e: gl.constexpr,
    l2_ws_stride_n: gl.constexpr,
    l2_ws_stride_k: gl.constexpr,
    num_stages: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    num_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    num_global_experts: gl.constexpr,
    num_routes: gl.constexpr,
    experts_per_rank: gl.constexpr,
    max_routes: gl.constexpr,
    rank: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    use_swap_ab: gl.constexpr,
    a_tma_regs: gl.constexpr,
    b_tma_regs: gl.constexpr,
    dispatch_regs: gl.constexpr,
):
    """One resident CTA per SM for dispatch, FC1 publication, and FC2."""
    a_buffers = gl.allocate_shared_memory(
        l1_a_desc.dtype,
        [num_stages] + l1_a_desc.block_type.shape,
        l1_a_desc.layout,
    )
    b_buffers = gl.allocate_shared_memory(
        l1_b_desc.dtype,
        [num_stages] + l1_b_desc.block_type.shape,
        l1_b_desc.layout,
    )
    sfa_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [_GLUON_BLOCK_M],
        gl.float32,
    )
    sfa_lo_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, _GLUON_BLOCK_M],
        sfa_layout,
    )
    sfa_hi_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, _GLUON_BLOCK_M],
        sfa_layout,
    )
    l2_epilogue_buffer = gl.allocate_shared_memory(
        l2_store_desc.dtype,
        l2_store_desc.block_type.shape,
        l2_store_desc.layout,
    )
    barrier_layout: gl.constexpr = mbarrier.MBarrierLayout()
    stage_empty = gl.allocate_shared_memory(
        gl.int64,
        [num_stages, 1],
        barrier_layout,
    )
    stage_ready = gl.allocate_shared_memory(
        gl.int64,
        [num_stages, 1],
        barrier_layout,
    )
    for stage in gl.static_range(num_stages):
        mbarrier.init(stage_empty.index(stage), count=1)
        mbarrier.init(
            stage_ready.index(stage),
            count=_GLUON_1D2D_PRODUCERS_CONSTEXPR,
        )

    barriers = (stage_empty, stage_ready)
    buffers = (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
    )
    task_state = (
        expert_state,
        l1_arrival,
        dispatch_counter,
    )
    dispatch_state = (
        pool_acts,
        pool_acts_sf,
        pool_topk_weights,
        token_src_metadata,
        num_padded_sf_pool_tokens,
        input_topk_idx,
        actual_num_pool_rows,
    )
    symmetric_state = (
        peer_input_sf_ptrs,
        peer_input_topk_weights_ptrs,
        symmetric_source_routes,
        symmetric_recv_count,
        peer_source_routes_ptrs,
        peer_recv_count_ptrs,
        peer_expert_state_ptrs,
        dispatch_barrier,
        peer_dispatch_barrier_ptrs,
    )
    dispatch_peer_descs = (
        dispatch_acts_desc_0,
        dispatch_acts_desc_1,
        dispatch_acts_desc_2,
        dispatch_acts_desc_3,
        dispatch_acts_desc_4,
        dispatch_acts_desc_5,
        dispatch_acts_desc_6,
        dispatch_acts_desc_7,
    )
    dispatch_descs = (dispatch_peer_descs, dispatch_pool_desc)
    dispatch_buffers_0 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_buffers_1 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_barriers_0 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    dispatch_barriers_1 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    mbarrier.init(dispatch_barriers_0, count=1)
    mbarrier.init(dispatch_barriers_1, count=1)

    gl.warp_specialize(
        [
            (
                _groupgemm_phase_math_partition,
                (
                    barriers, buffers, l2_store_desc, l2_epilogue_buffer,
                    l2_acts_sf, output, combine_buffer,
                    input_topk_idx, token_src_metadata,
                    peer_combine_buffer_ptrs, fused_barrier,
                    peer_fused_barrier_ptrs, fc2_scatter_grid_counter,
                    combine_cross_rank_ready, pool_topk_weights, l2_arrival,
                    l1_weight_scales, l2_weight_scales, expert_state,
                    l1_n, l1_k, l2_n, l2_k, E,
                    l1_ws_stride_e, l1_ws_stride_n, l1_ws_stride_k,
                    l2_ws_stride_e, l2_ws_stride_n, l2_ws_stride_k,
                    l1_n_blocks, l2_n_blocks,
                    num_experts_per_wave, num_sms,
                    scheduler_count_capacity, scheduler_counts_per_lane,
                    num_padded_sf_pool_tokens, num_tokens, max_tokens, topk,
                    world_size,
                    activation_clamp, has_activation_clamp, fast_math,
                    use_swap_ab,
                ),
            ),
            (
                _groupgemm_phase_a_sfa_tma_partition,
                (
                    l1_a_desc, l1_sfa_desc, l2_a_desc, l2_sfa_desc,
                    expert_state, l1_arrival, l2_arrival, barriers,
                    (a_buffers, sfa_lo_buffers, sfa_hi_buffers), E,
                    l1_k, l2_k, l1_n_blocks, l2_n_blocks,
                    num_experts_per_wave, num_sms,
                    scheduler_count_capacity, scheduler_counts_per_lane,
                    num_padded_sf_pool_tokens, world_size,
                ),
            ),
            (
                _groupgemm_phase_b_tma_partition,
                (
                    l1_b_desc, l2_b_desc, expert_state, barriers,
                    b_buffers, E, l1_n, l1_k, l2_n, l2_k,
                    l1_n_blocks, l2_n_blocks, num_experts_per_wave,
                    num_sms, scheduler_count_capacity,
                    scheduler_counts_per_lane, world_size,
                ),
            ),
            (
                _fused_pool_dispatch_partition,
                (
                    task_state, dispatch_state, symmetric_state,
                    dispatch_descs, dispatch_barriers_0,
                    dispatch_buffers_0, expert_send_state, l1_k, topk,
                    _GLUON_BLOCK_M, num_sms, E,
                    num_global_experts, num_routes, experts_per_rank,
                    max_routes, rank, world_size, 0, 2,
                ),
            ),
            (
                _fused_pool_dispatch_partition,
                (
                    task_state, dispatch_state, symmetric_state,
                    dispatch_descs, dispatch_barriers_1,
                    dispatch_buffers_1, expert_send_state, l1_k, topk,
                    _GLUON_BLOCK_M, num_sms, E,
                    num_global_experts, num_routes, experts_per_rank,
                    max_routes, rank, world_size, 1, 2,
                ),
            ),
        ],
        [1, 1, 1, 1],
        [a_tma_regs, b_tma_regs, dispatch_regs, dispatch_regs],
    )
    dispatch_buffers_0._keep_alive()
    dispatch_buffers_1._keep_alive()
    dispatch_barriers_0._keep_alive()
    dispatch_barriers_1._keep_alive()


def run_sm90_fused_dispatch_1d2d_compact_symmetric(
    ctx: SM90MegaMoESymmetricContext,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor | None,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_weight: torch.Tensor,
    local_b_scales: torch.Tensor,
    local_l2_weight: torch.Tensor,
    local_l2_b_scales: torch.Tensor,
    *,
    block_m: int = 64,
    group_n: int = 128,
    group_k: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    num_stages: int = _GLUON_1D2D_NUM_STAGES,
    maxnreg: int = _GLUON_FUSED_MAXNREG,
    dispatch_regs: int = 48,
    num_experts_per_wave: int | None = None,
    activation_clamp: float = math.inf,
    fast_math: bool = False,
    workspace: SM90MegaMoEFusedFC1Workspace | None = None,
    routed_scaling_factor: float = 1.0,
    pre_dispatch_result: SM90MegaMoEPreDispatchResult | None = None,
    use_swap_ab: bool = True,
) -> SM90MegaMoEFusedFC1Result:
    """Register inputs, then run fused dispatch, FC1 epilogue, and FC2.

    ``x_fp8`` retains its historical name for call-site compatibility, but it
    may now be either BF16 or FP8 E4M3.  BF16 requires ``x_sf=None`` and is
    quantized by :func:`run_sm90_mega_moe_pre_dispatch`; FP8 requires its
    FP32 per-token/per-128 scales.  A separately measured pre-dispatch result
    may be supplied to skip that registration launch.  The persistent kernel
    always consumes the registered context buffers and never republishes raw
    input tensors from its two dispatch warps.

    Each source rank release-adds its count and contributor bit into the
    destination's packed expert states.  A, B, and math acquire those states,
    cache the low-32 counts, and derive compact offsets locally.  Only the A
    producer then acquires each block's
    ``l1_arrival`` counter and publishes the matching A+SFA tiles through the
    shared stage barrier.  Math consumes that shared publication without a
    second per-block dispatch poll.  The two resident dispatch partitions use
    one 3D-OOB TMA load/store pair per activation token.

    A caller measuring repeated launches may pass ``result.workspace`` from a
    completed warmup.  Device buffers and their TMA descriptors are then
    reused.  The next call resets only compact control and arrival arrays in a
    single CTA; stale padded payload, scale, and metadata rows remain
    unobservable behind the valid-row masks.  This avoids both the former
    multi-MiB reset and a dispatch/combine tail-cleanup dependency.

    ``local_weight`` is the granularity-8 gate/up-interleaved FC1 tensor and
    ``local_b_scales`` keeps canonical gate-block then up-block ordering.
    ``local_l2_weight`` and its per-128 scale tensor extend the existing model
    ABI without changing the dispatch input ABI.  ``output`` is the complete
    token-major result after FC2 peer scatter and source-local top-k sum.

    ``use_swap_ab=True`` selects transposed N128-by-token-bucket ownership for
    both FC1 and FC2.  FC2 still splits every K128 tile into two K64 WGMMA
    groups so each half consumes its independent per-64 activation scale.
    """
    _require_sm90_and_gluon()
    if ctx.world_size <= 1:
        raise ValueError("fused symmetric dispatch requires world_size > 1")
    if x_fp8.ndim != 2 or x_fp8.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise ValueError("x_fp8 must be a two-dimensional BF16 or FP8 E4M3 tensor")
    num_tokens, K = x_fp8.shape
    if num_tokens > ctx.max_tokens or K != ctx.hidden:
        raise ValueError("input exceeds the symmetric context capacity")
    if x_fp8.dtype == torch.bfloat16:
        if x_sf is not None:
            raise ValueError("x_sf must be None when fused input is BF16")
    else:
        if x_sf is None or x_sf.shape != (num_tokens, K // group_k):
            raise ValueError("FP8 x_sf must be [num_tokens, K // group_k]")
        if x_sf.dtype != torch.float32:
            raise ValueError("x_sf must have dtype float32")
    if topk_idx.shape != (num_tokens, ctx.topk):
        raise ValueError("topk_idx does not match the symmetric context")
    if topk_weights.shape != topk_idx.shape:
        raise ValueError("topk_weights must match topk_idx")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_idx must have dtype int32 or int64")
    if topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must have dtype float32")
    if local_weight.ndim != 3:
        raise ValueError("local_weight must have shape [experts_per_rank, N, K]")
    E, N, weight_k = local_weight.shape
    if E != ctx.experts_per_rank:
        raise ValueError("local_weight must contain exactly this rank's experts")
    if weight_k != K:
        raise ValueError("dispatch input and local_weight K dimensions must match")
    if local_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("local_weight must have dtype float8_e4m3fn")
    if block_m != _GLUON_BLOCK_M_VALUE:
        raise ValueError(f"fused compact FC1 requires block_m={_GLUON_BLOCK_M_VALUE}")
    if group_n != _GLUON_BLOCK_N_VALUE or group_k != _GLUON_BLOCK_K_VALUE:
        raise ValueError("fused compact FC1 requires group_n=group_k=128")
    if K % group_k or N % 256:
        raise ValueError("H must be divisible by 128 and FC1 2I by 256")
    intermediate_hidden = N // 2
    if local_b_scales.shape != (E, N // group_n, K // group_k):
        raise ValueError("local_b_scales must be [E, N // 128, K // 128]")
    if local_b_scales.dtype != torch.float32:
        raise ValueError("local_b_scales must have dtype float32")
    if local_l2_weight.shape != (E, K, intermediate_hidden):
        raise ValueError("local_l2_weight must be [E, H, I]")
    if local_l2_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("local_l2_weight must have dtype float8_e4m3fn")
    if local_l2_b_scales.shape != (
        E,
        K // 128,
        intermediate_hidden // 128,
    ):
        raise ValueError("local_l2_b_scales must be [E, H/128, I/128]")
    if local_l2_b_scales.dtype != torch.float32:
        raise ValueError("local_l2_b_scales must have dtype float32")
    if output_dtype != torch.bfloat16:
        raise ValueError("fused compact FC1 currently supports bfloat16 output only")
    tensors = (
        topk_idx,
        topk_weights,
        local_weight,
        local_b_scales,
        local_l2_weight,
        local_l2_b_scales,
    )
    if x_sf is not None:
        tensors = (x_sf, *tensors)
    if any(tensor.device != x_fp8.device for tensor in tensors):
        raise ValueError("all fused FC1 tensors must be on the same device")
    if x_fp8.device != ctx.device:
        raise ValueError("fused FC1 inputs must use the symmetric context device")
    if any(
        not tensor.is_contiguous()
        for tensor in (x_fp8, *tensors)
    ):
        raise ValueError("all fused FC1 inputs and weights must be contiguous")
    if local_b_scales.stride(-1) != 1:
        raise ValueError("local_b_scales must be contiguous along K groups")
    if local_l2_b_scales.stride(-1) != 1:
        raise ValueError("local_l2_b_scales must be contiguous along K groups")
    if topk_idx.numel() == 0:
        raise ValueError("the dispatch contract requires at least one route slot")

    device_sms = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    if num_sms is None:
        num_sms = device_sms
    if not 0 < num_sms <= device_sms:
        raise ValueError(f"num_sms must be in [1, {device_sms}], got {num_sms}")
    if not 2 <= num_stages <= 8:
        raise ValueError(f"num_stages must be in [2, 8], got {num_stages}")
    if not 32 <= maxnreg <= 255:
        raise ValueError(f"maxnreg must be in [32, 255], got {maxnreg}")
    if not 32 <= dispatch_regs <= 255:
        raise ValueError("dispatch_regs must be in [32, 255]")
    if not isinstance(use_swap_ab, bool):
        raise ValueError("use_swap_ab must be a bool")
    if not isinstance(fast_math, bool):
        raise ValueError("fast_math must be a bool")
    if num_experts_per_wave is None:
        num_experts_per_wave = min(E, 2)
    if not 0 < num_experts_per_wave <= E:
        raise ValueError("num_experts_per_wave must be in [1, E]")
    activation_clamp = float(activation_clamp)
    if math.isnan(activation_clamp) or activation_clamp <= 0:
        raise ValueError("activation_clamp must be positive or infinity")

    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    if pre_dispatch_result is None:
        registered_inputs = run_sm90_mega_moe_pre_dispatch(
            ctx,
            x_fp8,
            topk_idx,
            topk_weights,
            x_sf=x_sf,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        _validate_sm90_mega_moe_pre_dispatch_result(
            ctx,
            pre_dispatch_result,
            num_tokens=num_tokens,
            hidden=K,
            source_dtype=x_fp8.dtype,
            routed_scaling_factor=routed_scaling_factor,
        )
        registered_inputs = pre_dispatch_result

    max_global_routes = ctx.world_size * ctx.max_routes
    max_pool_blocks = triton.cdiv(
        max_global_routes + E * (block_m - 1),
        block_m,
    )
    num_pool_rows = max_pool_blocks * block_m
    sf_block_m = triton.cdiv(block_m, 128) * 128
    num_padded_sf_pool_tokens = max_pool_blocks * sf_block_m
    device = x_fp8.device
    if workspace is None:
        pool = SM90MegaMoESingleRankPool(
            acts=torch.empty(
                (num_pool_rows, K),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            acts_sf_mn_major=torch.empty(
                (K // group_k, num_padded_sf_pool_tokens),
                dtype=torch.float32,
                device=device,
            ),
            topk_weights=torch.empty(
                num_pool_rows,
                dtype=torch.float32,
                device=device,
            ),
            token_src_metadata=torch.empty(
                (num_pool_rows, 3),
                dtype=torch.int64,
                device=device,
            ),
            expert_state=ctx.expert_state,
            source_routes=ctx.source_routes,
        )
        l2_acts = torch.empty(
            (num_pool_rows, intermediate_hidden),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        l2_acts_sf_mn_major = torch.empty(
            (intermediate_hidden // 64, num_padded_sf_pool_tokens),
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(
            (ctx.max_tokens, K),
            dtype=output_dtype,
            device=device,
        )
        # Arrival arrays are compact control state.  Recycle them in the
        # existing prologue reset rather than coupling dispatch and combine.
        l1_arrival = torch.zeros(
            max_pool_blocks,
            dtype=torch.int32,
            device=device,
        )
        l2_arrival = torch.zeros(
            max_pool_blocks,
            dtype=torch.int32,
            device=device,
        )
        fc2_scatter_grid_counter = torch.empty(
            (), dtype=torch.int32, device=device
        )
        combine_cross_rank_ready = torch.empty(
            (), dtype=torch.int32, device=device
        )
        actual_num_pool_rows = torch.empty((), dtype=torch.int32, device=device)
        dispatch_counter = torch.empty((), dtype=torch.int32, device=device)
        expert_send_state = torch.empty(
            ctx.num_experts,
            dtype=torch.int64,
            device=device,
        )

        a_layout = gl.NVMMASharedLayout.get_default_for(
            [_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_K_VALUE],
            gl.float8e4nv,
        )
        l1_a_desc = TensorDescriptor.from_tensor(
            pool.acts,
            [_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_K_VALUE],
            a_layout,
        )
        sfa_layout = gl.NVMMASharedLayout.get_default_for(
            [_GLUON_BLOCK_M_VALUE],
            gl.float32,
        )
        l1_sfa_desc = TensorDescriptor.from_tensor(
            pool.acts_sf_mn_major.view(-1),
            [_GLUON_BLOCK_M_VALUE],
            sfa_layout,
        )
        b_layout = gl.NVMMASharedLayout.get_default_for(
            [_GLUON_BLOCK_N_VALUE, _GLUON_BLOCK_K_VALUE],
            gl.float8e4nv,
        )
        l1_b_desc = TensorDescriptor.from_tensor(
            local_weight.view(E * N, K),
            [_GLUON_BLOCK_N_VALUE, _GLUON_BLOCK_K_VALUE],
            b_layout,
        )
        l2_store_layout = gl.NVMMASharedLayout.get_default_for(
            [_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_N_VALUE // 2],
            gl.float8e4nv,
        )
        l2_store_desc = TensorDescriptor.from_tensor(
            l2_acts,
            [_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_N_VALUE // 2],
            l2_store_layout,
        )
        l2_a_desc = TensorDescriptor.from_tensor(
            l2_acts,
            [_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_K_VALUE],
            a_layout,
        )
        l2_sfa_desc = TensorDescriptor.from_tensor(
            l2_acts_sf_mn_major.view(-1),
            [_GLUON_BLOCK_M_VALUE],
            sfa_layout,
        )
        l2_b_desc = TensorDescriptor.from_tensor(
            local_l2_weight.view(E * K, intermediate_hidden),
            [_GLUON_BLOCK_N_VALUE, _GLUON_BLOCK_K_VALUE],
            b_layout,
        )
        dispatch_descs = (
            create_sm90_symmetric_dispatch_tma_oob_3d_descriptors(
                ctx,
                pool.acts,
            )
        )
        workspace = SM90MegaMoEFusedFC1Workspace(
            pool=pool,
            l2_acts=l2_acts,
            l2_acts_sf_mn_major=l2_acts_sf_mn_major,
            output=output,
            l1_arrival=l1_arrival,
            l2_arrival=l2_arrival,
            fc2_scatter_grid_counter=fc2_scatter_grid_counter,
            combine_cross_rank_ready=combine_cross_rank_ready,
            actual_num_pool_rows=actual_num_pool_rows,
            dispatch_counter=dispatch_counter,
            expert_send_state=expert_send_state,
            l1_a_desc=l1_a_desc,
            l1_sfa_desc=l1_sfa_desc,
            l1_b_desc=l1_b_desc,
            l2_store_desc=l2_store_desc,
            l2_a_desc=l2_a_desc,
            l2_sfa_desc=l2_sfa_desc,
            l2_b_desc=l2_b_desc,
            dispatch_descs=dispatch_descs,
            l1_weight_data_ptr=local_weight.data_ptr(),
            l2_weight_data_ptr=local_l2_weight.data_ptr(),
            max_pool_blocks=max_pool_blocks,
            num_pool_rows=num_pool_rows,
            num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
        )
    else:
        expected_workspace_shapes = (
            workspace.pool.acts.shape == (num_pool_rows, K),
            workspace.pool.acts_sf_mn_major.shape
            == (K // group_k, num_padded_sf_pool_tokens),
            workspace.l2_acts.shape
            == (num_pool_rows, intermediate_hidden),
            workspace.l2_acts_sf_mn_major.shape
            == (intermediate_hidden // 64, num_padded_sf_pool_tokens),
            workspace.output.shape == (ctx.max_tokens, K),
            workspace.l1_arrival.shape == (max_pool_blocks,),
            workspace.l2_arrival.shape == (max_pool_blocks,),
            workspace.fc2_scatter_grid_counter.shape == (),
            workspace.combine_cross_rank_ready.shape == (),
            workspace.expert_send_state.shape == (ctx.num_experts,),
            workspace.num_pool_rows == num_pool_rows,
            workspace.max_pool_blocks == max_pool_blocks,
            workspace.num_padded_sf_pool_tokens == num_padded_sf_pool_tokens,
        )
        if not all(expected_workspace_shapes):
            raise ValueError("fused FC1/FC2 workspace does not match this problem shape")
        if workspace.pool.acts.device != device:
            raise ValueError("fused FC1 workspace must use the input device")
        if workspace.pool.source_routes.data_ptr() != ctx.source_routes.data_ptr():
            raise ValueError("fused FC1 workspace belongs to a different context")
        if workspace.pool.expert_state.data_ptr() != ctx.expert_state.data_ptr():
            raise ValueError("fused FC1 workspace belongs to a different context")
        if workspace.l1_weight_data_ptr != local_weight.data_ptr():
            raise ValueError("fused workspace belongs to a different FC1 weight")
        if workspace.l2_weight_data_ptr != local_l2_weight.data_ptr():
            raise ValueError("fused workspace belongs to a different FC2 weight")

    pool = workspace.pool
    expert_state = pool.expert_state
    l2_acts = workspace.l2_acts
    l2_acts_sf_mn_major = workspace.l2_acts_sf_mn_major
    output = workspace.output
    l1_arrival = workspace.l1_arrival
    l2_arrival = workspace.l2_arrival
    fc2_scatter_grid_counter = workspace.fc2_scatter_grid_counter
    combine_cross_rank_ready = workspace.combine_cross_rank_ready
    actual_num_pool_rows = workspace.actual_num_pool_rows
    dispatch_counter = workspace.dispatch_counter
    expert_send_state = workspace.expert_send_state
    l1_a_desc = workspace.l1_a_desc
    l1_sfa_desc = workspace.l1_sfa_desc
    l1_b_desc = workspace.l1_b_desc
    l2_store_desc = workspace.l2_store_desc
    l2_a_desc = workspace.l2_a_desc
    l2_sfa_desc = workspace.l2_sfa_desc
    l2_b_desc = workspace.l2_b_desc
    dispatch_descs = workspace.dispatch_descs

    l1_n_blocks = _host_cdiv(N, _GLUON_BLOCK_N_VALUE)
    l2_n_blocks = _host_cdiv(K, _GLUON_BLOCK_N_VALUE)
    scheduler_count_capacity = _host_next_power_of_2(E)
    scheduler_counts_per_lane = max(
        _host_cdiv(scheduler_count_capacity, 32),
        1,
    )

    # Reset only compact control state.  Payload, SF padding, and metadata are
    # overwritten on valid rows and never capacity-cleared.  Keeping arrival
    # reset in this launch avoids the fragile dispatch/combine tail rendezvous
    # without restoring the old multi-MiB fixed cost.
    reset_layout = gl.BlockedLayout(
        [1],
        [32],
        [_SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS],
        [0],
    )
    reset_elements = max(
        max_pool_blocks,
        ctx.world_size,
        E,
        ctx.num_experts,
    )
    _sm90_mega_moe_fused_control_reset_kernel[
        (triton.cdiv(reset_elements, _SM90_MEGA_MOE_FUSED_RESET_BLOCK_SIZE), )
    ](
        l1_arrival,
        l2_arrival,
        actual_num_pool_rows,
        dispatch_counter,
        ctx.dispatch_barrier,
        ctx.fused_barrier,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        ctx.expert_state,
        expert_send_state,
        ctx.world_size,
        E,
        ctx.num_experts,
        max_pool_blocks,
        _SM90_MEGA_MOE_FUSED_RESET_BLOCK_SIZE,
        reset_layout,
        num_warps=_SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS,
    )
    ctx.barrier()
    # Keep the single-transport kernel under a 3D-OOB-specific JIT symbol so
    # cached artifacts and assembly reports cannot be confused with the
    # former multi-backend kernel.
    compiled = _sm90_fused_dispatch_1d2d_compact_3d_oob_tma_kernel[
        (num_sms, )
    ](
        pool.acts,
        l1_a_desc,
        l1_sfa_desc,
        l1_b_desc,
        l2_store_desc,
        l2_a_desc,
        l2_sfa_desc,
        l2_b_desc,
        *dispatch_descs,
        pool.acts_sf_mn_major,
        l2_acts_sf_mn_major,
        output,
        ctx.combine_buffer,
        ctx.peer_combine_buffer_ptrs,
        ctx.fused_barrier,
        ctx.peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        local_b_scales,
        local_l2_b_scales,
        expert_state,
        expert_send_state,
        l1_arrival,
        l2_arrival,
        actual_num_pool_rows,
        dispatch_counter,
        registered_inputs.input_topk_idx,
        pool.topk_weights,
        pool.token_src_metadata,
        ctx.peer_input_sf_ptrs,
        ctx.peer_input_topk_weights_ptrs,
        ctx.source_routes,
        ctx.recv_count,
        ctx.peer_source_routes_ptrs,
        ctx.peer_recv_count_ptrs,
        ctx.peer_expert_state_ptrs,
        ctx.dispatch_barrier,
        ctx.peer_dispatch_barrier_ptrs,
        N,
        K,
        K,
        intermediate_hidden,
        E,
        local_b_scales.stride(0),
        local_b_scales.stride(1),
        local_b_scales.stride(2),
        local_l2_b_scales.stride(0),
        local_l2_b_scales.stride(1),
        local_l2_b_scales.stride(2),
        num_stages,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
        scheduler_count_capacity,
        scheduler_counts_per_lane,
        num_padded_sf_pool_tokens,
        num_tokens,
        ctx.max_tokens,
        ctx.topk,
        ctx.num_experts,
        num_tokens * ctx.topk,
        ctx.experts_per_rank,
        ctx.max_routes,
        ctx.rank,
        ctx.world_size,
        activation_clamp,
        math.isfinite(activation_clamp),
        fast_math,
        use_swap_ab,
        _GLUON_TMA_REGS_VALUE,
        _GLUON_TMA_REGS_VALUE,
        dispatch_regs,
        num_warps=_GLUON_1D2D_MATH_WARPS,
        maxnreg=maxnreg,
    )
    return SM90MegaMoEFusedFC1Result(
        pool=pool,
        l2_acts=l2_acts,
        l2_acts_sf_mn_major=l2_acts_sf_mn_major,
        l2_arrival=l2_arrival,
        output=output[:num_tokens],
        combine_buffer=ctx.combine_buffer[:num_tokens],
        l1_arrival=l1_arrival,
        fc2_scatter_grid_counter=fc2_scatter_grid_counter,
        combine_cross_rank_ready=combine_cross_rank_ready,
        actual_num_pool_rows=actual_num_pool_rows,
        dispatch_counter=dispatch_counter,
        workspace=workspace,
        pre_dispatch=registered_inputs,
        compiled=compiled,
    )
