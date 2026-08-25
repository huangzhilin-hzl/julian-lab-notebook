################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT License.
#
################################################################################
"""Exact host contracts and an auditable single-rank SM90 MegaMoE oracle.

The routines here mirror DeepGEMM release/v0.1.5.  They intentionally stay
independent of ``triton_dist`` so they can validate both staged Gluon kernels
and the final persistent kernel under a Triton 3.6 runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


FP8_E4M3_MAX = 448.0
SM90_SMEM_CAPACITY = 232448
CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192)
LCM_CANDIDATE_BLOCK_M = 384
MAX_CANDIDATE_BLOCK_M = 192


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, alignment: int) -> int:
    return cdiv(x, alignment) * alignment


def get_num_max_pool_tokens(
    num_ranks: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> int:
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (MAX_CANDIDATE_BLOCK_M - 1),
        LCM_CANDIDATE_BLOCK_M,
    )


def get_num_sf_pool_tokens(num_pool_tokens: int, block_m: int) -> int:
    return (num_pool_tokens // block_m) * align(block_m, 128)


@dataclass(frozen=True)
class SM90MegaMoEProblem:
    num_ranks: int
    num_experts: int
    num_max_tokens_per_rank: int
    num_tokens: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_sms: int
    enable_swap_ab: bool = True

    def validate(self) -> None:
        if self.num_ranks <= 0 or self.num_experts % self.num_ranks:
            raise ValueError("num_experts must be divisible by a positive num_ranks")
        if self.num_tokens < 0 or self.num_tokens > self.num_max_tokens_per_rank:
            raise ValueError("num_tokens must be within the registered per-rank capacity")
        if self.num_max_tokens_per_rank % LCM_CANDIDATE_BLOCK_M:
            raise ValueError("num_max_tokens_per_rank must be aligned to 384")
        if self.num_topk <= 0 or self.num_topk > 32:
            raise ValueError("num_topk must be in [1, 32]")
        if self.hidden % 128 or self.intermediate_hidden % 128:
            raise ValueError("hidden and intermediate_hidden must be divisible by 128")
        if self.intermediate_hidden > 4096:
            raise ValueError("SM90 MegaMoE requires intermediate_hidden <= 4096")
        if self.num_sms <= 1 or self.num_sms % 2:
            raise ValueError("the DeepGEMM scheduler requires an even num_sms > 1")

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks

    @property
    def expected_tokens_per_expert(self) -> float:
        return self.num_tokens * self.num_ranks * self.num_topk / self.num_experts


@dataclass(frozen=True)
class SM90MegaMoEConfig:
    block_m: int
    block_n: int
    block_k: int
    cluster_size: int
    num_max_pool_tokens: int
    num_padded_sf_pool_tokens: int
    num_experts_per_wave: int
    num_stages: int
    smem_size: int
    num_dispatch_threads: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int
    use_swap_ab: bool
    reuse_accum_as_final: bool
    l2_arrival_counter: bool
    l2_epilogue_requires_full_sync: bool
    split_phase_hot_path: bool


def _get_num_wave_pool_tokens(
    num_ranks: int,
    num_topk: int,
    num_max_tokens_per_rank: int,
    num_experts_per_wave: int,
    block_m: int,
) -> int:
    if num_max_tokens_per_rank % block_m:
        raise ValueError("aligned max token capacity must be divisible by block_m")
    all_rank_tokens = num_max_tokens_per_rank * num_ranks
    if num_experts_per_wave == 1:
        return all_rank_tokens
    return min(
        all_rank_tokens * num_experts_per_wave,
        align(
            all_rank_tokens * num_topk + num_experts_per_wave * (block_m - 1),
            block_m,
        ),
    )


def _get_num_experts_per_wave(
    problem: SM90MegaMoEProblem,
    block_m: int,
    block_n: int,
    num_max_pool_tokens: int,
) -> int:
    epr = problem.num_experts_per_rank
    expected = problem.num_tokens * problem.num_topk / epr

    # SM90-specific early exits in sm90_mega_moe.hpp.
    if expected < 1.0 or expected > 4.0:
        return epr
    if block_m == 64 and problem.intermediate_hidden >= 3072:
        single_wave_blocks = epr * ((2 * problem.intermediate_hidden) // block_n)
        if single_wave_blocks >= 4 * problem.num_sms:
            return epr

    max_epw = epr
    while max_epw > 0 and _get_num_wave_pool_tokens(
        problem.num_ranks,
        problem.num_topk,
        problem.num_max_tokens_per_rank,
        max_epw,
        block_m,
    ) > num_max_pool_tokens:
        max_epw -= 1
    if max_epw == 0:
        raise ValueError("MegaMoE pool is too small for even one expert wave")

    expected_m_blocks = max(cdiv(math.ceil(expected), block_m), 1)
    l1_n_blocks = (2 * problem.intermediate_hidden) // block_n
    expected_l1_blocks_per_expert = expected_m_blocks * l1_n_blocks
    min_epw = cdiv(2 * problem.num_sms, expected_l1_blocks_per_expert)
    if expected < 1.0:
        min_epw = epr
    if min_epw >= max_epw:
        return max_epw
    if expected_l1_blocks_per_expert >= problem.num_sms:
        return min_epw

    best_epw = min_epw
    best_tail_ratio = -1.0
    for epw in range(min_epw, min(max_epw, min_epw * 2) + 1):
        remainder = epr % epw
        tail_ratio = 1.0 if remainder == 0 else remainder / epw
        if tail_ratio > best_tail_ratio:
            best_tail_ratio = tail_ratio
            best_epw = epw
    return best_epw


def get_sm90_mega_moe_config(problem: SM90MegaMoEProblem) -> SM90MegaMoEConfig:
    """Python port of ``get_mega_moe_config_sm90`` from DeepGEMM."""
    problem.validate()
    expected = problem.expected_tokens_per_expert
    if expected >= 64.0:
        block_m, num_epilogue_threads = 128, 512
    else:
        block_m, num_epilogue_threads = 64, 256

    decode_split_n = block_m == 64 and num_epilogue_threads == 256
    use_swap_ab = (
        problem.enable_swap_ab
        and decode_split_n
        and problem.num_tokens <= 128
        and (problem.num_tokens * problem.num_topk / problem.num_experts_per_rank) > 0.0
    )
    decode_bn256 = (
        decode_split_n
        and problem.intermediate_hidden >= 3072
        and expected >= 0.25
        and (2 * problem.intermediate_hidden) % 256 == 0
        and problem.hidden % 256 == 0
    )
    block_n = 128 if use_swap_ab else (256 if block_m == 128 or decode_bn256 else 128)
    block_k = 128

    num_max_pool_tokens = get_num_max_pool_tokens(
        problem.num_ranks,
        problem.num_max_tokens_per_rank,
        problem.num_topk,
        problem.num_experts_per_rank,
    )
    num_padded_sf_pool_tokens = max(
        get_num_sf_pool_tokens(num_max_pool_tokens, candidate)
        for candidate in CANDIDATE_BLOCK_M
    )
    num_experts_per_wave = _get_num_experts_per_wave(
        problem,
        block_m,
        block_n,
        num_max_pool_tokens,
    )

    num_dispatch_threads = 64
    num_non_epilogue_threads = 64
    num_dispatch_warps = num_dispatch_threads // 32
    num_epilogue_warps = num_epilogue_threads // 32

    smem_expert_count = align(problem.num_experts * 4, 1024)
    smem_send_buffers = align(problem.hidden * num_dispatch_warps, 1024)
    smem_dispatch = smem_expert_count + smem_send_buffers
    smem_cd_l1 = block_m * (block_n // 2)
    smem_cd_l2 = block_m * block_n * 2
    smem_cd_swap_l1 = block_m * (block_n // 2) * 5 if use_swap_ab else 0
    smem_cd = align(max(smem_cd_l1, smem_cd_l2, smem_cd_swap_l1), 1024)
    smem_sfa_per_stage = align(2 * block_m * 4, 128)
    smem_per_stage = block_m * block_k + block_n * block_k + smem_sfa_per_stage
    smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8
    smem_fixed = smem_dispatch + smem_cd + smem_barriers_fixed
    smem_barriers_per_stage = 16
    num_stages = (SM90_SMEM_CAPACITY - smem_fixed) // (
        smem_per_stage + smem_barriers_per_stage
    )
    if num_stages < 2:
        raise ValueError("selected SM90 MegaMoE configuration has fewer than two stages")
    smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)

    split_mn = block_m == 128 and block_n == 256 and num_epilogue_threads == 512
    decode_l2_counter = decode_split_n and block_n == 256 and 4 <= problem.num_tokens <= 128
    l2_arrival_counter = split_mn or decode_l2_counter
    return SM90MegaMoEConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        cluster_size=1,
        num_max_pool_tokens=num_max_pool_tokens,
        num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
        num_experts_per_wave=num_experts_per_wave,
        num_stages=num_stages,
        smem_size=smem_size,
        num_dispatch_threads=num_dispatch_threads,
        num_non_epilogue_threads=num_non_epilogue_threads,
        num_epilogue_threads=num_epilogue_threads,
        use_swap_ab=use_swap_ab,
        reuse_accum_as_final=block_m == 128,
        l2_arrival_counter=l2_arrival_counter,
        l2_epilogue_requires_full_sync=not l2_arrival_counter,
        split_phase_hot_path=split_mn and problem.hidden >= 7168,
    )


def quantize_per_token_per_128(
    x: torch.Tensor,
    min_amax: float = 1.0e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match SM90 pre-dispatch's continuous per-token/per-128 FP32 scale."""
    if x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("x must have shape [M, K] with K divisible by 128")
    view = x.float().view(x.shape[0], x.shape[1] // 128, 128)
    scale = view.abs().amax(dim=-1).clamp_min(min_amax) / FP8_E4M3_MAX
    quantized = (view / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.view_as(x).contiguous(), scale.contiguous()


def dequantize_per_token_per_128(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("x must have shape [M, K] with K divisible by 128")
    view = x.float().view(x.shape[0], x.shape[1] // 128, 128)
    return (view * scale.unsqueeze(-1)).view(x.shape[0], x.shape[1])


def quantize_weight_block_128x128(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [E,N,K] weights with DeepGEMM's FP32 block scales."""
    if weight.ndim != 3 or weight.shape[-2] % 128 or weight.shape[-1] % 128:
        raise ValueError("weight must be [E, N, K], with N and K divisible by 128")
    e, n, k = weight.shape
    view = weight.float().view(e, n // 128, 128, k // 128, 128)
    scale = view.abs().amax(dim=(-1, -3)).clamp_min(1.0e-4) / FP8_E4M3_MAX
    quantized = (view / scale.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return quantized.view(e, n, k).contiguous(), scale.contiguous()


def dequantize_weight_block_128x128(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 3 or weight.shape[-2] % 128 or weight.shape[-1] % 128:
        raise ValueError("weight must be [E, N, K], with N and K divisible by 128")
    e, n, k = weight.shape
    view = weight.float().view(e, n // 128, 128, k // 128, 128)
    return (view * scale.unsqueeze(-1).unsqueeze(-3)).view(e, n, k)


def interleave_l1_weight_for_sm90(weight: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    """Match ``transform_weights_for_mega_moe_sm90``'s gate/up interleave."""
    if weight.ndim < 2 or weight.shape[1] % (2 * granularity):
        raise ValueError("L1 N dimension must be divisible by 2 * granularity")
    groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(groups, half // granularity, granularity, *rest)
    up = weight[:, half:].reshape(groups, half // granularity, granularity, *rest)
    return torch.stack((gate, up), dim=2).reshape_as(weight).contiguous()


def swiglu_and_quantize_per_64(
    gate_up: torch.Tensor,
    topk_weight: torch.Tensor,
    activation_clamp: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the fused L1 epilogue, including route weight before quantize."""
    half = gate_up.shape[-1] // 2
    gate = gate_up[..., :half].float()
    up = gate_up[..., half:].float()
    if activation_clamp is not None:
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
    activation = torch.nn.functional.silu(gate) * up
    activation = activation * topk_weight.float().unsqueeze(-1)
    if half % 64:
        raise ValueError("intermediate_hidden must be divisible by 64")
    view = activation.view(*activation.shape[:-1], half // 64, 64)
    scale = view.abs().amax(dim=-1).clamp_min(1.0e-10) / FP8_E4M3_MAX
    quantized = (view / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.view_as(activation).contiguous(), scale.contiguous()


@dataclass
class SingleRankDispatch:
    acts: torch.Tensor
    acts_sf_mn_major: torch.Tensor
    topk_weights: torch.Tensor
    token_src_metadata: torch.Tensor
    expert_recv_count: torch.Tensor
    expert_pool_block_offsets: torch.Tensor
    num_pool_tokens: int


def stable_dispatch_single_rank(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_m: int,
) -> SingleRankDispatch:
    """Build DeepGEMM's expert pool contract in stable token/top-k order."""
    if topk_idx.shape != topk_weights.shape:
        raise ValueError("topk_idx and topk_weights must have identical shapes")
    if topk_idx.shape[0] != x_fp8.shape[0] or x_sf.shape[0] != x_fp8.shape[0]:
        raise ValueError("all dispatch inputs must have the same token dimension")
    flat_experts = topk_idx.reshape(-1)
    valid = flat_experts >= 0
    counts = torch.bincount(flat_experts[valid].long(), minlength=num_experts)
    num_blocks = torch.tensor(
        [cdiv(int(count), block_m) for count in counts.tolist()],
        dtype=torch.int64,
        device=counts.device,
    )
    block_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=counts.device)
    block_offsets[1:] = torch.cumsum(num_blocks, dim=0)
    num_pool_tokens = int(block_offsets[-1].item()) * block_m
    num_sf_pool_tokens = int(block_offsets[-1].item()) * align(block_m, 128)

    acts = torch.zeros(
        (num_pool_tokens, x_fp8.shape[1]),
        dtype=x_fp8.dtype,
        device=x_fp8.device,
    )
    acts_sf = torch.zeros(
        (x_sf.shape[1], num_sf_pool_tokens),
        dtype=torch.float32,
        device=x_sf.device,
    )
    route_weights = torch.zeros(num_pool_tokens, dtype=torch.float32, device=x_fp8.device)
    metadata = torch.full(
        (num_pool_tokens, 3),
        -1,
        dtype=torch.int64,
        device=x_fp8.device,
    )

    topk = topk_idx.shape[1]
    for expert in range(num_experts):
        route_positions = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        if route_positions.numel() == 0:
            continue
        start = int(block_offsets[expert].item()) * block_m
        pool_positions = start + torch.arange(route_positions.numel(), device=x_fp8.device)
        source_tokens = torch.div(route_positions, topk, rounding_mode="floor")
        source_topk = route_positions % topk
        acts[pool_positions] = x_fp8[source_tokens]
        route_weights[pool_positions] = topk_weights.reshape(-1)[route_positions]
        metadata[pool_positions, 0] = 0
        metadata[pool_positions, 1] = source_tokens
        metadata[pool_positions, 2] = source_topk

        expert_block_offset = int(block_offsets[expert].item())
        for local_idx, source_token in enumerate(source_tokens.tolist()):
            block = local_idx // block_m
            row = local_idx % block_m
            sf_token = (expert_block_offset + block) * align(block_m, 128) + row
            acts_sf[:, sf_token] = x_sf[source_token]

    return SingleRankDispatch(
        acts=acts,
        acts_sf_mn_major=acts_sf,
        topk_weights=route_weights,
        token_src_metadata=metadata,
        expert_recv_count=counts.to(torch.int32),
        expert_pool_block_offsets=block_offsets,
        num_pool_tokens=num_pool_tokens,
    )


def mega_moe_reference_single_rank(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weight_fp8: torch.Tensor,
    l1_weight_sf: torch.Tensor,
    l2_weight_fp8: torch.Tensor,
    l2_weight_sf: torch.Tensor,
    activation_clamp: Optional[float] = None,
) -> torch.Tensor:
    """FP32/BF16 oracle for the complete one-rank SM90 MegaMoE contract."""
    if l1_weight_fp8.shape[0] != l2_weight_fp8.shape[0]:
        raise ValueError("L1 and L2 must contain the same number of experts")
    num_tokens, topk = topk_idx.shape
    hidden = x_fp8.shape[1]
    num_experts = l1_weight_fp8.shape[0]
    x = dequantize_per_token_per_128(x_fp8, x_sf)
    l1_weight = dequantize_weight_block_128x128(l1_weight_fp8, l1_weight_sf)
    l2_weight = dequantize_weight_block_128x128(l2_weight_fp8, l2_weight_sf)
    combine = torch.zeros(
        (num_tokens, topk, hidden),
        dtype=torch.float32,
        device=x.device,
    )

    for slot in range(topk):
        valid = topk_idx[:, slot] >= 0
        if not bool(valid.any()):
            continue
        token_indices = torch.nonzero(valid, as_tuple=False).flatten()
        expert_indices = topk_idx[token_indices, slot].long()
        if bool((expert_indices >= num_experts).any()):
            raise ValueError("topk_idx contains an out-of-range expert")
        l1 = torch.einsum(
            "mk,mnk->mn",
            x[token_indices],
            l1_weight[expert_indices],
        )
        l2_fp8, l2_scale = swiglu_and_quantize_per_64(
            l1,
            topk_weights[token_indices, slot],
            activation_clamp,
        )
        l2_input = (
            l2_fp8.float().view(l2_fp8.shape[0], -1, 64)
            * l2_scale.unsqueeze(-1)
        ).view(l2_fp8.shape)
        l2 = torch.einsum(
            "mk,mnk->mn",
            l2_input,
            l2_weight[expert_indices],
        )
        combine[token_indices, slot] = l2.to(torch.bfloat16).float()

    return combine.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16).contiguous()


def enumerate_persistent_tasks(
    expert_recv_count: torch.Tensor,
    *,
    block_m: int,
    l1_n_blocks: int,
    l2_n_blocks: int,
    num_experts_per_wave: int,
    num_sms: int,
) -> list[list[tuple[int, int, int, int, int]]]:
    """CPU oracle for ``MegaMoEScheduler::get_next_block``.

    Each task is ``(phase, expert, m_block, n_block, pool_block)`` where phase
    is 1 for Linear1 and 2 for Linear2.
    """
    counts = [int(value) for value in expert_recv_count.cpu().tolist()]
    num_experts = len(counts)
    num_m_blocks = [cdiv(value, block_m) for value in counts]
    pool_offsets = [0]
    for blocks in num_m_blocks:
        pool_offsets.append(pool_offsets[-1] + blocks)

    traces: list[list[tuple[int, int, int, int, int]]] = []
    for sm in range(num_sms):
        block_idx = sm
        expert = 0
        phase = 1
        trace: list[tuple[int, int, int, int, int]] = []
        while expert < num_experts:
            wave_end = min(align(expert + 1, num_experts_per_wave), num_experts)
            n_blocks = l1_n_blocks if phase == 1 else l2_n_blocks
            found = False
            while expert < wave_end:
                expert_tasks = num_m_blocks[expert] * n_blocks
                if block_idx < expert_tasks:
                    m_block = block_idx // n_blocks
                    n_block = block_idx - m_block * n_blocks
                    trace.append((
                        phase,
                        expert,
                        m_block,
                        n_block,
                        pool_offsets[expert] + m_block,
                    ))
                    block_idx += num_sms
                    found = True
                    break
                block_idx -= expert_tasks
                expert += 1
            if found:
                continue
            if phase == 1:
                phase = 2
                expert = ((expert - 1) // num_experts_per_wave) * num_experts_per_wave
            else:
                phase = 1
        traces.append(trace)
    return traces
