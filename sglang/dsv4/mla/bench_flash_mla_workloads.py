from __future__ import annotations

import argparse
import csv
import dataclasses
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
for path in (REPO_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SPLITKV_KERNEL = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
COMBINE_KERNEL = "flash_fwd_mla_combine_kernel"

flash_mla = None
kk = None
quant = None
ref = None
ExtraTestParamForDecode = None
KVScope = None
TestcaseForDecode = None
TestParam = None


def load_runtime_deps() -> None:
    global flash_mla, kk, quant, ref
    global ExtraTestParamForDecode, KVScope, TestcaseForDecode, TestParam
    if flash_mla is not None:
        return

    import flash_mla as flash_mla_module
    import kernelkit as kk_module
    import quant as quant_module
    import ref as ref_module
    from lib import (
        ExtraTestParamForDecode as ExtraTestParamForDecodeCls,
        KVScope as KVScopeCls,
        TestcaseForDecode as TestcaseForDecodeCls,
        TestParam as TestParamCls,
    )

    flash_mla = flash_mla_module
    kk = kk_module
    quant = quant_module
    ref = ref_module
    ExtraTestParamForDecode = ExtraTestParamForDecodeCls
    KVScope = KVScopeCls
    TestcaseForDecode = TestcaseForDecodeCls
    TestParam = TestParamCls


@dataclasses.dataclass(frozen=True)
class WorkloadProfile:
    name: str
    model_id: str
    h_q: int
    index_topk: int
    d_qk: int = 512
    d_v: int = 512
    h_kv: int = 1
    swa_window: int = 128
    swa_block_size: int = 256
    c4_block_size: int = 64
    c128_block_size: int = 2
    c128_topk: int = 1024


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    compress_ratio: int
    extra_topk_attr: Optional[str]
    extra_block_size_attr: Optional[str]


@dataclasses.dataclass
class KVScopeInput:
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor
    blocked_k: torch.Tensor
    blocked_k_quantized: torch.Tensor
    abs_indices: torch.Tensor
    indices_in_kvcache: torch.Tensor
    topk_length: torch.Tensor
    logical_lengths: torch.Tensor


@dataclasses.dataclass
class BenchCase:
    workload: WorkloadProfile
    scenario: Scenario
    batch_size: int
    s_q: int
    mean_seq_len: int
    original_seq_lens: torch.Tensor
    primary_scope: KVScopeInput
    extra_scope: Optional[KVScopeInput]
    q: torch.Tensor
    attn_sink: torch.Tensor
    softmax_scale: float
    test_param: TestParam
    testcase: TestcaseForDecode


WORKLOADS: Dict[str, WorkloadProfile] = {
    "dsv4-pro": WorkloadProfile(
        name="dsv4-pro",
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        h_q=128,
        index_topk=1024,
    ),
    "dsv4-flash": WorkloadProfile(
        name="dsv4-flash",
        model_id="sgl-project/DeepSeek-V4-Flash-FP8",
        h_q=64,
        index_topk=512,
    ),
}


SCENARIOS: Dict[str, Scenario] = {
    "swa": Scenario(
        name="swa",
        compress_ratio=0,
        extra_topk_attr=None,
        extra_block_size_attr=None,
    ),
    "c4": Scenario(
        name="c4",
        compress_ratio=4,
        extra_topk_attr="index_topk",
        extra_block_size_attr="c4_block_size",
    ),
    "c128": Scenario(
        name="c128",
        compress_ratio=128,
        extra_topk_attr="c128_topk",
        extra_block_size_attr="c128_block_size",
    ),
}


def parse_int_list(value: str) -> List[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_seq_lens(batch_size: int, mean_seq_len: int, varlen: bool) -> torch.Tensor:
    if not varlen:
        return torch.full((batch_size,), mean_seq_len, dtype=torch.int32, device="cpu")

    values = []
    for _ in range(batch_size):
        sample = int(random.normalvariate(mean_seq_len, max(mean_seq_len / 2, 1)))
        values.append(min(max(sample, 1), max(mean_seq_len * 2, 1)))
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def make_block_table(
    batch_size: int,
    max_logical_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    num_blocks_per_seq = max(ceil_div(max_logical_len, block_size), 1)
    block_table = torch.arange(
        batch_size * num_blocks_per_seq,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, num_blocks_per_seq)
    return block_table.view(-1)[torch.randperm(block_table.numel(), device=device)].view(
        batch_size,
        num_blocks_per_seq,
    )


def make_abs_indices(
    logical_lengths: torch.Tensor,
    s_q: int,
    topk_capacity: int,
    topk_lengths: torch.Tensor,
    pattern: str,
    device: torch.device,
) -> torch.Tensor:
    batch_size = logical_lengths.numel()
    indices = torch.full(
        (batch_size, s_q, topk_capacity),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for batch_idx in range(batch_size):
        logical_len = int(logical_lengths[batch_idx].item())
        topk_len = int(topk_lengths[batch_idx].item())
        if logical_len <= 0 or topk_len <= 0:
            continue
        topk_len = min(topk_len, topk_capacity, logical_len)
        if pattern == "recent":
            row = torch.arange(
                logical_len - topk_len,
                logical_len,
                dtype=torch.int32,
                device=device,
            )
        elif pattern == "random":
            row = torch.randperm(logical_len, device=device, dtype=torch.int64)[:topk_len]
            row = row.to(torch.int32).sort().values
        else:
            raise ValueError(f"unsupported index pattern: {pattern}")
        indices[batch_idx, :, :topk_len] = row.view(1, topk_len).expand(s_q, topk_len)
    return indices


def make_kv_scope(
    *,
    test_param: TestParam,
    logical_lengths_cpu: torch.Tensor,
    s_q: int,
    topk_capacity: int,
    block_size: int,
    pattern: str,
    device: torch.device,
) -> KVScopeInput:
    batch_size = logical_lengths_cpu.numel()
    logical_lengths_cpu = logical_lengths_cpu.clamp_min(1)
    max_logical_len = int(logical_lengths_cpu.max().item())
    max_logical_len = max(align_up(max_logical_len, block_size), block_size)

    block_table = make_block_table(batch_size, max_logical_len, block_size, device)
    num_blocks = block_table.numel()
    blocked_k = torch.randn(
        (num_blocks, block_size, test_param.h_kv, test_param.d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    blocked_k = (blocked_k / 10).clamp_(min=-1.0, max=1.0)

    topk_lengths_cpu = torch.minimum(
        logical_lengths_cpu,
        torch.full_like(logical_lengths_cpu, topk_capacity),
    )
    topk_length = topk_lengths_cpu.to(device=device, dtype=torch.int32)
    logical_lengths = logical_lengths_cpu.to(device=device, dtype=torch.int32)
    abs_indices = make_abs_indices(
        logical_lengths,
        s_q,
        topk_capacity,
        topk_length,
        pattern,
        device,
    )
    indices_in_kvcache = quant.abs_indices2indices_in_kvcache(
        abs_indices,
        block_table,
        block_size,
    )

    blocked_k_quantized = quant.quantize_k_cache(
        blocked_k,
        quant.FP8KVCacheLayout.MODEL1_FP8Sparse,
    )
    return KVScopeInput(
        cache_seqlens=logical_lengths,
        block_table=block_table,
        blocked_k=blocked_k,
        blocked_k_quantized=blocked_k_quantized,
        abs_indices=abs_indices,
        indices_in_kvcache=indices_in_kvcache,
        topk_length=topk_length,
        logical_lengths=logical_lengths,
    )


def to_lib_kv_scope(test_param: TestParam, scope: KVScopeInput) -> KVScope:
    return KVScope(
        test_param,
        scope.cache_seqlens,
        scope.block_table,
        scope.blocked_k,
        scope.abs_indices,
        scope.indices_in_kvcache,
        scope.topk_length,
        scope.blocked_k_quantized,
    )


def build_case(
    *,
    workload: WorkloadProfile,
    scenario: Scenario,
    batch_size: int,
    s_q: int,
    mean_seq_len: int,
    seed: int,
    varlen: bool,
    device: torch.device,
) -> BenchCase:
    set_seed(seed)
    seq_lens_cpu = make_seq_lens(batch_size, mean_seq_len, varlen)
    primary_lengths_cpu = torch.minimum(
        seq_lens_cpu,
        torch.full_like(seq_lens_cpu, workload.swa_window),
    )

    extra_topk = (
        getattr(workload, scenario.extra_topk_attr)
        if scenario.extra_topk_attr is not None
        else None
    )
    extra_block_size = (
        getattr(workload, scenario.extra_block_size_attr)
        if scenario.extra_block_size_attr is not None
        else None
    )
    extra_s_k = (
        int(ceil_div(int(seq_lens_cpu.max().item()), scenario.compress_ratio))
        if scenario.compress_ratio
        else None
    )

    test_param = TestParam(
        s_q=s_q,
        s_kv=int(primary_lengths_cpu.max().item()),
        topk=workload.swa_window,
        h_q=workload.h_q,
        h_kv=workload.h_kv,
        d_qk=workload.d_qk,
        d_v=workload.d_v,
        seed=seed,
        check_correctness=False,
        is_all_indices_invalid=False,
        num_runs=0,
        have_attn_sink=True,
        have_topk_length=True,
        decode=ExtraTestParamForDecode(
            b=batch_size,
            is_varlen=varlen,
            have_zero_seqlen_k=False,
            extra_s_k=extra_s_k,
            extra_topk=extra_topk,
            block_size=workload.swa_block_size,
            extra_block_size=extra_block_size,
            have_extra_topk_length=extra_topk is not None,
        ),
    )

    q = torch.randn(
        (batch_size, s_q, workload.h_q, workload.d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    q = (q / 10).clamp_(min=-1.0, max=1.0)
    attn_sink = torch.zeros((workload.h_q,), dtype=torch.float32, device=device)
    softmax_scale = workload.d_qk ** -0.5

    primary_scope = make_kv_scope(
        test_param=test_param,
        logical_lengths_cpu=primary_lengths_cpu,
        s_q=s_q,
        topk_capacity=workload.swa_window,
        block_size=workload.swa_block_size,
        pattern="recent",
        device=device,
    )

    extra_scope = None
    if scenario.compress_ratio:
        assert extra_topk is not None and extra_block_size is not None
        extra_lengths_cpu = torch.clamp(
            torch.div(
                seq_lens_cpu + scenario.compress_ratio - 1,
                scenario.compress_ratio,
                rounding_mode="floor",
            ),
            min=1,
        )
        extra_scope = make_kv_scope(
            test_param=test_param,
            logical_lengths_cpu=extra_lengths_cpu,
            s_q=s_q,
            topk_capacity=extra_topk,
            block_size=extra_block_size,
            pattern="random",
            device=device,
        )

    testcase = TestcaseForDecode(
        p=test_param,
        q=q,
        attn_sink=attn_sink,
        sm_scale=softmax_scale,
        kv_scope=to_lib_kv_scope(test_param, primary_scope),
        extra_kv_scope=to_lib_kv_scope(test_param, extra_scope)
        if extra_scope is not None
        else None,
    )
    return BenchCase(
        workload=workload,
        scenario=scenario,
        batch_size=batch_size,
        s_q=s_q,
        mean_seq_len=mean_seq_len,
        original_seq_lens=seq_lens_cpu.to(device=device, dtype=torch.int32),
        primary_scope=primary_scope,
        extra_scope=extra_scope,
        q=q,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        test_param=test_param,
        testcase=testcase,
    )


def make_runner(case: BenchCase):
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

    def run_decode():
        return flash_mla.flash_mla_with_kvcache(
            q=case.q,
            k_cache=case.primary_scope.blocked_k_quantized,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=case.workload.d_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=case.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=case.primary_scope.indices_in_kvcache,
            attn_sink=case.attn_sink,
            extra_k_cache=case.extra_scope.blocked_k_quantized
            if case.extra_scope is not None
            else None,
            extra_indices_in_kvcache=case.extra_scope.indices_in_kvcache
            if case.extra_scope is not None
            else None,
            topk_length=case.primary_scope.topk_length,
            extra_topk_length=case.extra_scope.topk_length
            if case.extra_scope is not None
            else None,
        )

    return run_decode


def summarize_scope(scope: KVScopeInput) -> Tuple[float, int]:
    topk_avg = float(scope.topk_length.float().mean().item())
    valid = scope.indices_in_kvcache >= 0
    if scope.topk_length is not None:
        topk = scope.indices_in_kvcache.shape[-1]
        valid &= torch.arange(topk, device=scope.indices_in_kvcache.device).view(
            1, 1, topk
        ) < scope.topk_length.view(-1, 1, 1)
    unique_tokens = int(scope.indices_in_kvcache[valid].unique().numel())
    return topk_avg, unique_tokens


def compute_perf_stats(case: BenchCase, elapsed_s: float) -> Tuple[float, float, float]:
    primary_topk_avg, primary_unique = summarize_scope(case.primary_scope)
    total_attended = int(case.primary_scope.topk_length.sum().item()) * case.s_q
    total_unique = primary_unique
    extra_topk_avg = 0.0
    if case.extra_scope is not None:
        extra_topk_avg, extra_unique = summarize_scope(case.extra_scope)
        total_attended += int(case.extra_scope.topk_length.sum().item()) * case.s_q
        total_unique += extra_unique

    flop = 2 * case.workload.h_q * total_attended * (
        case.workload.d_qk + case.workload.d_v
    )
    kv_token_size = case.primary_scope.blocked_k_quantized.shape[-1]
    mem_bytes = sum(
        [
            2 * case.batch_size * case.s_q * case.workload.h_q * case.workload.d_qk,
            total_unique * kv_token_size,
            2 * case.batch_size * case.s_q * case.workload.h_q * case.workload.d_v,
        ]
    )
    tflops = flop / elapsed_s / 1e12
    gbps = mem_bytes / elapsed_s / 1e9
    return tflops, gbps, extra_topk_avg or primary_topk_avg


def check_correctness(case: BenchCase, out: torch.Tensor, lse: torch.Tensor) -> bool:
    out_ref, lse_ref = ref.ref_sparse_attn_decode(case.test_param, case.testcase)
    is_out_correct = kk.check_is_allclose(
        "out",
        out,
        out_ref,
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    is_lse_correct = kk.check_is_allclose(
        "lse",
        lse,
        lse_ref,
        abs_tol=1e-6,
        rel_tol=8.01 / 65536,
    )
    return bool(is_out_correct and is_lse_correct)


def get_kineto_times(run_decode, num_runs: int) -> Tuple[float, float]:
    result = kk.bench_kineto(run_decode, num_runs)
    splitkv_us = result.get_kernel_time(SPLITKV_KERNEL) * 1e6
    combine_candidates = [COMBINE_KERNEL in name for name in result.get_kernel_names()]
    combine_us = (
        result.get_kernel_time(COMBINE_KERNEL) * 1e6 if any(combine_candidates) else 0.0
    )
    return splitkv_us, combine_us


def iter_cases(args) -> Iterable[Tuple[WorkloadProfile, Scenario, int, int]]:
    workloads = WORKLOADS.values() if args.workload == "all" else [WORKLOADS[args.workload]]
    scenarios = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]
    for workload in workloads:
        for scenario in scenarios:
            for batch_size in args.batch_sizes:
                for seq_len in args.seq_lens:
                    yield workload, scenario, batch_size, seq_len


def run_case(case: BenchCase, args) -> Dict[str, object]:
    run_decode = make_runner(case)

    torch.cuda.synchronize()
    out = lse = None
    for _ in range(args.warmup):
        out, lse = run_decode()
    torch.cuda.synchronize()

    if args.check_correctness:
        assert out is not None and lse is not None
        is_correct = check_correctness(case, out, lse)
    else:
        is_correct = True

    elapsed_s = kk.bench_by_cuda_events(
        run_decode,
        num_warmups_each=0,
        num_runs_each=args.runs,
    )
    splitkv_us = 0.0
    combine_us = 0.0
    if args.kineto:
        splitkv_us, combine_us = get_kineto_times(run_decode, args.kineto_runs)

    primary_topk_avg, primary_unique = summarize_scope(case.primary_scope)
    extra_topk_avg = 0.0
    extra_unique = 0
    if case.extra_scope is not None:
        extra_topk_avg, extra_unique = summarize_scope(case.extra_scope)

    tflops, gbps, _ = compute_perf_stats(case, elapsed_s)
    original_seq_lens = case.original_seq_lens

    return {
        "workload": case.workload.name,
        "model_id": case.workload.model_id,
        "scenario": case.scenario.name,
        "compress_ratio": case.scenario.compress_ratio,
        "batch_size": case.batch_size,
        "s_q": case.s_q,
        "seq_len_target": case.mean_seq_len,
        "seq_len_mean": f"{original_seq_lens.float().mean().item():.1f}",
        "seq_len_min": int(original_seq_lens.min().item()),
        "seq_len_max": int(original_seq_lens.max().item()),
        "h_q": case.workload.h_q,
        "h_kv": case.workload.h_kv,
        "d_qk": case.workload.d_qk,
        "d_v": case.workload.d_v,
        "primary_topk": case.primary_scope.indices_in_kvcache.shape[-1],
        "primary_topk_avg": f"{primary_topk_avg:.1f}",
        "primary_unique_tokens": primary_unique,
        "extra_topk": case.extra_scope.indices_in_kvcache.shape[-1]
        if case.extra_scope is not None
        else 0,
        "extra_topk_avg": f"{extra_topk_avg:.1f}",
        "extra_unique_tokens": extra_unique,
        "primary_block_size": case.primary_scope.blocked_k_quantized.shape[1],
        "extra_block_size": case.extra_scope.blocked_k_quantized.shape[1]
        if case.extra_scope is not None
        else 0,
        "time_us": f"{elapsed_s * 1e6:.1f}",
        "splitkv_us": f"{splitkv_us:.1f}",
        "combine_us": f"{combine_us:.1f}",
        "tflops": f"{tflops:.1f}",
        "gbps": f"{gbps:.0f}",
        "correct": int(is_correct),
        "device": torch.cuda.get_device_name(),
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark flash_mla_with_kvcache with DSV4 Pro and DSV4 Flash workload shapes."
    )
    parser.add_argument(
        "--workload",
        choices=["all", *WORKLOADS.keys()],
        default="all",
        help="Workload profile to run.",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="Attention scenario: SWA-only, SWA+C4, or SWA+C128.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("1,32,128"),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--seq-lens",
        type=parse_int_list,
        default=parse_int_list("4096,32768"),
        help="Comma-separated target original context lengths.",
    )
    parser.add_argument("--s-q", type=int, default=1, help="Query tokens per request.")
    parser.add_argument("--runs", type=int, default=20, help="CUDA-event timed runs.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per case.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-seqlen", action="store_true", help="Disable varlen sampling.")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--kineto", action="store_true", help="Also collect splitkv/combine kernel times.")
    parser.add_argument("--kineto-runs", type=int, default=10)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flash_mla_workload_perf.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--no-cooldown", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    case_specs = list(iter_cases(args))
    if args.case_limit:
        case_specs = case_specs[: args.case_limit]

    if args.dry_run:
        for workload, scenario, batch_size, seq_len in case_specs:
            print(
                f"{workload.name},{scenario.name},batch={batch_size},"
                f"s_q={args.s_q},seq_len={seq_len}"
            )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark flash_mla_with_kvcache.")
    load_runtime_deps()
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(device)
    torch.set_num_threads(32)

    rows = []
    for idx, (workload, scenario, batch_size, seq_len) in enumerate(case_specs, start=1):
        if rows and not args.no_cooldown:
            time.sleep(0.3)
        print(
            f"[{idx}/{len(case_specs)}] workload={workload.name} "
            f"scenario={scenario.name} batch={batch_size} seq_len={seq_len}"
        )
        case = build_case(
            workload=workload,
            scenario=scenario,
            batch_size=batch_size,
            s_q=args.s_q,
            mean_seq_len=seq_len,
            seed=args.seed + idx - 1,
            varlen=not args.fixed_seqlen,
            device=device,
        )
        row = run_case(case, args)
        rows.append(row)
        print(
            f"  {row['time_us']} us, {row['tflops']} TFLOPS, "
            f"{row['gbps']} GB/s, correct={row['correct']}"
        )

    if rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
