import argparse
import csv
import dataclasses
import math
import time
from pathlib import Path
from types import SimpleNamespace

from bench_humming import bench_humming

from humming.config import GemmType
from humming.utils.test import save_benchmark_result


@dataclasses.dataclass(frozen=True)
class Dsv4FlashConfig:
    workload: str = "dsv4-flash"
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    top_k: int = 6


@dataclasses.dataclass(frozen=True)
class LegSpec:
    name: str
    shape_n: int
    shape_k: int
    is_moe_down: bool


DSV4_FLASH = Dsv4FlashConfig()
DEFAULT_TP_SIZES = [4, 8]
DEFAULT_GLOBAL_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

ACTIVATION_DTYPE = "float8e4m3"
WEIGHT_DTYPE = "float4e2m1"
WEIGHT_SCALE_DTYPE = "float8e8m0"
OUTPUT_DTYPE = "float16"
INPUT_SCALE_GROUP_SIZE = 0
WEIGHT_SCALE_GROUP_SIZE = 32
LOCAL_GEMM_TOP_K = 1


def parse_int_list(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def parse_leg_list(value: str) -> list[str]:
    result = [x.strip() for x in value.split(",") if x.strip()]
    invalid = sorted(set(result) - {"w13", "w2"})
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid leg(s): {', '.join(invalid)}")
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated leg list")
    return result


def make_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks/results/h20") / f"dsv4_flash_mxfp4a8_moe_{timestamp}"


def leg_specs_for_tp(tp_size: int) -> dict[str, LegSpec]:
    if DSV4_FLASH.n_routed_experts % tp_size != 0:
        raise ValueError(
            f"n_routed_experts={DSV4_FLASH.n_routed_experts} is not divisible by tp_size={tp_size}"
        )
    if DSV4_FLASH.moe_intermediate_size % tp_size != 0:
        raise ValueError(
            "moe_intermediate_size="
            f"{DSV4_FLASH.moe_intermediate_size} is not divisible by tp_size={tp_size}"
        )

    intermediate_per_partition = DSV4_FLASH.moe_intermediate_size // tp_size
    return {
        "w13": LegSpec(
            name="w13",
            shape_n=2 * intermediate_per_partition,
            shape_k=DSV4_FLASH.hidden_size,
            is_moe_down=False,
        ),
        "w2": LegSpec(
            name="w2",
            shape_n=DSV4_FLASH.hidden_size,
            shape_k=intermediate_per_partition,
            is_moe_down=True,
        ),
    }


def local_rows_for_global_tokens(global_tokens: list[int], tp_size: int) -> list[int]:
    return [max(1, math.ceil(tokens * DSV4_FLASH.top_k / tp_size)) for tokens in global_tokens]


def make_raw_args(
    *,
    output_file: Path,
    tp_size: int,
    local_experts: int,
    leg: LegSpec,
    global_tokens: list[int],
    local_rows: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        workload=DSV4_FLASH.workload,
        tp_size=tp_size,
        local_experts=local_experts,
        global_tokens_list=global_tokens,
        local_rows_list=local_rows,
        leg=leg.name,
        shape_n=leg.shape_n,
        shape_k=leg.shape_k,
        a_dtype=ACTIVATION_DTYPE,
        b_dtype=WEIGHT_DTYPE,
        bs_dtype=WEIGHT_SCALE_DTYPE,
        c_dtype=OUTPUT_DTYPE,
        input_scale_group_size=INPUT_SCALE_GROUP_SIZE,
        weight_scale_group_size=WEIGHT_SCALE_GROUP_SIZE,
        zero_point=False,
        use_fp_zero_point=False,
        use_f16_accum=False,
        num_experts=local_experts,
        top_k=LOCAL_GEMM_TOP_K,
        is_moe_down=leg.is_moe_down,
        balanced=True,
        expert_max_tokens=None,
        shape_m_list=local_rows,
        gemm_type=GemmType.INDEXED.value,
        output_file=str(output_file),
    )


def benchmark_one_group(
    *,
    tp_size: int,
    leg: LegSpec,
    global_tokens: list[int],
    output_dir: Path,
) -> list[dict[str, int | float | str | bool]]:
    local_experts = DSV4_FLASH.n_routed_experts // tp_size
    local_rows = local_rows_for_global_tokens(global_tokens, tp_size)

    print(
        f"[run] workload={DSV4_FLASH.workload} tp={tp_size} leg={leg.name} "
        f"local_experts={local_experts} shape_n={leg.shape_n} shape_k={leg.shape_k} "
        f"local_rows={local_rows}"
    )

    result = bench_humming(
        shape_n=leg.shape_n,
        shape_k=leg.shape_k,
        a_dtype=ACTIVATION_DTYPE,
        b_dtype=WEIGHT_DTYPE,
        c_dtype=OUTPUT_DTYPE,
        bs_dtype=WEIGHT_SCALE_DTYPE,
        input_scale_group_size=INPUT_SCALE_GROUP_SIZE,
        weight_scale_group_size=WEIGHT_SCALE_GROUP_SIZE,
        num_experts=local_experts,
        top_k=LOCAL_GEMM_TOP_K,
        has_zero_point=False,
        is_fp_zero_point=False,
        use_f16_accum=False,
        is_moe_down=leg.is_moe_down,
        balanced=True,
        expert_max_tokens=None,
        shape_m_list=local_rows,
        gemm_type=GemmType.INDEXED,
    )

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_json = raw_dir / f"{DSV4_FLASH.workload}_tp{tp_size}_{leg.name}.json"
    raw_args = make_raw_args(
        output_file=raw_json,
        tp_size=tp_size,
        local_experts=local_experts,
        leg=leg,
        global_tokens=global_tokens,
        local_rows=local_rows,
    )
    save_benchmark_result(result, raw_args)

    result_by_local_rows = {int(row["shape_m"]): row for row in result}
    rows: list[dict[str, int | float | str | bool]] = []
    for tokens, rows_per_rank in zip(global_tokens, local_rows):
        result_row = result_by_local_rows[rows_per_rank]
        rows.append(
            {
                "workload": DSV4_FLASH.workload,
                "tp_size": tp_size,
                "leg": leg.name,
                "global_tokens": tokens,
                "local_routed_rows": rows_per_rank,
                "local_experts": local_experts,
                "shape_n": leg.shape_n,
                "shape_k": leg.shape_k,
                "a_dtype": ACTIVATION_DTYPE,
                "b_dtype": WEIGHT_DTYPE,
                "bs_dtype": WEIGHT_SCALE_DTYPE,
                "input_scale_group_size": INPUT_SCALE_GROUP_SIZE,
                "weight_scale_group_size": WEIGHT_SCALE_GROUP_SIZE,
                "gemm_type": GemmType.INDEXED.value,
                "is_moe_down": leg.is_moe_down,
                "time_ms": float(result_row["time"]),
                "compute_tops": float(result_row["compute_tops"]),
                "memory_gbps": float(result_row["memory_gbps"]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, int | float | str | bool]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def combined_rows(rows: list[dict[str, int | float | str | bool]]) -> list[dict[str, int | float | str]]:
    by_key: dict[tuple[int, int], dict[str, dict[str, int | float | str | bool]]] = {}
    for row in rows:
        key = (int(row["tp_size"]), int(row["global_tokens"]))
        by_key.setdefault(key, {})[str(row["leg"])] = row

    combined: list[dict[str, int | float | str]] = []
    for key in sorted(by_key):
        legs = by_key[key]
        if "w13" not in legs or "w2" not in legs:
            continue
        w13 = legs["w13"]
        w2 = legs["w2"]
        time_ms = float(w13["time_ms"]) + float(w2["time_ms"])
        nbytes = (
            float(w13["memory_gbps"]) * float(w13["time_ms"]) * 1e6
            + float(w2["memory_gbps"]) * float(w2["time_ms"]) * 1e6
        )
        ops = (
            int(w13["local_routed_rows"]) * int(w13["shape_n"]) * int(w13["shape_k"]) * 2
            + int(w2["local_routed_rows"]) * int(w2["shape_n"]) * int(w2["shape_k"]) * 2
        )
        combined.append(
            {
                "tp_size": key[0],
                "global_tokens": key[1],
                "local_routed_rows": int(w13["local_routed_rows"]),
                "combined_time_ms": time_ms,
                "combined_compute_tops": ops / time_ms / 1e9,
                "combined_memory_gbps": nbytes / time_ms / 1e6,
            }
        )
    return combined


def markdown_table(rows: list[dict[str, int | float | str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    *,
    output_file: Path,
    rows: list[dict[str, int | float | str | bool]],
    combined: list[dict[str, int | float | str]],
) -> None:
    summary_lines = [
        "# DSV4 Flash Humming MXFP4A8 MoE microbench",
        "",
        "## Model and benchmark config",
        "",
        f"- workload: `{DSV4_FLASH.workload}`",
        f"- hidden_size: `{DSV4_FLASH.hidden_size}`",
        f"- moe_intermediate_size: `{DSV4_FLASH.moe_intermediate_size}`",
        f"- n_routed_experts: `{DSV4_FLASH.n_routed_experts}`",
        f"- model_top_k: `{DSV4_FLASH.top_k}`",
        f"- local GEMM top_k: `{LOCAL_GEMM_TOP_K}`",
        f"- activation/weight/scale: `{ACTIVATION_DTYPE}` / `{WEIGHT_DTYPE}` / `{WEIGHT_SCALE_DTYPE}`",
        f"- input_scale_group_size: `{INPUT_SCALE_GROUP_SIZE}`",
        f"- weight_scale_group_size: `{WEIGHT_SCALE_GROUP_SIZE}`",
        "- routing: balanced synthetic local routing",
        "",
        "## Per-leg results",
        "",
        markdown_table(
            rows,
            [
                "tp_size",
                "leg",
                "global_tokens",
                "local_routed_rows",
                "local_experts",
                "shape_n",
                "shape_k",
                "time_ms",
                "compute_tops",
                "memory_gbps",
            ],
        ),
        "",
        "## Combined per-rank MoE GEMM cost",
        "",
        markdown_table(
            combined,
            [
                "tp_size",
                "global_tokens",
                "local_routed_rows",
                "combined_time_ms",
                "combined_compute_tops",
                "combined_memory_gbps",
            ],
        ),
        "",
    ]
    output_file.write_text("\n".join(summary_lines))


def validate_rows(rows: list[dict[str, int | float | str | bool]]) -> None:
    for row in rows:
        for key in ("time_ms", "compute_tops", "memory_gbps"):
            value = float(row[key])
            if value <= 0:
                raise RuntimeError(f"non-positive {key} in row: {row}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DSV4 Flash serving-shard Humming MXFP4A8 MoE GEMM shapes."
    )
    parser.add_argument("--tp-sizes", type=parse_int_list, default=DEFAULT_TP_SIZES)
    parser.add_argument("--global-tokens", type=parse_int_list, default=DEFAULT_GLOBAL_TOKENS)
    parser.add_argument("--legs", type=parse_leg_list, default=["w13", "w2"])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    output_dir = args.output_dir or make_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str | bool]] = []
    for tp_size in args.tp_sizes:
        leg_specs = leg_specs_for_tp(tp_size)
        for leg_name in args.legs:
            rows.extend(
                benchmark_one_group(
                    tp_size=tp_size,
                    leg=leg_specs[leg_name],
                    global_tokens=args.global_tokens,
                    output_dir=output_dir,
                )
            )

    validate_rows(rows)
    write_csv(rows, output_dir / "dsv4_flash_mxfp4a8_moe.csv")
    combined = combined_rows(rows)
    write_summary(output_file=output_dir / "summary.md", rows=rows, combined=combined)

    print(f"[done] output_dir={output_dir}")
    print(f"[done] csv={output_dir / 'dsv4_flash_mxfp4a8_moe.csv'}")
    print(f"[done] summary={output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
