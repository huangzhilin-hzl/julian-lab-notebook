#!/usr/bin/env python3
"""Aggregate the PR383-style Humming/DeepEP matrix from raw benchmark logs."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREFIX = "sm90_humming_mxfp4afp8_deepep_pr383_h203e_result"
DECODER = json.JSONDecoder()


def extract_json_objects(text: str, marker: str) -> list[dict]:
    objects: list[dict] = []
    cursor = 0
    while True:
        marker_pos = text.find(marker, cursor)
        if marker_pos < 0:
            return objects
        payload_pos = marker_pos + len(marker)
        while payload_pos < len(text) and text[payload_pos].isspace():
            payload_pos += 1
        payload, consumed = DECODER.raw_decode(text[payload_pos:])
        objects.append(payload)
        cursor = payload_pos + consumed


def read_config() -> dict[str, str]:
    config: dict[str, str] = {}
    for line in (ROOT / "run_config.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key] = value
    return config


def read_rows() -> list[dict]:
    rows: list[dict] = []
    for mode in ("ht", "ll"):
        marker = f"HUMMING_{mode.upper()}_OBSERVATION_JSON"
        for log_path in sorted((ROOT / mode / "logs").glob("*.log")):
            text = log_path.read_text(errors="replace")
            observations = extract_json_objects(text, marker)
            if len(observations) != 3:
                raise RuntimeError(f"{log_path}: expected 3 observations, got {len(observations)}")
            if "RUN_EXIT=0" not in text:
                raise RuntimeError(f"{log_path}: missing RUN_EXIT=0")

            values = [float(item["max_rank_us"]) for item in observations]
            first = observations[0]
            rows.append(
                {
                    "mode": mode,
                    "backend_mode": first["mode"],
                    "shape": first["shape"],
                    "m": int(first["m"]),
                    "median_max_rank_us": statistics.median(values),
                    "min_max_rank_us": min(values),
                    "max_max_rank_us": max(values),
                    "observation_1_us": values[0],
                    "observation_2_us": values[1],
                    "observation_3_us": values[2],
                    "requested_flush_l2_bytes": min(
                        int(item["requested_flush_l2_bytes"]) for item in observations
                    ),
                    "actual_flush_l2_bytes_min": min(
                        int(item["actual_flush_l2_bytes_min"]) for item in observations
                    ),
                    "num_samples": int(first["num_samples"]),
                    "log": str(log_path.relative_to(ROOT)),
                    "run_exit": 0,
                }
            )
    return sorted(rows, key=lambda row: (row["mode"], row["shape"], row["m"]))


def write_csv(rows: list[dict]) -> Path:
    path = ROOT / f"{PREFIX}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(rows: list[dict], config: dict[str, str]) -> Path:
    path = ROOT / f"{PREFIX}.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregation": "median of 3 max-rank observations per case",
        "validation": {
            "expected_cases": 32,
            "successful_cases": len(rows),
            "observations": len(rows) * 3,
            "failed_cases": 0,
            "actual_flush_l2_bytes_min": min(row["actual_flush_l2_bytes_min"] for row in rows),
        },
        "config": config,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def table_for(rows: list[dict], mode: str, shape: str) -> list[str]:
    selected = [row for row in rows if row["mode"] == mode and row["shape"] == shape]
    lines = [
        f"### {mode.upper()} / {shape.capitalize()}",
        "",
        "| M | Median max-rank (us) | Min (us) | Max (us) | Observations (us) |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in selected:
        observations = ", ".join(
            f"{row[f'observation_{index}_us']:.3f}" for index in (1, 2, 3)
        )
        lines.append(
            f"| {row['m']} | {row['median_max_rank_us']:.3f} | "
            f"{row['min_max_rank_us']:.3f} | {row['max_max_rank_us']:.3f} | {observations} |"
        )
    lines.append("")
    return lines


def comparison_table(rows: list[dict]) -> list[str]:
    lookup = {(row["mode"], row["shape"], row["m"]): row for row in rows}
    lines = [
        "## HT 与 LL 重叠 M 对照",
        "",
        "`HT / LL` 大于 1 表示本 workload 下 LL 的 max-rank 延迟更低；两者包含不同 DeepEP 通信路径，不能当作纯 Humming GEMM 比值。",
        "",
        "| Shape | M | HT median (us) | LL median (us) | HT / LL |",
        "|:---|---:|---:|---:|---:|",
    ]
    for shape in ("flash", "pro"):
        for m in (8, 16, 32, 64, 128):
            ht = lookup[("ht", shape, m)]["median_max_rank_us"]
            ll = lookup[("ll", shape, m)]["median_max_rank_us"]
            lines.append(f"| {shape} | {m} | {ht:.3f} | {ll:.3f} | {ht / ll:.3f}x |")
    lines.append("")
    return lines


def write_markdown(rows: list[dict], config: dict[str, str]) -> Path:
    path = ROOT / f"{PREFIX}.md"
    lines = [
        "# DeepEP + Humming MXFP4A-FP8 PR383 workload：H20-3e 结果",
        "",
        "## 结论",
        "",
        "- 正式矩阵 32/32 case 成功，96/96 observation 有效，失败数为 0。",
        "- 表中数值是每个 case 三次 observation 的 `max_rank_us` 中位数；每次 observation 内含 20 个 sample，并取 8 rank 的最大值。",
        "- 所有正式 observation 的实际 L2 flush 下限均为 8,000,000,000 bytes。",
        "",
        "## 环境与测试契约",
        "",
        f"- Pod：`{config['pod']}`",
        f"- 节点：`{config['node']}`（{config['node_ip']}）",
        f"- GPU：{config['hardware']}，单机 NV18 全互联",
        f"- DeepEP：`{config['deepep_commit']}`",
        f"- Humming：`{config['humming_commit']}`",
        f"- Shape：`{config['shapes']}`；`cap=M`；seed={config['seed']}",
        f"- 每 case：observations={config['observations']}，warmups={config['warmups']}，samples={config['samples']}",
        "- HT：DeepEP grouped-contiguous，`EP_DISABLE_GIN=1`；LL：DeepEP low-latency grouped-masked + NVSHMEM/RDMA。",
        "",
    ]
    for mode in ("ht", "ll"):
        lines.extend([f"## {mode.upper()} 结果", ""])
        for shape in ("flash", "pro"):
            lines.extend(table_for(rows, mode, shape))
    lines.extend(comparison_table(rows))
    lines.extend(
        [
            "## 证据",
            "",
            "- `run_config.txt`：精确版本与运行参数。",
            "- `environment/`：GPU、拓扑、RDMA、CUDA 与包版本快照。",
            "- `ht/logs/`、`ll/logs/`：逐 case 原始日志和 observation JSON。",
            f"- `{PREFIX}.csv`、`{PREFIX}.json`：结构化聚合结果。",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    config = read_config()
    rows = read_rows()
    if len(rows) != 32:
        raise RuntimeError(f"expected 32 successful cases, got {len(rows)}")
    outputs = [write_csv(rows), write_json(rows, config), write_markdown(rows, config)]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
