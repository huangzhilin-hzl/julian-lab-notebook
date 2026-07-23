#!/usr/bin/env python3
"""Benchmark one fused-MoE implementation for one external workload.

The benchmark intentionally has no built-in model shapes. Each invocation reads
one JSON workload and runs one backend for one local-rank fused-MoE problem. The
quantization contract is selected by each workload through a provider registry:

* Humming: StandardDispatchOutput -> SGLang MoeRunner(HUMMING) -> final output.
* FlashInfer: provider-specific cutlass_fused_moe path -> final output.
* SGLang CUTLASS: native tensorwise-FP8 W4A8 fused-MoE path -> final output.
* SGLang Marlin: StandardDispatchOutput -> MoeRunner(MARLIN) -> final output.

Model-independent input generation, weight preprocessing, compilation and
autotuning happen outside the timed region. Route-dependent adapter work stays
inside the backend callable. Callers invoke the script separately for every
backend; the script does not orchestrate or compare multiple runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable


torch: Any

SUPPORTED_ROUTING_MODES = ("balanced", "random", "trace")
BACKENDS = ("humming", "flashinfer", "sglang_cutlass", "sglang_marlin")
BENCHMARK_SCOPE = "local_rank_fused_moe_no_router_no_comm"


@dataclasses.dataclass(frozen=True)
class BackendImplementation:
    validate_workload: Callable[[Any], None]
    validate_device: Callable[[Any, Any], None]
    make_state: Callable[[Any, Any], dict[str, Any]]
    make_call: Callable[..., Callable[[], Any]]
    autotune: bool = False


@dataclasses.dataclass(frozen=True)
class QuantProvider:
    mode: str
    aliases: tuple[str, ...]
    contract_name: str
    comparison_semantics: str
    normalize_params: Callable[[dict[str, Any]], dict[str, Any]]
    validate_workload: Callable[[Any], None]
    implementations: dict[str, BackendImplementation]


QUANT_PROVIDERS: dict[str, QuantProvider] = {}
QUANT_MODE_ALIASES: dict[str, str] = {}


def compact_quant_mode(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def json_integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a JSON integer")
    return value


def json_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    return float(value)


def register_quant_provider(provider: QuantProvider) -> None:
    if provider.mode in QUANT_PROVIDERS:
        raise RuntimeError(f"duplicate quant provider: {provider.mode}")
    QUANT_PROVIDERS[provider.mode] = provider
    for alias in (provider.mode, *provider.aliases):
        key = compact_quant_mode(alias)
        previous = QUANT_MODE_ALIASES.setdefault(key, provider.mode)
        if previous != provider.mode:
            raise RuntimeError(f"quant mode alias {alias!r} is ambiguous")


def get_quant_provider(value: str) -> QuantProvider:
    canonical = QUANT_MODE_ALIASES.get(compact_quant_mode(value))
    if canonical is None:
        supported = tuple(QUANT_PROVIDERS)
        raise ValueError(f"unsupported quant_mode={value!r}; supported={supported}")
    return QUANT_PROVIDERS[canonical]


def get_backend_implementation(
    provider: QuantProvider, backend: str
) -> BackendImplementation:
    implementation = provider.implementations.get(backend)
    if implementation is None:
        available = tuple(provider.implementations)
        raise ValueError(
            f"quant_mode={provider.mode!r} does not support backend={backend!r}; "
            f"available_backends={available}"
        )
    return implementation


def quant_contract_name(mode: str, params: dict[str, Any]) -> str:
    provider = get_quant_provider(mode)
    if provider.mode == "wint4_afp8":
        return (
            "hopper_signed_int4_bf16g"
            f"{int(params['weight_group_size'])}_x_fp8e4m3"
        )
    if provider.mode == "wint4_a16":
        return (
            "sglang_marlin_signed_int4_bf16g"
            f"{int(params['weight_group_size'])}_x_bfloat16"
        )
    return provider.contract_name


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_tokens: tuple[int, ...]
    quant_mode: str
    quant_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    top_k_scope: str = "local"
    routing: str = "balanced"
    routing_file: str | None = None
    weight_seed: int = 41
    input_seed: int = 42
    routing_seed: int = 43
    routed_scaling_factor: float | None = None
    swiglu_limit: float | None = None
    input_amplitude: float = 0.1
    use_fused_finalize: bool = True
    normalize_trace_weights: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Workload":
        accepted = {
            "name",
            "quant_mode",
            "quant_params",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "num_tokens",
            "tokens",
            "batches",
            "tp_size",
            "tp_rank",
            "ep_size",
            "ep_rank",
            "top_k_scope",
            "routing",
            "routing_file",
            "weight_seed",
            "input_seed",
            "routing_seed",
            "routed_scaling_factor",
            "swiglu_limit",
            "input_amplitude",
            "use_fused_finalize",
            "normalize_trace_weights",
        }
        unknown = sorted(set(value) - accepted)
        if unknown:
            raise ValueError(f"unknown workload fields: {', '.join(unknown)}")
        token_fields = [
            field for field in ("num_tokens", "tokens", "batches") if field in value
        ]
        if len(token_fields) != 1:
            raise ValueError(
                "workload requires exactly one of num_tokens/tokens/batches"
            )
        ep_size = json_integer(value.get("ep_size", 1), "ep_size")
        if ep_size > 1 and value.get("top_k_scope") != "local":
            raise ValueError(
                "ep_size > 1 requires explicit top_k_scope='local'; "
                "this benchmark does not interpret model-global top-k"
            )
        tokens = value.get("num_tokens", value.get("tokens", value.get("batches")))
        if type(tokens) is int:
            tokens = [tokens]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("workload requires non-empty num_tokens/tokens/batches")
        normalized_tokens = tuple(
            json_integer(token, f"{token_fields[0]}[{index}]")
            for index, token in enumerate(tokens)
        )

        required = (
            "name",
            "quant_mode",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"workload is missing fields: {', '.join(missing)}")

        provider = get_quant_provider(str(value["quant_mode"]))
        raw_quant_params = value.get("quant_params", {})
        if not isinstance(raw_quant_params, dict):
            raise ValueError(f"{value['name']}: quant_params must be an object")
        quant_params = provider.normalize_params(dict(raw_quant_params))
        use_fused_finalize = value.get("use_fused_finalize", True)
        normalize_trace_weights = value.get("normalize_trace_weights", False)
        for field_name, field_value in (
            ("use_fused_finalize", use_fused_finalize),
            ("normalize_trace_weights", normalize_trace_weights),
        ):
            if not isinstance(field_value, bool):
                raise ValueError(
                    f"{value['name']}: {field_name} must be a JSON boolean"
                )

        workload = cls(
            name=str(value["name"]),
            hidden_size=json_integer(value["hidden_size"], "hidden_size"),
            intermediate_size=json_integer(
                value["intermediate_size"], "intermediate_size"
            ),
            num_experts=json_integer(value["num_experts"], "num_experts"),
            top_k=json_integer(value["top_k"], "top_k"),
            num_tokens=normalized_tokens,
            quant_mode=provider.mode,
            quant_params=quant_params,
            tp_size=json_integer(value.get("tp_size", 1), "tp_size"),
            tp_rank=json_integer(value.get("tp_rank", 0), "tp_rank"),
            ep_size=ep_size,
            ep_rank=json_integer(value.get("ep_rank", 0), "ep_rank"),
            top_k_scope=str(value.get("top_k_scope", "local")),
            routing=str(value.get("routing", "balanced")),
            routing_file=value.get("routing_file"),
            weight_seed=json_integer(value.get("weight_seed", 41), "weight_seed"),
            input_seed=json_integer(value.get("input_seed", 42), "input_seed"),
            routing_seed=json_integer(
                value.get("routing_seed", 43), "routing_seed"
            ),
            routed_scaling_factor=(
                None
                if value.get("routed_scaling_factor") is None
                else json_number(
                    value["routed_scaling_factor"], "routed_scaling_factor"
                )
            ),
            swiglu_limit=(
                None
                if value.get("swiglu_limit") is None
                else json_number(value["swiglu_limit"], "swiglu_limit")
            ),
            input_amplitude=json_number(
                value.get("input_amplitude", 0.1), "input_amplitude"
            ),
            use_fused_finalize=use_fused_finalize,
            normalize_trace_weights=normalize_trace_weights,
        )
        workload.validate()
        return workload

    @property
    def local_intermediate_size(self) -> int:
        return self.intermediate_size // self.tp_size

    @property
    def local_num_experts(self) -> int:
        return self.num_experts // self.ep_size

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{self.name}: {name} must be positive, got {value}")
        if any(m <= 0 for m in self.num_tokens):
            raise ValueError(f"{self.name}: every num_tokens value must be positive")
        finite_values = {
            "input_amplitude": self.input_amplitude,
            "routed_scaling_factor": self.routed_scaling_factor,
            "swiglu_limit": self.swiglu_limit,
        }
        for name, value in finite_values.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{self.name}: {name} must be finite")
        provider = get_quant_provider(self.quant_mode)
        if self.routing not in SUPPORTED_ROUTING_MODES:
            raise ValueError(
                f"{self.name}: unsupported routing={self.routing!r}; "
                f"supported={SUPPORTED_ROUTING_MODES}"
            )
        if self.top_k_scope != "local":
            raise ValueError(f"{self.name}: only top_k_scope='local' is supported")
        if self.routing == "trace" and not self.routing_file:
            raise ValueError(f"{self.name}: routing=trace requires routing_file")
        if self.intermediate_size % self.tp_size:
            raise ValueError(
                f"{self.name}: intermediate_size must be divisible by tp_size"
            )
        if self.num_experts % self.ep_size:
            raise ValueError(f"{self.name}: num_experts must be divisible by ep_size")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"{self.name}: tp_rank is outside [0, tp_size)")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"{self.name}: ep_rank is outside [0, ep_size)")
        if self.top_k > self.local_num_experts:
            raise ValueError(
                f"{self.name}: this local-rank benchmark requires top_k <= "
                f"local_num_experts ({self.local_num_experts})"
            )
        provider.validate_workload(self)

    def public_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        provider = get_quant_provider(self.quant_mode)
        value["num_tokens"] = list(self.num_tokens)
        value["local_intermediate_size"] = self.local_intermediate_size
        value["local_num_experts"] = self.local_num_experts
        value["quant_contract"] = quant_contract_name(
            self.quant_mode, self.quant_params
        )
        value["comparison_semantics"] = provider.comparison_semantics
        value["cross_backend_contract_matched"] = (
            provider.comparison_semantics == "contract_matched"
        )
        return value


def load_workload(path: Path) -> Workload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "workloads" in payload or "cases" in payload:
        raise ValueError("workload file must contain exactly one JSON object")
    item = dict(payload)
    routing_file = item.get("routing_file")
    if routing_file and not Path(routing_file).is_absolute():
        item["routing_file"] = str((path.parent / routing_file).resolve())
    return Workload.from_dict(item)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize_times(values: Iterable[float], prefix: str) -> dict[str, float]:
    samples = [float(x) for x in values]
    if not samples:
        return {}
    return {
        f"{prefix}_min_ms": min(samples),
        f"{prefix}_p20_ms": percentile(samples, 0.20),
        f"{prefix}_p50_ms": percentile(samples, 0.50),
        f"{prefix}_p80_ms": percentile(samples, 0.80),
        f"{prefix}_p90_ms": percentile(samples, 0.90),
        f"{prefix}_mean_ms": statistics.fmean(samples),
        f"{prefix}_std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def random_uint8(shape: tuple[int, ...], seed: int, device: Any):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, 256, shape, dtype=torch.uint8, device=device, generator=generator
    ).contiguous()


def random_e8m0_bytes(shape: tuple[int, ...], seed: int, device: Any):
    generator = torch.Generator(device=device).manual_seed(seed)
    # A conservative exponent range keeps large synthetic GEMMs finite.
    return torch.randint(
        114, 128, shape, dtype=torch.uint8, device=device, generator=generator
    ).contiguous()


def random_bfloat16_scales(
    shape: tuple[int, ...],
    seed: int,
    amplitude: float,
    device: Any,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    # Positive, nonzero scales model symmetric group quantization and avoid
    # synthetic cancellation from randomly signed scale tensors.
    values = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return (values * (amplitude * 0.875) + amplitude * 0.125).to(torch.bfloat16)


def normalize_dtype_name(value: Any) -> str:
    key = compact_quant_mode(str(value))
    aliases = {
        "float8e4m3": "float8e4m3",
        "float8e4m3fn": "float8e4m3",
        "fp8e4m3": "float8e4m3",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float8e8m0": "float8e8m0",
        "float8e8m0fnu": "float8e8m0",
        "e8m0": "float8e8m0",
        "float32": "float32",
        "fp32": "float32",
    }
    if key not in aliases:
        raise ValueError(f"unsupported quantization dtype {value!r}")
    return aliases[key]


def normalize_quant_params(
    raw: dict[str, Any],
    defaults: dict[str, Any],
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    aliases = aliases or {}
    normalized: dict[str, Any] = {}
    for original_name, value in raw.items():
        name = aliases.get(original_name, original_name)
        if name not in defaults:
            raise ValueError(f"unknown quant_params field: {original_name}")
        if name in normalized:
            raise ValueError(f"quant_params field supplied more than once: {name}")
        normalized[name] = value
    return defaults | normalized


def positive_finite_float(params: dict[str, Any], name: str) -> float:
    value = json_number(params[name], f"quant_params.{name}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"quant_params.{name} must be finite and positive")
    return value


def json_boolean(params: dict[str, Any], name: str) -> bool:
    value = params[name]
    if not isinstance(value, bool):
        raise ValueError(f"quant_params.{name} must be a JSON boolean")
    return value


def normalize_mxfp4_fp8_params(raw: dict[str, Any]) -> dict[str, Any]:
    params = normalize_quant_params(
        raw,
        defaults={
            "weight_group_size": 32,
            "weight_scale_dtype": "float8e8m0",
            "activation_dtype": "float8e4m3",
            "input_scale_group_size": 0,
            "input_scale_dtype": "float32",
        },
        aliases={"group_size": "weight_group_size"},
    )
    params["weight_group_size"] = json_integer(
        params["weight_group_size"], "quant_params.weight_group_size"
    )
    params["input_scale_group_size"] = json_integer(
        params["input_scale_group_size"], "quant_params.input_scale_group_size"
    )
    params["weight_scale_dtype"] = normalize_dtype_name(params["weight_scale_dtype"])
    params["activation_dtype"] = normalize_dtype_name(params["activation_dtype"])
    params["input_scale_dtype"] = normalize_dtype_name(params["input_scale_dtype"])
    return params


def normalize_wint4_afp8_params(raw: dict[str, Any]) -> dict[str, Any]:
    params = normalize_quant_params(
        raw,
        defaults={
            "weight_group_size": 128,
            "weight_scale_dtype": "bfloat16",
            "activation_dtype": "float8e4m3",
            "input_scale_group_size": 0,
            "input_scale_dtype": "float32",
            "weight_scale_amplitude": 0.005,
            "fc1_input_scale": 0.002,
            "fc2_input_scale": 0.0001,
            "fc1_prequant_scale": 1.0,
            "fc2_prequant_scale": 1.0,
            "weight_scale_2": 1.0,
        },
        aliases={"group_size": "weight_group_size"},
    )
    params["weight_group_size"] = json_integer(
        params["weight_group_size"], "quant_params.weight_group_size"
    )
    params["input_scale_group_size"] = json_integer(
        params["input_scale_group_size"], "quant_params.input_scale_group_size"
    )
    params["weight_scale_dtype"] = normalize_dtype_name(params["weight_scale_dtype"])
    params["activation_dtype"] = normalize_dtype_name(params["activation_dtype"])
    params["input_scale_dtype"] = normalize_dtype_name(params["input_scale_dtype"])
    for name in (
        "weight_scale_amplitude",
        "fc1_input_scale",
        "fc2_input_scale",
        "fc1_prequant_scale",
        "fc2_prequant_scale",
        "weight_scale_2",
    ):
        params[name] = positive_finite_float(params, name)
    return params


def normalize_wint4_a16_params(raw: dict[str, Any]) -> dict[str, Any]:
    params = normalize_quant_params(
        raw,
        defaults={
            "weight_group_size": 128,
            "weight_scale_dtype": "bfloat16",
            "activation_dtype": "bfloat16",
            "weight_scale_amplitude": 0.005,
            "symmetric": True,
            "has_zero_point": False,
            "act_order": False,
        },
        aliases={"group_size": "weight_group_size"},
    )
    params["weight_group_size"] = json_integer(
        params["weight_group_size"], "quant_params.weight_group_size"
    )
    params["weight_scale_dtype"] = normalize_dtype_name(params["weight_scale_dtype"])
    params["activation_dtype"] = normalize_dtype_name(params["activation_dtype"])
    params["weight_scale_amplitude"] = positive_finite_float(
        params, "weight_scale_amplitude"
    )
    for name in ("symmetric", "has_zero_point", "act_order"):
        params[name] = json_boolean(params, name)
    return params


def make_raw_mxfp4(workload: Workload, order: str, device: Any):
    """Create the same logical gate/up/down weights in backend-native W13 order."""
    e = workload.local_num_experts
    n = workload.local_intermediate_size
    k = workload.hidden_size
    seed = workload.weight_seed

    gate = random_uint8((e, n, k // 2), seed + 11, device)
    up = random_uint8((e, n, k // 2), seed + 13, device)
    gate_scale = random_e8m0_bytes((e, n, k // 32), seed + 17, device)
    up_scale = random_e8m0_bytes((e, n, k // 32), seed + 19, device)
    if order == "gate_up":
        w13 = torch.cat((gate, up), dim=1)
        w13_scale = torch.cat((gate_scale, up_scale), dim=1)
    elif order == "up_gate":
        w13 = torch.cat((up, gate), dim=1)
        w13_scale = torch.cat((up_scale, gate_scale), dim=1)
    else:
        raise ValueError(f"unknown W13 order: {order}")
    del gate, up, gate_scale, up_scale

    w2 = random_uint8((e, k, n // 2), seed + 23, device)
    w2_scale = random_e8m0_bytes((e, k, n // 32), seed + 29, device)
    return w13, w13_scale, w2, w2_scale


def convert_centered_u4_encoding(packed: Any, encoding: str):
    """Encode logical values represented by centered uint4 bytes for a backend.

    Humming interprets each nibble as ``u4 - 8``. FlashInfer's W4A8 path
    interprets it as signed two's-complement INT4. Flipping bit 3 in every
    nibble converts between the two without materializing unpacked weights.
    """
    if encoding == "humming_centered_uint4":
        return packed
    if encoding == "flashinfer_signed_int4":
        return torch.bitwise_xor(packed, 0x88)
    raise ValueError(f"unknown INT4 encoding: {encoding}")


def make_raw_wint4(
    workload: Workload,
    order: str,
    encoding: str,
    device: Any,
):
    """Create identical logical signed-INT4 weights in backend-native encodings."""
    e = workload.local_num_experts
    n = workload.local_intermediate_size
    k = workload.hidden_size
    seed = workload.weight_seed
    group_size = int(workload.quant_params["weight_group_size"])
    amplitude = float(workload.quant_params["weight_scale_amplitude"])

    gate = random_uint8((e, n, k // 2), seed + 11, device)
    up = random_uint8((e, n, k // 2), seed + 13, device)
    gate_scale = random_bfloat16_scales(
        (e, n, k // group_size), seed + 17, amplitude, device
    )
    up_scale = random_bfloat16_scales(
        (e, n, k // group_size), seed + 19, amplitude, device
    )
    if order == "gate_up":
        w13 = torch.cat((gate, up), dim=1)
        w13_scale = torch.cat((gate_scale, up_scale), dim=1)
    elif order == "up_gate":
        w13 = torch.cat((up, gate), dim=1)
        w13_scale = torch.cat((up_scale, gate_scale), dim=1)
    else:
        raise ValueError(f"unknown W13 order: {order}")
    del gate, up, gate_scale, up_scale

    w2 = random_uint8((e, k, n // 2), seed + 23, device)
    w2_scale = random_bfloat16_scales(
        (e, k, n // group_size), seed + 29, amplitude, device
    )
    return (
        convert_centered_u4_encoding(w13, encoding),
        w13_scale,
        convert_centered_u4_encoding(w2, encoding),
        w2_scale,
    )


def make_routing(workload: Workload, m: int, device: Any):
    e = workload.local_num_experts
    top_k = workload.top_k
    seed = workload.routing_seed + m * 65537

    trace_weights = None
    if workload.routing == "trace":
        assert workload.routing_file is not None
        trace_path = Path(workload.routing_file)
        if trace_path.suffix == ".npz":
            import numpy as np

            payload = np.load(trace_path)
            ids = torch.as_tensor(payload["topk_ids"], device=device)
            if "topk_weights" in payload:
                trace_weights = torch.as_tensor(payload["topk_weights"], device=device)
        else:
            try:
                payload = torch.load(trace_path, map_location=device, weights_only=True)
            except TypeError:
                payload = torch.load(trace_path, map_location=device)
            if isinstance(payload, dict):
                ids = payload["topk_ids"]
                trace_weights = payload.get("topk_weights")
            else:
                ids = payload
            ids = ids.to(device=device)
            if trace_weights is not None:
                trace_weights = trace_weights.to(device=device)
        if ids.shape[0] < m:
            raise ValueError(
                f"{workload.name}: routing trace has {ids.shape[0]} rows, needs {m}"
            )
        ids = ids[:m, :top_k].to(torch.int32).contiguous()
    elif workload.routing == "balanced":
        base = torch.arange(m, dtype=torch.int64, device=device).unsqueeze(1) * top_k
        offsets = torch.arange(top_k, dtype=torch.int64, device=device).unsqueeze(0)
        ids = ((base + offsets) % e).to(torch.int32).contiguous()
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        scores = torch.randn(
            m, e, dtype=torch.float32, device=device, generator=generator
        )
        ids = torch.topk(scores, top_k, dim=-1).indices.to(torch.int32).contiguous()
        del scores

    if ids.ndim != 2 or tuple(ids.shape) != (m, top_k):
        raise ValueError(
            f"{workload.name}: topk_ids must have shape {(m, top_k)}, got {tuple(ids.shape)}"
        )
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= e):
        raise ValueError(f"{workload.name}: routing IDs must be local IDs in [0, {e})")
    if top_k > 1:
        sorted_ids = torch.sort(ids, dim=-1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()):
            raise ValueError(
                f"{workload.name}: every token must route to distinct experts"
            )

    if trace_weights is not None:
        weights = trace_weights[:m, :top_k].to(torch.float32).contiguous()
        if tuple(weights.shape) != (m, top_k):
            raise ValueError(
                f"{workload.name}: topk_weights must have shape {(m, top_k)}, "
                f"got {tuple(weights.shape)}"
            )
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError(f"{workload.name}: topk_weights must be finite")
        if workload.normalize_trace_weights:
            weight_sums = weights.sum(dim=-1, keepdim=True)
            if bool((weight_sums == 0).any().item()):
                raise ValueError(
                    f"{workload.name}: normalized trace weights must sum nonzero"
                )
            weights = weights / weight_sums
    else:
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        weights = torch.rand(
            m, top_k, dtype=torch.float32, device=device, generator=generator
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if workload.routed_scaling_factor is not None:
        weights = weights * workload.routed_scaling_factor
    return ids, weights.contiguous()


def route_statistics(ids: Any, local_experts: int) -> dict[str, Any]:
    counts = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=local_experts)
    active = counts[counts > 0]
    mean = float(counts.float().mean().item())
    return {
        "active_experts": int(active.numel()),
        "min_tokens_per_active_expert": (
            int(active.min().item()) if active.numel() else 0
        ),
        "max_tokens_per_active_expert": (
            int(active.max().item()) if active.numel() else 0
        ),
        "max_over_mean_tokens": (float(counts.max().item()) / mean if mean else 0.0),
    }


def make_hidden_states(workload: Workload, m: int, device: Any):
    generator = torch.Generator(device=device).manual_seed(
        workload.input_seed + m * 1009
    )
    return (
        torch.randn(
            m,
            workload.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * workload.input_amplitude
    ).contiguous()


def make_humming_runner_state(
    workload: Workload,
    device: Any,
    parameters: dict[str, Any],
    weight_quant_config: dict[str, Any],
    expected_weight_dtype: str,
):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import HummingMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.quantization.humming_utils import prepare_humming_moe_layer

    # The benchmark has no dispatcher/communication.  Force the same `none`
    # fused-op lookup without initializing the complete SGLang server stack.
    import sglang.srt.layers.moe.moe_runner.runner as runner_module

    class SyntheticFusedMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_name = f"bench.{workload.name}.experts"
            self.hidden_size = workload.hidden_size
            self.intermediate_size_per_partition = workload.local_intermediate_size
            self.num_experts = workload.num_experts
            self.num_local_experts = workload.local_num_experts
            self.params_dtype = torch.bfloat16
            self.with_bias = False

    layer = SyntheticFusedMoe().to(device)
    for name, tensor in parameters.items():
        layer.register_parameter(
            name,
            torch.nn.Parameter(tensor, requires_grad=False),
        )
    prepare_humming_moe_layer(layer, weight_quant_config)

    for sublayer in ("w13", "w2"):
        actual_activation = str(layer.humming_metas[sublayer].a_dtype)
        expected_activation = str(workload.quant_params["activation_dtype"])
        if actual_activation != expected_activation:
            raise RuntimeError(
                f"Humming {sublayer} activation dtype is {actual_activation}, "
                f"expected {expected_activation}"
            )
        actual_weight = str(layer.humming_metas[sublayer].b_dtype)
        if actual_weight != expected_weight_dtype:
            raise RuntimeError(
                f"Humming {sublayer} weight dtype is {actual_weight}, "
                f"expected {expected_weight_dtype}"
            )

    runner_config = MoeRunnerConfig(
        num_experts=workload.num_experts,
        num_local_experts=workload.local_num_experts,
        hidden_size=workload.hidden_size,
        intermediate_size_per_partition=workload.local_intermediate_size,
        layer_id=0,
        top_k=workload.top_k,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=False,
        routed_scaling_factor=None,
        swiglu_limit=workload.swiglu_limit,
        is_gated=True,
        layer=layer,
    )
    original_get_moe_a2a_backend = runner_module.get_moe_a2a_backend
    try:
        runner_module.get_moe_a2a_backend = lambda: MoeA2ABackend.NONE
        runner = MoeRunner(MoeRunnerBackend.HUMMING, runner_config)
    finally:
        runner_module.get_moe_a2a_backend = original_get_moe_a2a_backend
    if runner.fused_func is None:
        raise RuntimeError("SGLang did not register the none+humming fused function")
    quant_info_fields = {
        field.name for field in dataclasses.fields(HummingMoeQuantInfo)
    }
    quant_info = (
        HummingMoeQuantInfo(layer=layer)
        if "layer" in quant_info_fields
        else HummingMoeQuantInfo()
    )
    return {"layer": layer, "runner": runner, "quant_info": quant_info}


def make_mxfp4_humming_state(workload: Workload, device: Any):
    w13, s13, w2, s2 = make_raw_mxfp4(workload, "gate_up", device)
    return make_humming_runner_state(
        workload=workload,
        device=device,
        parameters={
            "w13_weight": w13,
            "w13_weight_scale": s13.view(torch.float8_e8m0fnu),
            "w2_weight": w2,
            "w2_weight_scale": s2.view(torch.float8_e8m0fnu),
        },
        weight_quant_config={"quant_method": "mxfp4"},
        expected_weight_dtype="float4e2m1",
    )


def make_wint4_humming_state(workload: Workload, device: Any):
    w13, s13, w2, s2 = make_raw_wint4(
        workload,
        order="gate_up",
        encoding="humming_centered_uint4",
        device=device,
    )
    group_size = int(workload.quant_params["weight_group_size"])
    return make_humming_runner_state(
        workload=workload,
        device=device,
        parameters={
            "w13_weight": w13.contiguous().view(torch.int32),
            "w13_weight_scale": s13,
            "w2_weight": w2.contiguous().view(torch.int32),
            "w2_weight_scale": s2,
        },
        weight_quant_config={
            "quant_method": "humming",
            "dtype": "int4",
            "group_size": group_size,
            "scale_dtype": "bfloat16",
            "scale_type": "group",
            "has_zero_point": False,
        },
        expected_weight_dtype="uint4",
    )


def make_wint4_sglang_cutlass_state(workload: Workload, device: Any):
    from sglang.srt.layers.quantization.w4afp8 import interleave_scales

    w13, s13, w2, s2 = make_raw_wint4(
        workload,
        order="gate_up",
        encoding="flashinfer_signed_int4",
        device=device,
    )
    e = workload.local_num_experts
    n = workload.local_intermediate_size
    k = workload.hidden_size
    a_strides1 = torch.full((e, 3), k, dtype=torch.int64, device=device)
    c_strides1 = torch.full((e, 3), 2 * n, dtype=torch.int64, device=device)
    a_strides2 = torch.full((e, 3), n, dtype=torch.int64, device=device)
    c_strides2 = torch.full((e, 3), k, dtype=torch.int64, device=device)

    state = {
        "w13": w13.contiguous().view(torch.int8),
        "s13": interleave_scales(s13).contiguous(),
        "w2": w2.contiguous().view(torch.int8),
        "s2": interleave_scales(s2).contiguous(),
        "a1_scale": torch.tensor(
            [float(workload.quant_params["fc1_input_scale"])],
            dtype=torch.float32,
            device=device,
        ),
        "a2_scale": torch.tensor(
            [float(workload.quant_params["fc2_input_scale"])],
            dtype=torch.float32,
            device=device,
        ),
        "a_strides1": a_strides1,
        "b_strides1": a_strides1,
        "c_strides1": c_strides1,
        "s_strides13": c_strides1,
        "a_strides2": a_strides2,
        "b_strides2": a_strides2,
        "c_strides2": c_strides2,
        "s_strides2": c_strides2,
        "expert_offsets": torch.empty(e + 1, dtype=torch.int32, device=device),
        "problem_sizes1": torch.empty((e, 3), dtype=torch.int32, device=device),
        "problem_sizes2": torch.empty((e, 3), dtype=torch.int32, device=device),
    }

    # This standalone process does not initialize SGLang distributed state.
    # Newer SGLang exposes a scoped runtime override; older releases import the
    # legacy getter into the CUTLASS module, so patch and restore that one symbol.
    parallel_override = None
    legacy_parallel_getter = None
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel_override = get_parallel().override(
            moe_ep_size=workload.ep_size,
            moe_ep_rank=workload.ep_rank,
        )
        parallel_override.__enter__()
    except ImportError:
        import sglang.srt.layers.moe.cutlass_w4a8_moe as cutlass_module

        legacy_parallel_getter = cutlass_module.get_moe_expert_parallel_world_size
        cutlass_module.get_moe_expert_parallel_world_size = lambda: workload.ep_size

    if parallel_override is not None:
        state["_parallel_override"] = parallel_override
    if legacy_parallel_getter is not None:
        state["_legacy_parallel_getter"] = legacy_parallel_getter
    return state


def make_wint4_sglang_marlin_state(workload: Workload, device: Any):
    # Import marlin_utils before gptq_kernels. Older SGLang releases otherwise
    # enter a gptq_kernels <-> compressed_tensors package import cycle.
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

    import sglang.srt.layers.moe.moe_runner.runner as runner_module

    e = workload.local_num_experts
    n = workload.local_intermediate_size
    k = workload.hidden_size
    group_size = int(workload.quant_params["weight_group_size"])
    raw_w13, raw_s13, raw_w2, raw_s2 = make_raw_wint4(
        workload,
        order="gate_up",
        encoding="humming_centered_uint4",
        device=device,
    )

    # GPTQ packs eight K-contiguous INT4 values into one int32. Preserve that
    # adjacency before transposing to the [E, K/8, output] checkpoint layout.
    q13_gptq = raw_w13.contiguous().view(torch.int32).transpose(1, 2).contiguous()
    q2_gptq = raw_w2.contiguous().view(torch.int32).transpose(1, 2).contiguous()
    empty_permutation = torch.empty((e, 0), dtype=torch.int32, device=device)
    w13 = gptq_marlin_moe_repack(q13_gptq, empty_permutation, k, 2 * n, 4)
    w2 = gptq_marlin_moe_repack(q2_gptq, empty_permutation, n, k, 4)
    s13 = marlin_moe_permute_scales(
        raw_s13.transpose(1, 2).contiguous(),
        size_k=k,
        size_n=2 * n,
        group_size=group_size,
    )
    s2 = marlin_moe_permute_scales(
        raw_s2.transpose(1, 2).contiguous(),
        size_k=n,
        size_n=k,
        group_size=group_size,
    )
    del raw_w13, raw_s13, raw_w2, raw_s2, q13_gptq, q2_gptq
    torch.cuda.empty_cache()

    runner_config = MoeRunnerConfig(
        num_experts=workload.num_experts,
        num_local_experts=e,
        hidden_size=k,
        intermediate_size_per_partition=n,
        layer_id=0,
        top_k=workload.top_k,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=False,
        routed_scaling_factor=None,
        swiglu_limit=workload.swiglu_limit,
        is_gated=True,
    )
    original_get_moe_a2a_backend = runner_module.get_moe_a2a_backend
    try:
        runner_module.get_moe_a2a_backend = lambda: MoeA2ABackend.NONE
        runner = MoeRunner(MoeRunnerBackend.MARLIN, runner_config)
    finally:
        runner_module.get_moe_a2a_backend = original_get_moe_a2a_backend
    quant_info = MarlinMoeQuantInfo(
        w13_qweight=w13,
        w2_qweight=w2,
        w13_scales=s13,
        w2_scales=s2,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
        global_num_experts=e,
    )
    return {"runner": runner, "quant_info": quant_info}


def make_sglang_runner_call(
    workload: Workload,
    state: dict[str, Any],
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        # Router/top-k is deliberately outside this benchmark.
        router_logits=torch.empty(
            (hidden_states.shape[0], 0),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    def run():
        return state["runner"].run(dispatch_output, state["quant_info"]).hidden_states

    return run


def make_wint4_sglang_cutlass_call(
    workload: Workload,
    state: dict[str, Any],
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
):
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe

    del workload

    def run():
        return cutlass_w4a8_moe(
            hidden_states,
            state["w13"],
            state["w2"],
            state["s13"],
            state["s2"],
            topk_weights,
            topk_ids,
            state["a_strides1"],
            state["b_strides1"],
            state["c_strides1"],
            state["a_strides2"],
            state["b_strides2"],
            state["c_strides2"],
            state["s_strides13"],
            state["s_strides2"],
            state["expert_offsets"],
            state["problem_sizes1"],
            state["problem_sizes2"],
            a1_scale=state["a1_scale"],
            a2_scale=state["a2_scale"],
            apply_router_weight_on_input=False,
            # make_routing() has already folded this factor into topk_weights.
            routed_scaling_factor=1.0,
        )

    return run


def make_mxfp4_flashinfer_state(workload: Workload, device: Any):
    from flashinfer.fused_moe import preprocess_moe_weights_for_sm90_mixed_gemm_humming

    w13, s13, w2, s2 = make_raw_mxfp4(workload, "up_gate", device)
    w13_il, s13_il, residual13 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(
        w13, s13
    )
    w2_il, s2_il, residual2 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, s2)
    del w13, s13, w2, s2
    torch.cuda.empty_cache()
    return {
        "w13": w13_il,
        "s13": s13_il,
        "residual13": residual13,
        "w2": w2_il,
        "s2": s2_il,
        "residual2": residual2,
    }


def make_wint4_flashinfer_state(workload: Workload, device: Any):
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    group_size = int(workload.quant_params["weight_group_size"])
    w13, s13, w2, s2 = make_raw_wint4(
        workload,
        order="up_gate",
        encoding="flashinfer_signed_int4",
        device=device,
    )
    # The synthetic prequant values are scalar channel vectors. Fold their
    # inverse into the corresponding weight scales so the logical BF16 weight
    # function remains aligned with Humming when a value other than 1 is used.
    s13 = (s13.float() / float(workload.quant_params["fc1_prequant_scale"])).to(
        torch.bfloat16
    )
    s2 = (s2.float() / float(workload.quant_params["fc2_prequant_scale"])).to(
        torch.bfloat16
    )
    w13_il = interleave_moe_weights_for_sm90_mixed_gemm(w13, "int4")
    w2_il = interleave_moe_weights_for_sm90_mixed_gemm(w2, "int4")
    # FlashInfer's Hopper W4A8 kernel consumes BF16 scales in a folded layout.
    s13_il = interleave_moe_scales_for_sm90_mixed_gemm(s13, group_size)
    s2_il = interleave_moe_scales_for_sm90_mixed_gemm(s2, group_size)
    del w13, s13, w2, s2
    torch.cuda.empty_cache()
    return {"w13": w13_il, "s13": s13_il, "w2": w2_il, "s2": s2_il}


def expert_contiguous_scale(counts: Any, residual: Any, num_routes: int):
    return torch.repeat_interleave(
        residual * 64.0,
        counts,
        output_size=num_routes,
    ).contiguous()


def make_mxfp4_flashinfer_call(
    workload: Workload,
    state: dict[str, Any],
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
):
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe

    global_ids = topk_ids + workload.ep_rank * workload.local_num_experts
    fc2_act_global = torch.ones((), dtype=torch.float32, device=hidden_states.device)
    swiglu_limit = (
        torch.full(
            (workload.local_num_experts,),
            workload.swiglu_limit,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if workload.swiglu_limit is not None
        else None
    )
    output = torch.empty_like(hidden_states)

    def run():
        # FlashInfer's Humming-style ABI requires two route-dependent scale
        # vectors. Keep their construction in the measured callable, just as
        # SGLang's Humming path keeps its own routing/layout work inside run().
        counts = torch.bincount(
            topk_ids.reshape(-1).to(torch.int64),
            minlength=workload.local_num_experts,
        )
        quant_scales = [
            state["s13"].view(torch.int32),
            expert_contiguous_scale(counts, state["residual13"], topk_ids.numel()),
            fc2_act_global,
            state["s2"].view(torch.int32),
            expert_contiguous_scale(counts, state["residual2"], topk_ids.numel()),
        ]
        cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=global_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=state["w13"],
            fc2_expert_weights=state["w2"],
            output_dtype=torch.bfloat16,
            quant_scales=quant_scales,
            swiglu_limit=swiglu_limit,
            output=output,
            tp_size=workload.tp_size,
            tp_rank=workload.tp_rank,
            ep_size=workload.ep_size,
            ep_rank=workload.ep_rank,
            use_w4_group_scaling=True,
            use_wfp4afp8_humming=True,
            use_fused_finalize=workload.use_fused_finalize,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=next_power_of_two(hidden_states.shape[0]),
        )
        return output

    return run


def make_wint4_flashinfer_call(
    workload: Workload,
    state: dict[str, Any],
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
):
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe

    params = workload.quant_params
    global_ids = topk_ids + workload.ep_rank * workload.local_num_experts
    output = torch.empty_like(hidden_states)
    dtype = torch.bfloat16
    device = hidden_states.device
    e = workload.local_num_experts
    k = workload.hidden_size
    n = workload.local_intermediate_size

    fc1_input_scale = torch.tensor(
        float(params["fc1_input_scale"]), dtype=torch.float32, device=device
    )
    fc2_input_scale = torch.tensor(
        float(params["fc2_input_scale"]), dtype=torch.float32, device=device
    )
    fc1_prequant = torch.full(
        (k,), float(params["fc1_prequant_scale"]), dtype=dtype, device=device
    )
    fc2_prequant = torch.full(
        (n,), float(params["fc2_prequant_scale"]), dtype=dtype, device=device
    )
    weight_scale_2 = float(params["weight_scale_2"])
    fc1_act_scale = (fc1_prequant / fc1_input_scale).to(dtype)
    fc2_act_scale = (fc2_prequant / fc2_input_scale).to(dtype)
    fc1_alpha = torch.full(
        (e,),
        weight_scale_2 * float(params["fc1_input_scale"]),
        dtype=torch.float32,
        device=device,
    )
    fc2_alpha = torch.full(
        (e,),
        weight_scale_2 * float(params["fc2_input_scale"]),
        dtype=torch.float32,
        device=device,
    )
    empty_zero_1 = torch.empty(0, dtype=dtype, device=device)
    empty_zero_2 = torch.empty(0, dtype=dtype, device=device)
    quant_scales = (
        state["s13"],
        state["s2"],
        fc1_act_scale,
        fc2_act_scale,
        empty_zero_1,
        empty_zero_2,
        fc1_alpha,
        fc2_alpha,
    )
    swiglu_limit = (
        torch.full(
            (e,),
            workload.swiglu_limit,
            dtype=torch.float32,
            device=device,
        )
        if workload.swiglu_limit is not None
        else None
    )

    def run():
        cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=global_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=state["w13"],
            fc2_expert_weights=state["w2"],
            output_dtype=dtype,
            quant_scales=quant_scales,
            swiglu_limit=swiglu_limit,
            output=output,
            tp_size=workload.tp_size,
            tp_rank=workload.tp_rank,
            ep_size=workload.ep_size,
            ep_rank=workload.ep_rank,
            use_w4_group_scaling=True,
            use_packed_weights=True,
            use_wfp4afp8_humming=False,
            use_fused_finalize=workload.use_fused_finalize,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=next_power_of_two(hidden_states.shape[0]),
        )
        return output

    return run


def validate_common_hopper_shape(workload: Workload, contract: str) -> None:
    if workload.hidden_size % 128 or workload.local_intermediate_size % 128:
        raise ValueError(
            f"{workload.name}: {contract} requires K and local N to be multiples of 128"
        )


def validate_mxfp4_fp8_workload(workload: Workload) -> None:
    params = workload.quant_params
    validate_common_hopper_shape(workload, "Hopper MXFP4xFP8 fused MoE")
    expected = {
        "weight_group_size": 32,
        "weight_scale_dtype": "float8e8m0",
        "activation_dtype": "float8e4m3",
        "input_scale_group_size": 0,
        "input_scale_dtype": "float32",
    }
    mismatches = {
        key: (params[key], value)
        for key, value in expected.items()
        if params[key] != value
    }
    if mismatches:
        raise ValueError(
            f"{workload.name}: unsupported MXFP4xFP8 quant_params {mismatches}; "
            "the current provider implements E8M0 group-32 weights and "
            "tensorwise dynamic FP8 E4M3 inputs"
        )


def validate_wint4_afp8_workload(workload: Workload) -> None:
    params = workload.quant_params
    validate_common_hopper_shape(workload, "Hopper WINT4xAFP8 fused MoE")
    fixed = {
        "weight_scale_dtype": "bfloat16",
        "activation_dtype": "float8e4m3",
        "input_scale_dtype": "float32",
    }
    mismatches = {
        key: (params[key], value)
        for key, value in fixed.items()
        if params[key] != value
    }
    if mismatches:
        raise ValueError(
            f"{workload.name}: unsupported WINT4xAFP8 quant_params {mismatches}; "
            "the current provider requires BF16 weight scales and FP8 E4M3 "
            "activations"
        )
    weight_group = int(params["weight_group_size"])
    if (
        weight_group < 32
        or workload.hidden_size % weight_group
        or workload.local_intermediate_size % weight_group
    ):
        raise ValueError(
            f"{workload.name}: weight_group_size must be >= 32 and divide "
            "both K and local N"
        )
    input_group = int(params["input_scale_group_size"])
    if input_group < 0:
        raise ValueError(
            f"{workload.name}: input_scale_group_size must be non-negative"
        )
    if input_group and (
        input_group < 32
        or workload.hidden_size % input_group
        or workload.local_intermediate_size % input_group
    ):
        raise ValueError(
            f"{workload.name}: nonzero input_scale_group_size must be >= 32 "
            "and divide both K and local N"
        )


def validate_wint4_a16_workload(workload: Workload) -> None:
    params = workload.quant_params
    fixed = {
        "weight_scale_dtype": "bfloat16",
        "activation_dtype": "bfloat16",
        "symmetric": True,
        "has_zero_point": False,
        "act_order": False,
    }
    mismatches = {
        key: (params[key], value)
        for key, value in fixed.items()
        if params[key] != value
    }
    if mismatches:
        raise ValueError(
            f"{workload.name}: unsupported Marlin W4A16 quant_params "
            f"{mismatches}; only symmetric no-zero-point/no-act-order INT4 "
            "with BF16 activation and scales is implemented"
        )
    group_size = int(params["weight_group_size"])
    if group_size not in (32, 64, 128):
        raise ValueError(
            f"{workload.name}: Marlin W4A16 weight_group_size must be one of "
            f"(32, 64, 128), got {group_size}"
        )
    if workload.hidden_size % 128:
        raise ValueError(
            f"{workload.name}: Marlin W4A16 requires K to be a multiple of 128"
        )
    n_alignment = max(64, group_size)
    if workload.local_intermediate_size % n_alignment:
        raise ValueError(
            f"{workload.name}: Marlin W4A16 requires local N to be a multiple "
            f"of {n_alignment} for group_size={group_size}"
        )


def validate_backend_workload_noop(workload: Workload) -> None:
    del workload


def validate_flashinfer_wint4_workload(workload: Workload) -> None:
    group_size = int(workload.quant_params["weight_group_size"])
    if group_size != 128:
        raise ValueError(
            "FlashInfer's SM90 packed-INT4 fused-MoE kernel requires "
            f"weight_group_size=128; got {group_size}"
        )


def validate_sglang_cutlass_workload(workload: Workload) -> None:
    group_size = int(workload.quant_params["weight_group_size"])
    if group_size != 128:
        raise ValueError(
            "SGLang CUTLASS W4A8 fused MoE requires "
            f"weight_group_size=128; got {group_size}"
        )
    if int(workload.quant_params["input_scale_group_size"]) != 0:
        raise ValueError(
            "SGLang CUTLASS W4A8 uses tensorwise FP8 activation scales and "
            "quantization and requires input_scale_group_size=0"
        )
    if workload.swiglu_limit is not None:
        raise ValueError("SGLang CUTLASS W4A8 does not support swiglu_limit")
    # w4a8_get_group_starts.cuh launches one CUDA block with E threads.
    if workload.local_num_experts > 1024:
        raise ValueError(
            "SGLang CUTLASS W4A8 supports at most 1024 local experts; got "
            f"{workload.local_num_experts}"
        )


def cuda_arch(device: Any) -> tuple[tuple[int, int], int]:
    capability = torch.cuda.get_device_capability(device)
    return capability, capability[0] * 10 + capability[1]


def validate_mxfp4_humming_device(workload: Workload, device: Any) -> None:
    del workload
    _, arch = cuda_arch(device)
    if arch < 89:
        raise RuntimeError(f"Humming MXFP4xFP8 requires SM89 or newer; got SM{arch}")


def validate_mxfp4_flashinfer_device(workload: Workload, device: Any) -> None:
    del workload
    capability, arch = cuda_arch(device)
    if capability != (9, 0):
        raise RuntimeError(
            f"FlashInfer's Humming-style MXFP4xFP8 provider requires SM90; got SM{arch}"
        )


def validate_wint4_humming_device(workload: Workload, device: Any) -> None:
    del workload
    _, arch = cuda_arch(device)
    if arch < 89:
        raise RuntimeError(f"Humming WINT4xAFP8 requires SM89 or newer; got SM{arch}")


def validate_wint4_flashinfer_device(workload: Workload, device: Any) -> None:
    del workload
    capability, arch = cuda_arch(device)
    if capability != (9, 0):
        raise RuntimeError(
            f"FlashInfer's WINT4xAFP8 fused MoE provider requires SM90; got SM{arch}"
        )


def validate_sglang_cutlass_device(workload: Workload, device: Any) -> None:
    del workload
    capability, arch = cuda_arch(device)
    if capability != (9, 0):
        raise RuntimeError(
            f"SGLang CUTLASS W4A8 fused MoE requires SM90; got SM{arch}"
        )


def validate_sglang_marlin_device(workload: Workload, device: Any) -> None:
    del workload
    _, arch = cuda_arch(device)
    if arch < 80:
        raise RuntimeError(f"SGLang Marlin W4A16 requires SM80 or newer; got SM{arch}")


def humming_input_quant_config(workload: Workload) -> dict[str, Any]:
    return {
        "dtype": workload.quant_params["activation_dtype"],
        "group_size": workload.quant_params["input_scale_group_size"],
        "scale_dtype": workload.quant_params["input_scale_dtype"],
    }


def quant_result_metadata(workload: Workload) -> dict[str, Any]:
    provider = get_quant_provider(workload.quant_mode)
    params = workload.quant_params
    return {
        "quant_contract": quant_contract_name(workload.quant_mode, params),
        "comparison_semantics": provider.comparison_semantics,
        "cross_backend_contract_matched": (
            provider.comparison_semantics == "contract_matched"
        ),
        "quant_params_json": json.dumps(params, sort_keys=True),
        "weight_group_size": params["weight_group_size"],
        "weight_scale_dtype": params["weight_scale_dtype"],
        "activation_dtype": params["activation_dtype"],
        "input_scale_group_size": params.get("input_scale_group_size"),
        "input_scale_dtype": params.get("input_scale_dtype"),
    }


register_quant_provider(
    QuantProvider(
        mode="mxfp4_fp8",
        aliases=("mxfp4xfp8", "mxfp4_afp8"),
        contract_name="hopper_mxfp4_e2m1_e8m0g32_x_fp8e4m3",
        comparison_semantics="contract_matched",
        normalize_params=normalize_mxfp4_fp8_params,
        validate_workload=validate_mxfp4_fp8_workload,
        implementations={
            "humming": BackendImplementation(
                validate_workload=validate_backend_workload_noop,
                validate_device=validate_mxfp4_humming_device,
                make_state=make_mxfp4_humming_state,
                make_call=make_sglang_runner_call,
            ),
            "flashinfer": BackendImplementation(
                validate_workload=validate_backend_workload_noop,
                validate_device=validate_mxfp4_flashinfer_device,
                make_state=make_mxfp4_flashinfer_state,
                make_call=make_mxfp4_flashinfer_call,
                autotune=True,
            ),
        },
    )
)
register_quant_provider(
    QuantProvider(
        mode="wint4_afp8",
        aliases=("wint4afp8", "wint4xafp8", "int4_fp8", "w4a8"),
        contract_name="hopper_signed_int4_bf16g128_x_fp8e4m3",
        comparison_semantics="shared_weight_native_activation_quant",
        normalize_params=normalize_wint4_afp8_params,
        validate_workload=validate_wint4_afp8_workload,
        implementations={
            "humming": BackendImplementation(
                validate_workload=validate_backend_workload_noop,
                validate_device=validate_wint4_humming_device,
                make_state=make_wint4_humming_state,
                make_call=make_sglang_runner_call,
            ),
            "flashinfer": BackendImplementation(
                validate_workload=validate_flashinfer_wint4_workload,
                validate_device=validate_wint4_flashinfer_device,
                make_state=make_wint4_flashinfer_state,
                make_call=make_wint4_flashinfer_call,
                autotune=True,
            ),
            "sglang_cutlass": BackendImplementation(
                validate_workload=validate_sglang_cutlass_workload,
                validate_device=validate_sglang_cutlass_device,
                make_state=make_wint4_sglang_cutlass_state,
                make_call=make_wint4_sglang_cutlass_call,
            ),
        },
    )
)
register_quant_provider(
    QuantProvider(
        mode="wint4_a16",
        aliases=("wint4a16", "wint4xa16", "int4_bf16", "w4a16"),
        contract_name="sglang_marlin_signed_int4_bf16g128_x_bfloat16",
        comparison_semantics="single_backend_native_contract",
        normalize_params=normalize_wint4_a16_params,
        validate_workload=validate_wint4_a16_workload,
        implementations={
            "sglang_marlin": BackendImplementation(
                validate_workload=validate_backend_workload_noop,
                validate_device=validate_sglang_marlin_device,
                make_state=make_wint4_sglang_marlin_state,
                make_call=make_sglang_runner_call,
            ),
        },
    )
)


def benchmark_callable(
    fn: Callable[[], Any],
    warmup: int,
    repeat: int,
    wall_repeat: int,
    cold_l2: bool,
    use_cupti: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    from flashinfer.testing import bench_gpu_time

    gpu_times = bench_gpu_time(
        fn=fn,
        dry_run_iters=warmup,
        repeat_iters=repeat,
        sleep_after_run=False,
        enable_cupti=use_cupti,
        use_cuda_graph=False,
        cold_l2_cache=cold_l2,
    )
    gpu_summary = summarize_times(gpu_times, "gpu")

    wall_times: list[float] = []
    if wall_repeat > 0:
        for _ in range(min(warmup, 5)):
            fn()
        torch.cuda.synchronize()
        for _ in range(wall_repeat):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            wall_times.append((time.perf_counter() - start) * 1000.0)
    return gpu_summary, summarize_times(wall_times, "wall")


def output_sample_stats(output: Any) -> dict[str, Any]:
    flat = output.detach().reshape(-1)
    if flat.numel() > 4096:
        step = max(1, flat.numel() // 4096)
        flat = flat[::step][:4096]
    sample = flat.float()
    finite = bool(torch.isfinite(sample).all().item())
    mean = float(sample.mean().item())
    std = float(sample.std().item()) if sample.numel() > 1 else 0.0
    absmax = float(sample.abs().max().item())
    probe_step = max(1, sample.numel() // 64)
    probe_values = sample[::probe_step][:64].cpu().tolist()
    return {
        "output_sample_finite": finite,
        "output_sample_mean": mean if math.isfinite(mean) else None,
        "output_sample_std": std if math.isfinite(std) else None,
        "output_sample_absmax": absmax if math.isfinite(absmax) else None,
        "output_probe": [
            float(value) if math.isfinite(float(value)) else None
            for value in probe_values
        ],
    }


def git_commit_for_module(module_name: str) -> str:
    try:
        module = __import__(module_name)
        module_file = Path(module.__file__).resolve()
        root = subprocess.check_output(
            ["git", "-C", str(module_file.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def collect_environment(backend: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "backend": backend,
        "benchmark_scope": BENCHMARK_SCOPE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
    }
    for name in ("sglang", "flashinfer", "humming"):
        try:
            module = __import__(name)
            values[f"{name}_version"] = getattr(module, "__version__", "")
            values[f"{name}_commit"] = git_commit_for_module(name)
        except Exception:
            values[f"{name}_version"] = ""
            values[f"{name}_commit"] = ""
    return values


def run_benchmark(args: argparse.Namespace) -> int:
    global torch

    workload_path = Path(args.workload).expanduser().resolve()
    workload = load_workload(workload_path)
    provider = get_quant_provider(workload.quant_mode)
    implementation = get_backend_implementation(provider, args.backend)
    implementation.validate_workload(workload)
    if args.dry_run:
        print(json.dumps(workload.public_dict(), indent=2, sort_keys=True))
        return 0

    # Backend configuration must be visible before importing SGLang/Humming.
    if args.backend == "humming":
        os.environ["SGLANG_HUMMING_INPUT_QUANT_CONFIG"] = json.dumps(
            humming_input_quant_config(workload)
        )
        os.environ["SGLANG_HUMMING_MOE_GEMM_TYPE"] = args.humming_gemm
        os.environ.setdefault("HUMMING_COMPILER", "nvcc")
    if args.backend in ("humming", "sglang_marlin"):
        os.environ["SGLANG_CI_DISABLE_MOE_FUSED_FUNC"] = "0"

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    implementation.validate_device(workload, device)
    state = implementation.make_state(workload, device)
    autotune_enabled = implementation.autotune and not args.no_autotune

    rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for m in workload.num_tokens:
                hidden = make_hidden_states(workload, m, device)
                ids, weights = make_routing(workload, m, device)
                fn = implementation.make_call(
                    workload, state, hidden, ids, weights
                )
                if autotune_enabled:
                    from flashinfer.autotuner import autotune

                    with autotune(True):
                        fn()

                # Compile any shape-specialized kernels before timing.
                preflight_output = fn()
                torch.cuda.synchronize()
                del preflight_output
                gpu, wall = benchmark_callable(
                    fn=fn,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    wall_repeat=args.wall_repeat,
                    cold_l2=args.cold_l2,
                    use_cupti=args.use_cupti,
                )
                verification_output = fn()
                torch.cuda.synchronize()
                flops = (
                    6.0
                    * m
                    * workload.top_k
                    * workload.hidden_size
                    * workload.local_intermediate_size
                )
                p50_ms = gpu["gpu_p50_ms"]
                if not math.isfinite(p50_ms) or p50_ms <= 0:
                    raise RuntimeError(f"invalid measured p50 latency: {p50_ms}")
                row = {
                    "name": workload.name,
                    "backend": args.backend,
                    "scope": BENCHMARK_SCOPE,
                    "quant_mode": workload.quant_mode,
                    **quant_result_metadata(workload),
                    "M": m,
                    "K": workload.hidden_size,
                    "N_global": workload.intermediate_size,
                    "N_local": workload.local_intermediate_size,
                    "E_global": workload.num_experts,
                    "E_local": workload.local_num_experts,
                    "top_k": workload.top_k,
                    "top_k_scope": workload.top_k_scope,
                    "tp_size": workload.tp_size,
                    "tp_rank": workload.tp_rank,
                    "ep_size": workload.ep_size,
                    "ep_rank": workload.ep_rank,
                    "routing": workload.routing,
                    "normalize_trace_weights": workload.normalize_trace_weights,
                    "weight_seed": workload.weight_seed,
                    "input_seed": workload.input_seed,
                    "routing_seed": workload.routing_seed,
                    "input_amplitude": workload.input_amplitude,
                    "routed_scaling_factor": workload.routed_scaling_factor,
                    "swiglu_limit": workload.swiglu_limit,
                    "use_fused_finalize": workload.use_fused_finalize,
                    "humming_gemm": (
                        args.humming_gemm if args.backend == "humming" else None
                    ),
                    "backend_autotune": autotune_enabled,
                    "flashinfer_autotune": (
                        args.backend == "flashinfer" and autotune_enabled
                    ),
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "wall_repeat": args.wall_repeat,
                    "cold_l2": args.cold_l2,
                    "use_cupti": args.use_cupti,
                    "algorithmic_tflops": flops / (p50_ms * 1.0e9),
                    "tokens_per_second": m * 1000.0 / p50_ms,
                    "token_expert_pairs_per_second": (
                        m * workload.top_k * 1000.0 / p50_ms
                    ),
                    **route_statistics(ids, workload.local_num_experts),
                    **output_sample_stats(verification_output),
                    **gpu,
                    **wall,
                }
                rows.append(row)
                wall_p50 = row.get("wall_p50_ms")
                wall_display = (
                    "disabled" if wall_p50 is None else f"{float(wall_p50):.4f} ms"
                )
                print(
                    f"[{args.backend}] {workload.name} M={m}: "
                    f"gpu_p50={p50_ms:.4f} ms wall_p50={wall_display}",
                    flush=True,
                )
                del hidden, ids, weights, verification_output, fn
                torch.cuda.empty_cache()
    finally:
        parallel_override = state.pop("_parallel_override", None)
        if parallel_override is not None:
            parallel_override.__exit__(None, None, None)
        legacy_parallel_getter = state.pop("_legacy_parallel_getter", None)
        if legacy_parallel_getter is not None:
            import sglang.srt.layers.moe.cutlass_w4a8_moe as cutlass_module

            cutlass_module.get_moe_expert_parallel_world_size = (
                legacy_parallel_getter
            )

    payload = {
        "benchmark_scope": BENCHMARK_SCOPE,
        "backend": args.backend,
        "workload_source": str(workload_path),
        "workload": workload.public_dict(),
        "environment": collect_environment(args.backend),
        "run_args": {
            "device": args.device,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "wall_repeat": args.wall_repeat,
            "cold_l2": args.cold_l2,
            "use_cupti": args.use_cupti,
            "backend_autotune": autotune_enabled,
            "flashinfer_autotune": (
                args.backend == "flashinfer" and autotune_enabled
            ),
            "humming_gemm": (
                args.humming_gemm if args.backend == "humming" else None
            ),
        },
        "rows": rows,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[OUTPUT] {output_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fused-MoE workload on exactly one Humming, FlashInfer, "
            "SGLang CUTLASS, or SGLang Marlin backend."
        )
    )
    parser.add_argument("--workload", required=True, help="single-workload JSON object")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output", help="result JSON file; required unless --dry-run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--wall-repeat", type=int, default=20)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--use-cupti", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument(
        "--humming-gemm", choices=("indexed", "grouped"), default="indexed"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.wall_repeat < 0:
        parser.error("warmup/repeat/wall-repeat must be non-negative and repeat > 0")
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is used")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
