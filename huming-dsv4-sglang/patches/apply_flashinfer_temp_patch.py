#!/usr/bin/env python3
import json
import shutil
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
QUANT_DIR = Path(
    "<python-site-packages>/sglang/srt/layers/quantization"
)
BACKUP_DIR = RUN_DIR / "site_packages_backup_flashinfer_mxfp4"
FILES = [
    "fp8.py",
    "mxfp4_flashinfer_trtllm_moe.py",
    "mxfp4_flashinfer_cutlass_moe.py",
]


def backup() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FILES:
        src = QUANT_DIR / name
        dst = BACKUP_DIR / name
        manifest[name] = {"existed": src.exists()}
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    (BACKUP_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    return text.replace(old, new)


def patch_fp8() -> None:
    path = QUANT_DIR / "fp8.py"
    text = path.read_text()
    old = """            if self.is_fp4_experts and get_moe_runner_backend().is_flashinfer_mxfp4():
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(fp8_method, prefix=prefix)
"""
    new = """            if self.is_fp4_experts and get_moe_runner_backend().is_flashinfer_mxfp4():
                # SM100 (Blackwell) -> trtllm-gen path.
                # SM90  (Hopper)    -> cutlass mixed-input path (FlashInfer #3084).
                if is_sm90_supported() and not is_sm100_supported():
                    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
                        Mxfp4FlashinferCutlassMoEMethod,
                    )

                    return Mxfp4FlashinferCutlassMoEMethod(fp8_method, prefix=prefix)

                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(fp8_method, prefix=prefix)
"""
    path.write_text(replace_once(text, old, new, path))


def patch_trtllm_helper() -> None:
    path = QUANT_DIR / "mxfp4_flashinfer_trtllm_moe.py"
    text = path.read_text()
    old = """    from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
        Mxfp4MarlinMoEMethod,
    )

    fused = isinstance(
        experts.quant_method, (Mxfp4FlashinferTrtllmMoEMethod, Mxfp4MarlinMoEMethod)
    )
"""
    new = """    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
        Mxfp4MarlinMoEMethod,
    )

    fused = isinstance(
        experts.quant_method,
        (
            Mxfp4FlashinferTrtllmMoEMethod,
            Mxfp4FlashinferCutlassMoEMethod,
            Mxfp4MarlinMoEMethod,
        ),
    )
"""
    path.write_text(replace_once(text, old, new, path))


def main() -> int:
    if not QUANT_DIR.exists():
        raise FileNotFoundError(QUANT_DIR)
    backup()
    shutil.copy2(RUN_DIR / "mxfp4_flashinfer_cutlass_moe.py", QUANT_DIR / "mxfp4_flashinfer_cutlass_moe.py")
    patch_fp8()
    patch_trtllm_helper()
    print(json.dumps({"status": "patched", "backup_dir": str(BACKUP_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
