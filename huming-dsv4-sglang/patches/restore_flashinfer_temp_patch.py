#!/usr/bin/env python3
import json
import shutil
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
QUANT_DIR = Path(
    "<python-site-packages>/sglang/srt/layers/quantization"
)
BACKUP_DIR = RUN_DIR / "site_packages_backup_flashinfer_mxfp4"


def main() -> int:
    manifest_path = BACKUP_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    restored = []
    removed = []
    for name, meta in manifest.items():
        dst = QUANT_DIR / name
        src = BACKUP_DIR / name
        if meta["existed"]:
            shutil.copy2(src, dst)
            restored.append(name)
        elif dst.exists():
            dst.unlink()
            removed.append(name)
    print(
        json.dumps(
            {
                "status": "restored",
                "restored": restored,
                "removed": removed,
                "backup_dir": str(BACKUP_DIR),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
