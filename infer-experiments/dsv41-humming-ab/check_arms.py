#!/usr/bin/env python3
"""Reject arm drift outside identity, metadata, docs and the MoE runner token."""
import argparse
import difflib
from pathlib import Path
import re
import shlex
import sys
import tomllib


def normalized(text):
    section = ''
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith('['):
            section = line.strip()
        if (section == '[task]' and re.match(r'^(name|description|keywords)\s*=', line)) or (
            section == '[metadata]' and re.match(r'^\w+\s*=', line)
        ):
            continue
        lines.append(line)
    return ''.join(lines)


def check(off, on):
    files = lambda root: {p.relative_to(root) for p in root.rglob('*') if p.is_file()}
    a, b = files(off), files(on)
    if a != b:
        raise ValueError(f'file sets differ: {sorted(a ^ b)}')
    for rel in sorted(a):
        x, y = off / rel, on / rel
        if x.is_symlink() or y.is_symlink():
            raise ValueError(f'symlink not allowed: {rel}')
        if x.stat().st_mode & 0o777 != y.stat().st_mode & 0o777:
            raise ValueError(f'file modes differ: {rel}')
        if str(rel) in {'README.md', 'SOURCES.md'}:
            continue
        if str(rel) == 'task.toml':
            sx, sy = x.read_text(), y.read_text()
            tx, ty = tomllib.loads(sx), tomllib.loads(sy)
            cx = tx['environment']['env']['SGLANG_LAUNCH_CMD']
            cy = ty['environment']['env']['SGLANG_LAUNCH_CMD']
            for cmd, backend in [(cx, 'flashinfer_mxfp4'), (cy, 'humming')]:
                tokens = shlex.split(cmd)
                assert tokens.count('--moe-runner-backend') == 1
                assert tokens[tokens.index('--moe-runner-backend') + 1] == backend
            expected = cx.replace('--moe-runner-backend flashinfer_mxfp4', '--moe-runner-backend humming')
            if cy != expected:
                raise ValueError('launch commands differ beyond the MoE runner token')
            sy = sy.replace(cy, cx, 1)
            nx, ny = normalized(sx), normalized(sy)
            if nx != ny:
                raise ValueError('unexpected task.toml diff:\n' + ''.join(difflib.unified_diff(nx.splitlines(True), ny.splitlines(True))))
        elif x.read_bytes() != y.read_bytes():
            raise ValueError(f'unexpected diff: {rel}')
    print('PASS: only allowed arm differences; launch backend flashinfer_mxfp4 -> humming')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'infer-tasks'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('off', nargs='?', type=Path, default=root / 'dsv41-flash-h200-fp4-lowlatency')
    parser.add_argument('on', nargs='?', type=Path, default=root / 'dsv41-flash-h200-fp4-lowlatency-humming')
    args = parser.parse_args()
    try:
        check(args.off, args.on)
    except (ValueError, KeyError, AssertionError, OSError) as exc:
        sys.exit(f'FAIL: {exc}')
