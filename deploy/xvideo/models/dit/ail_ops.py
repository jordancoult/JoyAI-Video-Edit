"""ai-livestream fork: small lossless fused ops for the DiT (bit-identical to the ops they replace).

1. fused_norm_scale_shift_bcast: joyomni_ops' LayerNorm/RMSNorm + adaLN kernel with a broadcast scale/shift
   (one row per sample) instead of an expanded [tokens, D] copy. Source ail_norm.cu is generated from
   joyomni_ops/csrc/fused_norm_scale_shift.cu with only the row indexing changed, built with the same nvcc flags.
   JIT-built once per GPU into $JOYOMNI_CACHE_ROOT/ail_ops (or ~/.cache), ~1 min; any failure -> None (caller
   falls back to the original path).
Off with JOYOMNI_AIL_OPS=0.
"""
import os
import sys
import threading

import torch

_lock = threading.Lock()
_norm_mod = None
_norm_tried = False
ENABLED = os.environ.get("JOYOMNI_AIL_OPS", "1") != "0"


def _build_dir():
    base = os.environ.get("JOYOMNI_CACHE_ROOT") or os.path.expanduser("~/.cache")
    major, minor = torch.cuda.get_device_capability()
    d = os.path.join(base, "ail_ops", f"sm{major}{minor}_torch{torch.__version__.split('+')[0]}")
    os.makedirs(d, exist_ok=True)
    return d, major, minor


def norm_module():
    global _norm_mod, _norm_tried
    if not ENABLED:
        return None
    if _norm_tried:
        return _norm_mod
    with _lock:
        if _norm_tried:
            return _norm_mod
        try:
            from torch.utils.cpp_extension import load
            _bin = os.path.dirname(sys.executable)   # ninja lives in the venv's bin, which may not be on PATH
            if _bin not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
            d, major, minor = _build_dir()
            arch = f"{major}{minor}" + ("a" if major >= 9 else "")
            src = os.path.join(os.path.dirname(__file__), "ail_norm.cu")
            _norm_mod = load(
                name="ail_norm", sources=[src], build_directory=d, verbose=False,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                                   "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                   "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                   f"-gencode=arch=compute_{arch},code=sm_{arch}"])
            print(f"#####[ail_ops] broadcast adaLN norm kernel ready ({d})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"#####[ail_ops] norm kernel unavailable, using joyomni_ops path: {exc!r}", flush=True)
            _norm_mod = None
        _norm_tried = True
    return _norm_mod


def fused_norm_scale_shift_bcast(x_2d, gamma, beta, scale_rows, shift_rows, norm_type, eps, rows_per_mod):
    """x_2d [M, D]; scale_rows/shift_rows [M // rows_per_mod, D] contiguous. Returns None if unavailable."""
    mod = norm_module()
    if mod is None:
        return None
    nt = 0 if norm_type == "layer" else 1
    return mod.fused_norm_scale_shift(x_2d, gamma, beta, scale_rows, shift_rows, nt, float(eps), int(rows_per_mod))
