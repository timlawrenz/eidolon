"""Probe the CUDA-lib re-exec branch condition in enrichment.py (lines 210-231).

Read-only. Reproduces the exact predicate that decides whether `enrich` restarts
itself to load cuDNN/cuBLAS from the geometry_pca venv, so the audit can explain
why the branch fired on 2026-11-10 but not 2026-11-11 with an existing path.
"""
import os

venv_base = '/home/tim/source/activity/eidolon/experiments/geometry_pca/.venv/lib/python3.14/site-packages'
cudnn_path = f"{venv_base}/nvidia/cudnn/lib"
cur = os.environ.get('LD_LIBRARY_PATH', '')

print("current_ld repr:", repr(cur))
print("cudnn_path:", cudnn_path)
print("os.path.exists(cudnn_path):", os.path.exists(cudnn_path))
print("cudnn_path in current_ld:", cudnn_path in cur)
print("would_restart:", (cudnn_path not in cur) and os.path.exists(cudnn_path))

try:
    from insightface.app import FaceAnalysis  # noqa: F401
    print("insightface IMPORT OK")
except ImportError as e:
    print("insightface ImportError:", e)
except Exception as e:  # noqa: BLE001
    print("other exc:", type(e).__name__, e)
