"""Phase 1: environment report + identity of every model/data file read from other projects -> results/metrics/env.json"""
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow  # noqa: E402
import numpy  # noqa: E402
import psutil  # noqa: E402
import tokenizers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from bench.load_models import MODELS  # noqa: E402  (Project 4, read-only)

files = {"tokenizer (Project 1)": flow.TOKENIZER, "embeddings: Project 1 small model": flow.EMB_CKPT,
         "scorer TinyStories-33M (Project 4 download)": os.path.join(flow.SCORER_DIR, "pytorch_model.bin")}
for k, v in MODELS.items():
    files[f"baseline {k}"] = v["path"]
for s in ("train", "val", "test"):
    files[f"token data {s} (Project 1)"] = os.path.join(flow.P1_DATA, f"{s}.bin")
info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(), "device": "cpu", "precision": "fp32",
        "data_license": "TinyStories (roneneldan/TinyStories): CDLA-Sharing-1.0",
        "scorer_license": "TinyStories-33M: MIT",
        "sha256": {k: flow.sha256(v) for k, v in files.items() if os.path.exists(v)}}
flow.save_json(info, flow.METRICS, "env.json")
print({k: v for k, v in info.items() if k != "sha256"})
for k, v in info["sha256"].items():
    print(f"  {k}: {v[:8]}...{v[-6:]}")
