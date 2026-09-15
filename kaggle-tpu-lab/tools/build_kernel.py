#!/usr/bin/env python3
"""Write a model's self-contained kernel script: what `launch.py serve` pushes, without a config
(CFG = None; the kernel then reads serve_config.json next to it), with the engine package and the
control module embedded. For runners other than launch.py, e.g. DeployMan's services.

    python tools/build_kernel.py glm53-flash ../../DeployMan/services/glm53-api/serve_glm53.py
    python tools/build_kernel.py qwen38-27b  ../../DeployMan/services/qwen38-api/serve_qwen38.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import launch  # noqa: E402


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in launch.MODELS:
        sys.exit(f"usage: build_kernel.py {{{'|'.join(sorted(launch.MODELS))}}} <output.py>")
    src = launch.build_kernel(sys.argv[1], {})
    src = re.sub(r"^CFG = \{\}$", "CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)", src,
                 count=1, flags=re.M)
    out = Path(sys.argv[2])
    out.write_text(src, encoding="utf-8", newline="\n")
    print(f"{out}: {len(src) // 1024} KB")


if __name__ == "__main__":
    main()
