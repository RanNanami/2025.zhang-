"""旧 Fig.9 historical runner 的兼容入口。

新代码应从 `experiments.fig9` 导入稳定 helper，或显式从
`experiments.historical.fig9_paper_snn` 导入历史诊断。此文件保留旧 import
路径和 `python experiments/fig9_paper_snn.py ...` 命令。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.historical.fig9_paper_snn import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    run(parse_args())
