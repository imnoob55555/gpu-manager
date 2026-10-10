"""``python -m gpu_manager`` 入口（daemon 后台化时会用到）。"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
