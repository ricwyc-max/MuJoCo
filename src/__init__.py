"""MuJoCo 课程作业 —— 任务二：重力补偿与保持。

任何以 ``src`` 为包的脚本被导入时，都把标准输出切到 UTF-8，
避免 Windows 控制台（默认 GBK）打印中文时抛 UnicodeEncodeError。
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 极少数无控制台场景
            pass