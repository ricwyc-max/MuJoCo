#!/usr/bin/env bash
# 在项目根目录创建 .venv 并用国内镜像安装依赖（幂等，可重复执行）。
#
#   bash scripts/setup_venv.sh
#
# 镜像按顺序尝试，清华同步不全时自动回退阿里云 / 北外。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
# 固定用 3.11：本机 python/py 指向不同版本，写死解释器避免串味
BASE_PYTHON="${BASE_PYTHON:-D:/python3.11/python.exe}"
MIRRORS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple/"
  "https://mirrors.bfsu.edu.cn/pypi/web/simple"
)

if [ ! -x "$VENV/Scripts/python.exe" ]; then
  echo "[setup] 用 $BASE_PYTHON 创建虚拟环境 .venv"
  "$BASE_PYTHON" -m venv "$VENV"
else
  echo "[setup] .venv 已存在，跳过创建"
fi

PY="$VENV/Scripts/python.exe"

for MIRROR in "${MIRRORS[@]}"; do
  echo "[setup] 源: $MIRROR"
  if "$PY" -m pip install -i "$MIRROR" -U pip setuptools wheel \
     && "$PY" -m pip install -i "$MIRROR" -r requirements.txt; then
    break
  fi
  echo "[setup] 该源失败，尝试下一个"
done

"$PY" - <<'EOF'
import mujoco, numpy, matplotlib, imageio
print(f"[setup] 完成：mujoco {mujoco.__version__} | numpy {numpy.__version__} "
      f"| matplotlib {matplotlib.__version__} | imageio {imageio.__version__}")
EOF

echo "[setup] 接下来可以运行： $PY scripts/run_all.py"