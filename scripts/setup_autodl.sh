#!/usr/bin/env bash
# AutoDL: PyTorch 2.1.0–2.1.2 / Python 3.10 / CUDA 12.1，复用镜像而非重装 PyTorch。
set -euo pipefail

usage() {
  printf '%s\n' \
    '用法：bash scripts/setup_autodl.sh [选项]' \
    '' \
    '适用镜像：Linux x86_64，Python 3.10，PyTorch 2.1.0–2.1.2，CUDA 12.1。' \
    '默认克隆镜像 base 环境，只补项目依赖，不修改原环境或重装 PyTorch。' \
    '' \
    '  --env-name NAME    目标 conda 环境，默认 wwtp-v6；存在则继续安装' \
    '  --source-env NAME  镜像中的源环境，默认 base' \
    '  --index-url URL    普通 Python 包下载源，默认 https://pypi.org/simple' \
    '  -h, --help         显示帮助' \
    '' \
    '示例：bash scripts/setup_autodl.sh --env-name wwtp-v6' \
    '      bash scripts/setup_autodl.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple' \
    '' \
    'MMCV 使用官方 cu121/torch2.1 wheel，禁止回退源码编译。' \
    '系统需有 libgl1、libglib2.0-0；缺失时需自行安装。' \
    '数据、伪几何和预训练权重需另行准备；本脚本不启动训练。'
}

fail() { printf '错误：%s\n' "$*" >&2; exit 1; }

env_name=wwtp-v6
source_env=base
index_url=https://pypi.org/simple
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --env-name|--source-env|--index-url)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "$1 缺少参数"
      case "$1" in
        --env-name) env_name="$2" ;;
        --source-env) source_env="$2" ;;
        --index-url) index_url="$2" ;;
      esac
      shift 2 ;;
    *) fail "未知参数：$1（使用 --help 查看用法）" ;;
  esac
done
for name in "$env_name" "$source_env"; do
  [[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "无效环境名：$name"
done
[[ "$env_name" != base && "$env_name" != root && "$env_name" != "$source_env" ]] \
  || fail '目标必须是独立环境，不能为 base/root 或与源环境相同'
[[ "$index_url" == https://* || "$index_url" == http://* ]] || fail '下载源需要 HTTP(S) URL'

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-}"
if [[ ! -x "$conda_exe" ]]; then
  conda_exe="$(type -P conda || true)"
fi
if [[ ! -x "$conda_exe" ]]; then
  for candidate in /root/miniconda3/bin/conda "$HOME/miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then conda_exe="$candidate"; break; fi
  done
fi
[[ -x "$conda_exe" ]] || fail '未找到 conda，请先加载 AutoDL 镜像中的 conda'

# 隔离用户站点和隐式额外源；保留代理及 pip 下载缓存。
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL="$index_url"
export PIP_EXTRA_INDEX_URL='' PIP_FIND_LINKS='' PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=5
unset PIP_TARGET PIP_PREFIX PIP_USER PIP_REQUIREMENT PIP_CONSTRAINT PIP_NO_INDEX
run_python() {
  local environment="$1"
  shift
  "$conda_exe" run --no-capture-output -n "$environment" python -I "$@"
}
check_base() {
  run_python "$1" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 10) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('需要 Linux x86_64 / Python 3.10，请选择对应的 AutoDL 基础镜像。')
import torch

if torch.__version__.split('+')[0] not in ('2.1.0', '2.1.1', '2.1.2') or torch.version.cuda != '12.1':
    raise SystemExit(f'镜像不匹配：torch={torch.__version__}, CUDA={torch.version.cuda}；'
                     '需要 PyTorch 2.1.0–2.1.2 / CUDA 12.1，不自动重装基础框架。')
print(f'基础环境：Python {platform.python_version()}, torch {torch.__version__}, '
      f'CUDA {torch.version.cuda}', flush=True)
PY
}

printf '检查镜像源环境：%s\n' "$source_env"
check_base "$source_env"
source_prefix="$(run_python "$source_env" -c 'import sys; print(sys.prefix)')"
existing="$("$conda_exe" env list --json | run_python "$source_env" -c \
  'import json, pathlib, sys; print(int(any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"])))' "$env_name")"
if [[ "$existing" == 1 ]]; then
  printf '复用已有目标环境：%s\n' "$env_name"
else
  printf '克隆 %s → %s，复用镜像中的 PyTorch/CUDA 和已安装依赖。\n' "$source_env" "$env_name"
  "$conda_exe" create -y -n "$env_name" --clone "$source_prefix"
fi
check_base "$env_name"
run_python "$env_name" - "$source_prefix" <<'PY'
from pathlib import Path
import sys

if Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve():
    raise SystemExit('目标与源环境实际路径相同，拒绝修改源环境。')
PY

# 有些镜像只预装 torch；在目标环境补配套 torchvision，不拉取 torch 依赖。
torchvision_version="$(run_python "$env_name" -c \
  "import torch; print({'2.1.0': '0.16.0', '2.1.1': '0.16.1', '2.1.2': '0.16.2'}[torch.__version__.split('+')[0]])")"
if ! run_python "$env_name" -c "import torchvision; assert torchvision.__version__.split('+')[0] == '$torchvision_version'"; then
  run_python "$env_name" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    --no-deps "torchvision==$torchvision_version+cu121"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
# 原始 requirements 不变；只把安装副本中的框架版本替换为镜像实际版本。
run_python "$env_name" - "$repo_root" "$tmp_dir" <<'PY'
from importlib.metadata import version
from pathlib import Path
import re
import sys

root, temp = map(Path, sys.argv[1:])
core = (root / 'requirements/core.txt').read_text()
for package in ('torch', 'torchvision'):
    core, count = re.subn(rf'(?m)^{package}==[^\n]+', f'{package}=={version(package)}', core)
    if count != 1:
        raise SystemExit(f'无法定位 requirements/core.txt 中的 {package} 固定项。')
(temp / 'core.txt').write_text(core)
(temp / 'runtime.txt').write_text((root / 'requirements/runtime.txt').read_text())
(temp / 'constraints.txt').write_text(core + '\nmmcv==2.1.0\n')
print('安装临时依赖清单，保留镜像的 torch/torchvision 版本。', flush=True)
PY

run_python "$env_name" -m pip install --index-url "$index_url" 'pip<25' 'setuptools<80' wheel
run_python "$env_name" -m pip install --index-url "$index_url" \
  -c "$tmp_dir/constraints.txt" -r "$tmp_dir/core.txt"

if run_python "$env_name" - <<'PY'
import mmcv
from mmcv.ops import get_compiling_cuda_version
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

if mmcv.__version__ != '2.1.0' or not get_compiling_cuda_version().startswith('12.1'):
    raise SystemExit('现有 MMCV 与目标环境不匹配。')
PY
then
  printf '复用镜像中已有的 MMCV 2.1.0 CUDA 扩展。\n'
else
  printf '安装官方 MMCV 2.1.0 cu121/torch2.1 二进制包。\n'
  mkdir "$tmp_dir/mmcv"
  run_python "$env_name" -m pip download --no-index --no-deps --only-binary=:all: \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    --dest "$tmp_dir/mmcv" mmcv==2.1.0
  wheels=("$tmp_dir/mmcv"/mmcv-2.1.0-*.whl)
  [[ ${#wheels[@]} == 1 && -f "${wheels[0]}" ]] || fail '未取得唯一匹配的 MMCV wheel'
  run_python "$env_name" -m pip uninstall -y mmcv mmcv-lite mmcv-full
  run_python "$env_name" -m pip install --no-deps "${wheels[0]}"
fi
run_python "$env_name" -m pip install --index-url "$index_url" \
  -c "$tmp_dir/constraints.txt" -r "$tmp_dir/runtime.txt" mmcv==2.1.0
run_python "$env_name" -m pip check

run_python "$env_name" - "$repo_root" <<'PY'
from pathlib import Path
import sys
import sysconfig

root = Path(sys.argv[1]).resolve()
site = Path(sysconfig.get_path('purelib')).resolve()
if not site.is_relative_to(Path(sys.prefix).resolve()):
    raise SystemExit('site-packages 不属于目标环境，拒绝写入项目路径。')
path = site / 'wwtp_autodl.pth'
content = str(root) + '\n'
previous = path.read_text() if path.exists() else None
if previous != content:
    path.write_text(content)
print(f'项目导入路径：{root}', flush=True)
PY

run_python "$env_name" - "$repo_root" <<'PY'
from pathlib import Path
import sys

import torch
import torchvision
import mmcv
import mmengine
import mmseg
import mmdet
from mmcv.ops import get_compiling_cuda_version
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmseg.registry import MODELS
import wwtpseg

torch_version = torch.__version__.split('+')[0]
expected_torchvision = {'2.1.0': '0.16.0', '2.1.1': '0.16.1', '2.1.2': '0.16.2'}.get(torch_version)
assert expected_torchvision is not None and torch.version.cuda == '12.1', '基础框架版本不匹配'
assert torchvision.__version__.split('+')[0] == expected_torchvision, 'torchvision 与 PyTorch 不配套'
assert (mmcv.__version__, mmengine.__version__, mmseg.__version__, mmdet.__version__) == ('2.1.0', '0.10.7', '1.2.2', '3.3.0'), 'OpenMMLab 版本不匹配'
assert get_compiling_cuda_version().startswith('12.1'), 'MMCV 不是 CUDA 12.1 扩展'
assert Path(wwtpseg.__file__).resolve() == Path(sys.argv[1]).resolve() / 'wwtpseg/__init__.py', '导入了其他目录中的 wwtpseg'
assert MODELS.get('RPGVNetV6') is not None, 'v6 未注册'
print('依赖版本、MMCV 扩展、项目导入与 v6 注册检查通过。', flush=True)
if torch.cuda.is_available():
    value = torch.randn(1, 4, 1, 4, device='cuda', requires_grad=True)
    shapes = torch.tensor([[2, 2]], dtype=torch.long, device='cuda')
    starts = torch.zeros(1, dtype=torch.long, device='cuda')
    locations = torch.rand(1, 2, 1, 1, 2, 2, device='cuda', requires_grad=True)
    weights = torch.rand(1, 2, 1, 1, 2, device='cuda', requires_grad=True)
    output = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locations, weights, 64)
    assert torch.isfinite(output).all(), 'MMCV CUDA 前向结果非有限值'
    output.square().mean().backward()
    for tensor in (value, locations, weights):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), 'MMCV CUDA 梯度异常'
    torch.cuda.synchronize()
    print(f'CUDA / MMCV 算子前后向通过：{torch.cuda.get_device_name()}', flush=True)
else:
    print('注意：当前 GPU 不可用，仅完成依赖检查，尚未验证 GPU。', flush=True)
PY

printf '\n环境就绪。后续运行：\n'
conda_base="$("$conda_exe" info --base)"
printf '  source %q\n  conda activate %q\n  cd %q\n' \
  "$conda_base/etc/profile.d/conda.sh" "$env_name" "$repo_root"
printf '  bash scripts/rpgv/train_rpgv_v6.sh\n\n'
printf '训练前请准备 1024 数据、伪几何与预训练权重；可设置 WWTP_DATA_ROOT / WWTP_PSEUDO_ROOT / RPGV_V6_WORK_ROOT。\n'
