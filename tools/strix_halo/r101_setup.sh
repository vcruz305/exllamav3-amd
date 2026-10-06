#!/usr/bin/env bash
# ROCm 10.1 side-by-side environment for exllamav3-amd on gfx1151.
#
# Creates ~/exllamav3-amd/.venv-r101 with torch 2.14.0+rocm10.1.0 (device-gfx1151) and the
# matching rocm[libraries,devel] SDK in the SAME venv (TheRock wheels ship hipcc + bitcode +
# headers, so the separate .venv-gfx1151 build venv is not needed). The production .venv
# (torch 2.10.0+rocm7.0, nightly 7.13 SDK) is not touched, so every number can be A/B'd.
#
# Then builds the extension against it into a separate .so (never overwrites the shipped one)
# and runs the smoke checks. Log: /tmp/r101_setup.log
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/exllamav3-amd
V=.venv-r101
IDX=https://stable.repo.amd.com/rocm/whl-next/
PY=3.12

echo "=== [1/5] venv ($(date +%T))"
[ -x $V/bin/python ] || uv venv -q --python $PY $V
$V/bin/python -m ensurepip -q 2>/dev/null || true
uv pip install -q --python $V/bin/python pip setuptools wheel ninja packaging 2>&1 | tail -2

echo "=== [2/5] torch 2.14.0+rocm10.1.0 [device-gfx1151] + rocm 10.1.0 SDK ($(date +%T))"
UV_HTTP_TIMEOUT=900 uv pip install --python $V/bin/python --index-url $IDX \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "rocm[libraries,devel,device-gfx1151]==10.1.0" 2>&1 | tail -5

echo "=== [3/5] runtime deps the fork imports ($(date +%T))"
uv pip install -q --python $V/bin/python --index-url https://pypi.org/simple \
  numpy safetensors tokenizers sentencepiece pillow marisa-trie rich pydantic \
  formatron kbnf pyperclip prompt_toolkit 2>&1 | tail -2

echo "=== [4/5] smoke: torch / HIP / arch / compute, WITHOUT the Ubuntu HSA LD_PRELOAD ($(date +%T))"
# The 7.0 wheel needed LD_PRELOAD of Ubuntu's libhsa-runtime64 (bundled one segfaulted on the
# first allocation). Test whether 10.1's bundled runtime still needs that crutch.
$V/bin/python - <<'PY' 2>&1 | tail -12
import torch, time
print("torch", torch.__version__, "hip", torch.version.hip)
print("arch_list has gfx1151:", "gfx1151" in torch.cuda.get_arch_list())
print("device:", torch.cuda.get_device_name(0))
p = torch.cuda.get_device_properties(0)
print(f"total_memory {p.total_memory/2**30:.1f} GiB  CUs {p.multi_processor_count}")
a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
torch.cuda.synchronize(); (a @ a); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20): a @ a
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 20
print(f"fp16 4096^3 matmul: {2*4096**3/dt/1e12:.1f} TFLOP/s  (rocm7.0 baseline ~31-34)")
print("PREFLIGHT_OK")
PY
echo "  (exit $? without LD_PRELOAD)"
echo DONE_R101_SETUP
