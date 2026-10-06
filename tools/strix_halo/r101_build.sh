#!/usr/bin/env bash
# Build exllamav3_ext against ROCm 10.1 (TheRock wheels in ~/exllamav3-amd/.venv-r101) in a
# separate git worktree, so the production tree's root .so, hipify output and build/ are never
# touched. Log: /tmp/r101_build.log
#
#   bash tools/strix_halo/r101_build.sh        # worktree at ~/exl3-r101, same commit as main
set -u
export PATH="$HOME/.local/bin:$PATH"
SRC=~/exllamav3-amd
WT=~/exl3-r101
V=$SRC/.venv-r101
PYV=$V/bin/python

echo "=== [1/4] worktree ($(date +%T))"
cd $SRC
if [ ! -d $WT ]; then git worktree add -q --detach $WT HEAD; else (cd $WT && git checkout -q --detach $(git -C $SRC rev-parse HEAD)); fi
echo "worktree at $(git -C $WT log --oneline -1)"

echo "=== [2/4] toolchain discovery ($(date +%T))"
SP=$($PYV -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
DEVEL="$SP/_rocm_sdk_devel"
[ -d "$DEVEL" ] || DEVEL=$($PYV -m rocm_sdk path --root 2>/dev/null)
echo "rocm_sdk path --root: $($PYV -m rocm_sdk path --root 2>&1 | tail -1)"
echo "DEVEL=$DEVEL"
ls "$DEVEL/bin/hipcc" "$DEVEL/bin/amdclang++" "$DEVEL/lib/llvm/bin/clang" 2>&1 | sed 's/^/  /'
BC=$(dirname $(find "$DEVEL" -path "*amdgcn/bitcode/ocml.bc" 2>/dev/null | head -1) 2>/dev/null)
echo "bitcode dir: $BC"
"$DEVEL/lib/llvm/bin/clang" --version 2>&1 | head -1 | sed 's/^/  /'
[ -x "$DEVEL/bin/hipcc" ] || { echo "NO hipcc in 10.1 devel wheel"; echo DONE_R101_BUILD_FAIL; exit 1; }

echo "=== [3/4] build ($(date +%T))"
cd $WT
export ROCM_HOME="$DEVEL" ROCM_PATH="$DEVEL" HIP_PATH="$DEVEL" HIPCXX="$DEVEL/bin/hipcc"
export PATH="$DEVEL/bin:$DEVEL/lib/llvm/bin:$V/bin:$PATH"
export HIP_DEVICE_LIB_PATH="$BC"
export HIPCC_COMPILE_FLAGS_APPEND="--rocm-device-lib-path=$BC --rocm-path=$DEVEL"
export PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=${MAX_JOBS:-8}
export EXL3_HIP_DEFINES="${EXL3_HIP_DEFINES:-EXL3_HIP_STG_PAD}"
export PYTHONPATH="$WT"
T0=$(date +%s)
$PYV -m pip install --no-build-isolation --no-deps . > /tmp/r101_build_pip.log 2>&1
RC=$?
echo "pip exit=$RC in $(( $(date +%s) - T0 )) s"
grep -n -E "error:|Error " /tmp/r101_build_pip.log | head -20
grep -c "warning:" /tmp/r101_build_pip.log | sed 's/^/warnings: /'
[ $RC -eq 0 ] || { tail -30 /tmp/r101_build_pip.log; echo DONE_R101_BUILD_FAIL; exit 1; }
cp $SP/exllamav3_ext.cpython-312-x86_64-linux-gnu.so $WT/ && echo "root .so in worktree: $(md5sum $WT/exllamav3_ext*.so | cut -c1-8)"

echo "=== [4/4] runtime smoke WITHOUT LD_PRELOAD and WITHOUT SDK env ($(date +%T))"
env -i HOME=$HOME PATH=/usr/bin:/bin PYTHONPATH=$WT $PYV - <<'PY' 2>&1 | tail -15
import torch, exllamav3_ext as e
print("torch", torch.__version__, "hip", torch.version.hip)
print("ext from:", e.__file__)
x = torch.zeros(4, device="cuda"); torch.cuda.synchronize(); print("alloc OK")
fn = [n for n in dir(e) if "gemv" in n.lower()][:8]
print("gemv bindings:", fn)
try:
    print("exl3_gemv_wmma_family:", e.exl3_gemv_wmma_family(0), "(must be 2)")
except Exception as ex:
    print("exl3_gemv_wmma_family err:", ex)
PY
echo DONE_R101_BUILD
