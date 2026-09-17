#!/usr/bin/env bash
# Rebuild the extension after editing hgemm.cu / hc_mix.cu. Build shell only — bench in a fresh shell.
set -u
cd ~/exllamav3-amd
source env.sh build > /dev/null
export PYTHONPATH="$PWD"
export EXL3_HIP_DEFINES="${EXL3_HIP_DEFINES:-EXL3_HIP_STG_PAD}"
# Regenerate hipify twins of the files we edit here (the .hip is gitignored, generated from the .cu)
for f in hgemm hc_mix; do rm -f exllamav3/exllamav3_ext/$f.hip exllamav3/exllamav3_ext/${f}_hip.cuh; find build -name "$f*.o" -delete 2>/dev/null; done
rm -f exllamav3/exllamav3_ext/bindings_hip.cpp; find build -name "bindings*.o" -delete 2>/dev/null
if [ "${FULL:-0}" = "1" ]; then
    # header change (exl3_gemv_kernel.cuh): every HIP TU that includes it must rebuild
    find exllamav3/exllamav3_ext -name '*.hip' -delete; find exllamav3/exllamav3_ext -name '*_hip.cuh' -delete; find exllamav3/exllamav3_ext -name 'hip_*.cuh' -path '*/hip/*' -prune -o -name '*_hip.*' -print -delete >/dev/null 2>&1
    rm -rf build
fi
echo "[build] start $(date +%T) defines=$EXL3_HIP_DEFINES"
.venv/bin/python -m pip install --no-build-isolation --no-deps . > /tmp/build_skinny.log 2>&1
echo "[build] exit=$? $(date +%T)"
grep -n "error:" /tmp/build_skinny.log | head -20
cp .venv/lib/python3.12/site-packages/exllamav3_ext.cpython-312-x86_64-linux-gnu.so ./exllamav3_ext.cpython-312-x86_64-linux-gnu.so && echo "[build] root .so refreshed (repo-root copy shadows site-packages)"
