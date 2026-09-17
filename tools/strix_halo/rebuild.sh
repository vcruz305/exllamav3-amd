#!/usr/bin/env bash
# Rebuild the extension after editing hgemm.cu / hc_mix.cu. Build shell only — bench in a fresh shell.
set -u
cd ~/exllamav3-amd
source env.sh build > /dev/null
export PYTHONPATH="$PWD"
export EXL3_HIP_DEFINES="${EXL3_HIP_DEFINES:-EXL3_HIP_STG_PAD}"
# Regenerate hipify twins of the files we edit here (the .hip is gitignored, generated from the .cu)
for f in hgemm hc_mix; do rm -f exllamav3/exllamav3_ext/$f.hip; find build -name "$f*.o" -delete 2>/dev/null; done
echo "[build] start $(date +%T) defines=$EXL3_HIP_DEFINES"
.venv/bin/python -m pip install --no-build-isolation --no-deps . > /tmp/build_skinny.log 2>&1
echo "[build] exit=$? $(date +%T)"
grep -n "error:" /tmp/build_skinny.log | head -20
cp .venv/lib/python3.12/site-packages/exllamav3_ext.cpython-312-x86_64-linux-gnu.so ./exllamav3_ext.cpython-312-x86_64-linux-gnu.so && echo "[build] root .so refreshed (repo-root copy shadows site-packages)"
