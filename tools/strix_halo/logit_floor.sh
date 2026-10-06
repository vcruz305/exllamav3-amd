#!/usr/bin/env bash
# Noise floor for logit_parity: what does a pure summation-order change look like on ROCm 7?
set -u
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
cd ~/exllamav3-amd
R7="env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 PYTHONPATH=$PWD .venv/bin/python"
CMP="env PYTHONPATH=$PWD MODE=cmp .venv-r101/bin/python tools/strix_halo/logit_parity.py"
echo "## r7 rerun (determinism)"
MODE=dump OUT=/tmp/lp_r7b.pt $R7 tools/strix_halo/logit_parity.py 2>&1 | grep -E "saved|Error"
A=/tmp/lp_r7.pt B=/tmp/lp_r7b.pt $CMP 2>&1 | tail -7
echo "## r7 torch norms (EXL3_TRITON_NORM=0 EXL3_GR_TRITON=0): summation-order-only change"
EXL3_TRITON_NORM=0 EXL3_GR_TRITON=0 MODE=dump OUT=/tmp/lp_r7tn.pt $R7 tools/strix_halo/logit_parity.py 2>&1 | grep -E "saved|Error"
A=/tmp/lp_r7.pt B=/tmp/lp_r7tn.pt $CMP 2>&1 | tail -7
echo "## r7 F32OUT_VIA_F16=0 (fp16 rounding of GDN projections removed)"
EXL3_HIP_F32OUT_VIA_F16=0 MODE=dump OUT=/tmp/lp_r7f.pt $R7 tools/strix_halo/logit_parity.py 2>&1 | grep -E "saved|Error"
A=/tmp/lp_r7.pt B=/tmp/lp_r7f.pt $CMP 2>&1 | tail -7
echo DONE_FLOOR
