#!/usr/bin/env bash
# Output identity for EXL3_MTP_DRAFT_VOCAB. With MTP on, the verify width depends on what was
# drafted, so near-tie flips are possible even for draft-only changes; the noise floor is a
# draft-only change that cannot alter any logit: EXL3_MTP_EMB_PINNED (same values, different
# upload path). Then the draft-vocab knob itself, at 512 tokens.
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH=$PWD NTOK=512
echo "## floor: same knob twice";            python tools/strix_halo/greedy_ab.py EXL3_MTP_DRAFT_VOCAB 0 0 2>&1 | tail -4
echo "## EXL3_MTP_DRAFT_VOCAB 0 vs 98304";   python tools/strix_halo/greedy_ab.py EXL3_MTP_DRAFT_VOCAB 0 98304 2>&1 | tail -4
echo "## EXL3_MTP_DRAFT_VOCAB 0 vs 65536";   python tools/strix_halo/greedy_ab.py EXL3_MTP_DRAFT_VOCAB 0 65536 2>&1 | tail -4
echo DONE_IDENT
