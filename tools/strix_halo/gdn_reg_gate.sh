#!/usr/bin/env bash
# Gate for the register-resident GDN recurrent kernel: bit-identity vs the previous .so
# (PPL exact, greedy identical w/o MTP), then speed (isolated verify forward at R=1/4, six-prompt).
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH=$PWD EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
NEW=$PWD/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
OLD=~/ab/so_backup/ship_c20e1f51.so
cp $NEW ~/ab/so_backup/gdn_reg.so
echo "old $(md5sum $OLD | cut -c1-8)  new $(md5sum $NEW | cut -c1-8)"
use() { cp ~/ab/so_backup/$1.so $NEW; }
for so in ship_c20e1f51 gdn_reg; do
    use $so
    echo "################ $so"
    echo "ppl: $(timeout 600 python eval/ppl.py -m ~/models/Qwen3.8-Flash-Next-EXL3 -r 20 -l 1024 2>&1 | grep -oE 'Perplexity: [0-9.]+')"
    for r in 1 4; do R=$r timeout 300 python tools/strix_halo/verify_hostdev.py 2>&1 | grep -E "^R=.*synced"; done
done
echo "## greedy identity, no MTP (kernel A/B: must be IDENTICAL)"
for so in ship_c20e1f51 gdn_reg; do
    use $so
    NOMTP=1 NTOK=256 timeout 600 python - <<'PY' > /tmp/greedy_$so.txt 2>&1
import os, sys, json
sys.path.insert(0, os.getcwd())
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
cfg = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
c = Cache(m, max_num_tokens=4096); m.load(progressbar=False)
g = Generator(model=m, cache=c, tokenizer=tok)
out = []
for p in ["Explain gradient descent in two sentences:", "Write a Python function that reverses a linked list, with comments:"]:
    g.enqueue(Job(input_ids=tok.encode(p, add_bos=True), max_new_tokens=256, sampler=GreedySampler()))
    ids = []
    while g.num_remaining_jobs():
        for r in g.iterate():
            t = r.get("token_ids")
            if t is not None: ids += t.flatten().tolist()
    out.append(ids)
print("IDS", json.dumps(out))
PY
done
python3 -c "
import json
a=[l for l in open('/tmp/greedy_ship_c20e1f51.txt') if l.startswith('IDS')][0]; b=[l for l in open('/tmp/greedy_gdn_reg.txt') if l.startswith('IDS')][0]
print('IDENTICAL' if a==b else 'MISMATCH')"
echo "## greedy identity with MTP"
use ship_c20e1f51; DUMP=/tmp/g_old.json NTOK=512 timeout 900 python tools/strix_halo/greedy_ab.py EXL3_MTP_DRAFT_VOCAB 98304 98304 2>&1 | tail -1
use gdn_reg;       DUMP=/tmp/g_new.json NTOK=512 timeout 900 python tools/strix_halo/greedy_ab.py EXL3_MTP_DRAFT_VOCAB 98304 98304 2>&1 | tail -1
python3 -c "
import json; a=json.load(open('/tmp/g_old.json'))['a']; b=json.load(open('/tmp/g_new.json'))['a']; print('MTP old vs new:', 'IDENTICAL' if a==b else 'MISMATCH')"
echo "## six-prompt, interleaved"
export NDT=3 DC=0.6 DDS=1 NTOK=512
for so in ship_c20e1f51 gdn_reg ship_c20e1f51 gdn_reg; do
    use $so
    echo "$so :: $(LABEL=$so timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E '^mean' | tr '\n' ' ')"
done
use gdn_reg
echo DONE_GDN_GATE
