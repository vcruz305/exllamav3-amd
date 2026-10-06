#!/usr/bin/env python
"""
Per-kernel register / LDS / scratch / spill metadata for the gfx1151 code objects embedded in a
built extension (.so) or object (.o). Diff two builds (e.g. ROCm 7 vs 10.1 toolchains).

  kernel_meta_so.py A.so [B.so] [substring]

Uses the llvm tools shipped in the 10.1 SDK wheel (works on either toolchain's output).
"""
import os, subprocess, sys, struct, re, tempfile, glob

LLVM = os.environ.get("LLVM_BIN") or (glob.glob(os.path.expanduser(
    "~/exllamav3-amd/.venv-r101/lib/python3.12/site-packages/_rocm_sdk_*/lib/llvm/bin")) + [""])[0]

def code_objects(path):
    """All gfx1151 code objects in every offload bundle of path (.hip_fatbin section)."""
    d = tempfile.mkdtemp(prefix="km_")
    fat = os.path.join(d, "fat.bin")
    subprocess.run([f"{LLVM}/llvm-objcopy", "--dump-section", f".hip_fatbin={fat}", path, os.path.join(d, "junk")],
                   check=True, capture_output=True)
    data = open(fat, "rb").read()
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    cos, pos = [], 0
    while True:
        pos = data.find(magic, pos)
        if pos < 0: break
        n = struct.unpack_from("<Q", data, pos + 24)[0]
        off = pos + 32
        for _ in range(n):
            o, sz, idlen = struct.unpack_from("<QQQ", data, off); off += 24
            tid = data[off:off + idlen].decode(errors="replace"); off += idlen
            if "gfx1151" in tid:
                f = os.path.join(d, f"co{len(cos)}.co"); open(f, "wb").write(data[pos + o: pos + o + sz]); cos.append(f)
        pos += len(magic)
    return cos

def meta(path, want):
    out = {}
    for co in code_objects(path):
        notes = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", co], capture_output=True, text=True).stdout
        # One YAML list item per kernel under amdhsa.kernels; the first key varies by LLVM
        # version (.agpr_count on older, .args on newer), so split on the item marker itself.
        body = notes.split("amdhsa.kernels:", 1)[-1].split("amdhsa.target", 1)[0]
        for blk in re.split(r"\n  - \.", body)[1:]:
            m = re.search(r"\n    \.name:\s+(\S+)", blk)
            if not m or want not in m.group(1): continue
            g = lambda k: (re.search(rf"\n    \.{k}:\s+(\S+)", blk) or [None, "?"])[1]
            out[m.group(1)] = dict(vgpr=g("vgpr_count"), sgpr=g("sgpr_count"), lds=g("group_segment_fixed_size"),
                                   scratch=g("private_segment_fixed_size"), spill_v=g("vgpr_spill_count"),
                                   spill_s=g("sgpr_spill_count"))
    return out

def short(n):
    n = subprocess.run([f"{LLVM}/llvm-cxxfilt"], input=n, capture_output=True, text=True).stdout.strip() or n
    n = re.sub(r"\(.*$", "", n).replace("(anonymous namespace)::", "").replace("exl3_moe_mk_ns::", "")
    return n[:72]

args = [a for a in sys.argv[1:]]
want = args.pop() if len(args) in (2, 3) and not args[-1].endswith((".so", ".o")) else ""
A = meta(args[0], want); B = meta(args[1], want) if len(args) > 1 else {}
keys = sorted(set(A) | set(B), key=short)
f = lambda d: f"v{d['vgpr']:>4} s{d['sgpr']:>4} lds{d['lds']:>6} scr{d['scratch']:>5} sp{d['spill_v']}/{d['spill_s']}" if d else "-" * 34
print(f"{'kernel':72}  {'A':34}  {'B':34}")
for k in keys:
    a, b = A.get(k), B.get(k)
    mark = "" if (a and b and a == b) or not B else "  <--"
    print(f"{short(k):72}  {f(a):34}  {f(b):34}{mark}")
