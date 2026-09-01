"""Regenerate the derived decode tables in src/ops/kernel/e8_root_codec.cuh.

c_e8_stage1_nib, c_e8_rad_even_i8 and c_e8_rad_odd_i8 are derived from c_e8_stage1_i8x8 and
c_e8_radius_scale. Edit either source table and these three go stale, which
verify_decode_tables.cu will catch. This script reads the header and prints the replacements to
tables.inc; paste them over the existing definitions.

    python gen_decode_tables.py
"""

import re, struct, math

import os
_here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(_here, "..", "..", "src", "ops", "kernel", "e8_root_codec.cuh"),
           encoding="utf-8", errors="replace").read()

def grab_u64(name):
    m = re.search(name + r"\[\d+\]\s*=\s*\{(.*?)\};", src, re.S)
    return [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)ULL", m.group(1))]

stage1 = grab_u64("c_e8_stage1_i8x8")
m = re.search(r"c_e8_radius_scale\[16\]\s*=\s*\{(.*?)\};", src, re.S)
rad = [float(x) for x in re.findall(r"(-?\d+\.\d+)f", m.group(1))]

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def rn(x):            # __float2int_rn: round half to even
    f = math.floor(x)
    d = x - f
    if d > 0.5:  return f + 1
    if d < 0.5:  return f
    return f if f % 2 == 0 else f + 1

def lanes(u):
    return [b - 256 if b > 127 else b for b in u.to_bytes(8, "little")]

# nibble table: nibble i = root_lane_i/2 + 2, in 0..4
nib = []
for u in stage1:
    w = 0
    for i, v in enumerate(lanes(u)):
        assert v in (-4, -2, 0, 2, 4), v
        w |= ((v // 2) + 2) << (4 * i)
    nib.append(w)

# per-radius packed value tables
even, odd = [], []
for s32 in [f32(s) for s in rad]:
    e = 0
    for j in range(5):                     # arg = 2*(j-2)  ->  -4,-2,0,2,4
        e |= (rn(f32((2 * (j - 2)) * s32)) & 0xFF) << (8 * j)
    even.append(e)
    o = 0
    for j in range(6):                     # arg = 2*j-5    ->  -5,-3,-1,1,3,5
        o |= (rn(f32((2 * j - 5) * s32)) & 0xFF) << (8 * j)
    odd.append(o)

def emit_u32(name, vals, per_line=8):
    out = [f"__device__ const std::uint32_t {name}[{len(vals)}] = {{"]
    for i in range(0, len(vals), per_line):
        out.append("    " + " ".join(f"0x{v:08x}u," for v in vals[i:i + per_line]))
    out.append("};")
    return "\n".join(out)

def emit_u64(name, vals, comments):
    out = [f"__device__ const std::uint64_t {name}[{len(vals)}] = {{"]
    for v, c in zip(vals, comments):
        out.append(f"    0x{v:016x}ULL, // {c}")
    out.append("};")
    return "\n".join(out)

ec = [f"rad {i}: " + " ".join(str(rn(f32((2*(j-2))*f32(rad[i])))) for j in range(5)) for i in range(16)]
oc = [f"rad {i}: " + " ".join(str(rn(f32((2*j-5)*f32(rad[i])))) for j in range(6)) for i in range(16)]

open("tables.inc", "w").write(
    emit_u32("c_e8_stage1_nib", nib) + "\n\n" +
    emit_u64("c_e8_rad_even_i8", even, ec) + "\n\n" +
    emit_u64("c_e8_rad_odd_i8", odd, oc) + "\n")
print("nibble table entries:", len(nib), "first:", hex(nib[0]), hex(nib[1]))
print("even[8] (scale .5):", hex(even[8]), "->", ec[8])
print("odd[8]  (scale .5):", hex(odd[8]), "->", oc[8])
print("even[15]:", ec[15]); print("odd[15]:", oc[15])
print("wrote tables.inc")
