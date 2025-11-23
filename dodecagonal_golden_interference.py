#!/usr/bin/env python3
# dodecagonal_golden_interference.py
# Minimal 12-fold golden-ratio quasicrystal interference generator
# Developed in Pythonista 3 on iPhone 14 Pro Max
# © 2025 AgapeIntelligence — MIT License

import re
import numpy as np
import warnings
from datetime import datetime
import os
import time
import sys

warnings.filterwarnings("ignore", category=np.ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='ignore')

# === Clean, permanent filenames ===
FIELD_FILE = 'dodecagonal_golden_field.npy'
STATE_FILE = 'dodecagonal_golden_state.txt'
LOCK_FILE  = 'dodecagonal_golden_build.lock'

# Remove stale lock
if os.path.exists(LOCK_FILE):
    print(f"[{datetime.now():%H:%M:%S}] Removing stale lock file")
    try: os.remove(LOCK_FILE)
    except: pass

def try_resume():
    if not (os.path.exists(STATE_FILE) and os.path.exists(FIELD_FILE)):
        return False
    print(f"[{datetime.now():%H:%M:%S}] Resuming from existing field")
    try:
        field = np.load(FIELD_FILE)
        with open(STATE_FILE) as f:
            state = f.read()
        nz_match = re.search(r'Non-zero entries\s*:\s*(\d+)', state)
        nz = int(nz_match.group(1)) if nz_match else len(np.flatnonzero(field))
        print(f"   → {nz:,} non-zero entries | Shape: {field.shape}")
        print(f"   → Memory: {field.nbytes / (1024**3):.3f} GiB")
        return True
    except Exception as e:
        print(f"   → Resume failed ({e}); rebuilding")
        for f in [FIELD_FILE, STATE_FILE]:
            try: os.remove(f)
            except: pass
        return False

if try_resume():
    sys.exit(0)

open(LOCK_FILE, 'w').close()

# === Parameters ===
N_NODES       = 144_000
N_WAVES       = 12
CARRIER_FREQ  = 432.0
PHI           = (1 + 5**0.5) / 2
CHUNK         = 256
THRESHOLD     = 0.5
DOWNSAMPLE    = 32

print(f"[{datetime.now():%H:%M:%S}] Starting 12-fold golden interference build")
print(f"   → {N_NODES:,} nodes × {N_NODES:,} implicit | {N_WAVES} waves | φ-phase offsets")

sparse = {}
total_chunks = (N_NODES + CHUNK - 1) // CHUNK
thetas = np.linspace(0, 2*np.pi, N_WAVES, endpoint=False)
phis   = thetas + np.pi / PHI

adds = 0
t0 = time.time()
BAR = 50

for c, start in enumerate(range(0, N_NODES, CHUNK)):
    end = min(start + CHUNK, N_NODES)
    t_local = np.linspace(0, 1, end - start, endpoint=False)
    carrier = np.exp(2j * np.pi * CARRIER_FREQ * t_local)

    n1 = np.arange(start, end)
    n2 = n1.copy()

    for k in range(N_WAVES):
        fwd = np.outer(carrier, np.sin(n1 * thetas[k]))
        rev = PHI**(-1) * np.outer(carrier, np.cos(n2 * phis[k]))
        wave = fwd + rev

        i, j = np.where(np.abs(wave) > THRESHOLD)
        for ii, jj in zip(i, j):
            key = ((start + ii) // DOWNSAMPLE, (start + jj) // DOWNSAMPLE)
            sparse[key] = sparse.get(key, 0j) + wave[ii, jj]
            adds += 1

    pct = (c + 1) / total_chunks
    bar = "█" * int(BAR * pct) + "░" * (BAR - int(BAR * pct))
    rate = adds / (time.time() - t0 + 1e-8)
    print(f"   → [{bar}] {pct:6.2%} | {len(sparse):,} points | {rate:,.0f} adds/s", end="\r")

print("\n\nBuild complete — densifying")

# === Convert to dense field ===
if sparse:
    rows = [k[0] for k in sparse]
    cols = [k[1] for k in sparse]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    field = np.zeros((h, w), dtype=np.float64)
    for (i, j), val in sparse.items():
        field[i - r0, j - c0] += val.real
else:
    field = np.zeros((1, 1))

# Normalize to [0, 1]
if field.ptp() > 0:
    field = (field - field.min()) / field.ptp()

# === Save ===
np.save(FIELD_FILE, field)
with open(STATE_FILE, 'w') as f:
    f.write(f"Generator: dodecagonal-golden-interference\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write(f"Non-zero entries: {len(sparse)}\n")
    f.write(f"Downsampled grid: {field.shape[1]} × {field.shape[0]}\n")
    f.write(f"Sparsity: {1 - len(sparse)/(field.size):.6%}\n")

try: os.remove(LOCK_FILE)
except: pass

mem_gb = field.nbytes / (1024**3)
print(f"Success! Field saved as: {FIELD_FILE}")
print(f"   → Shape: {field.shape} | Memory: {mem_gb:.3f} GiB")
print(f"   → State saved as: {STATE_FILE}")
print("12-fold golden quasicrystal field ready.")
