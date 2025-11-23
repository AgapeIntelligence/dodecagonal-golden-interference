#!/usr/bin/env python3
# dodecagonal_golden_interference.py
# Minimal 12-fold golden-ratio quasicrystal interference generator
# Originally developed in Pythonista 3 on iPhone 14 Pro Max

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

# === Proper, clean filenames ===
FIELD_FILE = 'dodecagonal_golden_field.npy'
STATE_FILE = 'dodecagonal_golden_state.txt'
LOCK_FILE  = 'dodecagonal_golden_build.lock'

# Remove stuck lock if present
if os.path.exists(LOCK_FILE):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Removing stale lock")
    try: os.remove(LOCK_FILE)
    except: pass

def try_resume():
    if not (os.path.exists(STATE_FILE) and os.path.exists(FIELD_FILE)):
        return False
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming from cached field")
    try:
        field = np.load(FIELD_FILE)
        with open(STATE_FILE) as f:
            state = f.read()
        m_nz = re.search(r'NZ\s*:\s*(\d+)', state)
        nonzero = int(m_nz.group(1)) if m_nz else len(np.argwhere(field!=0))
        print(f"   → {nonzero:,} nonzero elements | Shape: {field.shape}")
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

# Parameters
GRID_NODES     = 144_000
LEY_LINES      = 12
CARRIER_FREQ   = 432.0
GOLDEN_RATIO   = (1 + 5**0.5) / 2
CHUNK_SIZE     = 256
THRESHOLD      = 0.5
DOWNSAMPLE     = 32

print(f"[{datetime.now().strftime('%H:%M:%S')}] Building 12-fold golden interference field")
print(f"   → {GRID_NODES:,} nodes | {LEY_LINES} waves | {CARRIER_FREQ} carrier | φ offsets")

sparse_accum = {}
total_chunks = (GRID_NODES + CHUNK_SIZE - 1) // CHUNK_SIZE
thetas = np.linspace(0, 2*np.pi, LEY_LINES, endpoint=False)
phis   = thetas + np.pi / GOLDEN_RATIO

add_count = 0
start = time.time()
BAR_LEN = 50

for chunk_idx, chunk_start in enumerate(range(0, GRID_NODES, CHUNK_SIZE)):
    chunk_end = min(chunk_start + CHUNK_SIZE, GRID_NODES)
    t = np.linspace(0, 1, chunk_end - chunk_start, endpoint=False)
    carrier = np.exp(1j * 2 * np.pi * CARRIER_FREQ * t)

    n_row = np.arange(chunk_start, chunk_end)
    n_col = n_row.copy()

    for ley in range(LEY_LINES):
        sin_fwd = np.sin(n_row * thetas[ley])
        cos_rev = np.cos(n_col * phis[ley])

        fwd = np.outer(carrier, sin_fwd)
        rev = 0.618 * np.outer(carrier, cos_rev)
        wave = fwd + rev

        i_idx, j_idx = np.where(np.abs(wave) > THRESHOLD)
        for ii, jj in zip(i_idx, j_idx):
            i = chunk_start + ii
            j = chunk_start + jj
            key = (i // DOWNSAMPLE, j // DOWNSAMPLE)
            sparse_accum[key] = sparse_accum.get(key, 0j) + wave[ii, jj]
            add_count += 1

    pct = (chunk_idx + 1) / total_chunks
    bar = "█" * int(BAR_LEN * pct) + "░" * (BAR_LEN - int(BAR_LEN * pct))
    rate = add_count / max(time.time() - start, 1e-6)
    print(f"   → [{bar}] {pct*100:5.1f}% | {len(sparse_accum):,} unique | {rate:,.0f} adds/s", end="\r")

print("\nBuild complete")

# Build dense field
if sparse_accum:
    rows = [k[0] for k in sparse_accum]
    cols = [k[1] for k in sparse_accum]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    h, w = max_r - min_r + 1, max_c - min_c + 1
    field = np.zeros((h, w), dtype=np.float64)
    for (i, j), val in sparse_accum.items():
        field[i - min_r, j - min_c] += np.real(val)
else:
    field = np.zeros((1, 1))

# Normalize
if field.max() > field.min():
    field = (field - field.min()) / (field.max() - field.min())

# Save
np.save(FIELD_FILE, field)
with open(STATE_FILE, 'w') as f:
    f.write(f"NZ: {len(sparse_accum)} | Shape: {field.shape} | Generated: {datetime.now().isoformat()}\n")

try: os.remove(LOCK_FILE)
except: pass

mem = field.nbytes / (1024**3)
print(f"Field saved → {FIELD_FILE} ({field.shape}, {mem:.3f} GiB)")
print("Done.")
