"""
bench_encode.py — Hot-path microbenchmark for BgRLEngine
Validates encode_board_batch correctness and measures speedup.

Usage (from BgRLEngine\BgRLEngine\):
    python tests/bench_encode.py
    python tests/bench_encode.py --reps 20000
"""

import sys
import timeit
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reps", type=int, default=10_000)
args = parser.parse_args()

try:
    import torch
except ImportError:
    sys.exit("ERROR: torch not found. Activate the project venv first.")

sys.path.insert(0, ".")

try:
    from engine.state import BoardState, encode_board, encode_board_batch
except ImportError as e:
    sys.exit(f"ERROR importing engine.state: {e}")

try:
    from engine.dice import generate_plays
except ImportError as e:
    sys.exit(f"ERROR importing engine.dice: {e}")

try:
    from engine.network import TDNetwork, compute_equity
except ImportError as e:
    sys.exit(f"ERROR importing engine.network: {e}")

# ---------------------------------------------------------------------------
# Mid-game state
# ---------------------------------------------------------------------------
def make_midgame_state() -> BoardState:
    state = BoardState()
    state.points = np.zeros(24, dtype=np.int16)
    state.points[0]  =  2
    state.points[5]  =  3
    state.points[6]  =  3
    state.points[7]  =  2
    state.points[11] =  2
    state.points[18] =  1
    state.points[22] =  1
    state.points[23] = -2
    state.points[17] = -3
    state.points[16] = -3
    state.points[12] = -2
    state.points[4]  = -1
    state.bar_player   = 1
    state.bar_opponent = 0
    state.off_player   = 0
    state.off_opponent = 0
    return state

STATE      = make_midgame_state()
DIE1, DIE2 = 3, 5
plays      = generate_plays(STATE, DIE1, DIE2)
N          = max(len(plays), 1)
SUCCESSORS = [STATE] * N

device = torch.device("cpu")
net    = TDNetwork()
net.eval()

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
print(f"\nBgRLEngine hot-path microbenchmark")
print(f"  Successor count N : {N}")
print(f"  Reps              : {args.reps:,}")
print()
print("Correctness check ...", flush=True)

individual = np.stack([encode_board(s) for s in SUCCESSORS])
batched    = encode_board_batch(SUCCESSORS)

if np.allclose(individual, batched, atol=1e-6):
    print("  ✓ encode_board_batch matches encode_board x N\n")
else:
    max_diff = np.abs(individual - batched).max()
    print(f"  ✗ MISMATCH — max diff: {max_diff:.2e}")
    print("  Fix encode_board_batch before using in training.\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Timing phases
# ---------------------------------------------------------------------------
REPS = args.reps

batch_cached = torch.from_numpy(encode_board_batch(SUCCESSORS))

def old_encode_and_stack():
    return np.stack([encode_board(s) for s in SUCCESSORS])

def new_encode_batch():
    return encode_board_batch(SUCCESSORS)

def forward_only():
    with torch.no_grad():
        return net(batch_cached)

def old_full_select_play():
    features = [encode_board(s) for s in SUCCESSORS]
    batch = torch.from_numpy(np.stack(features))
    with torch.no_grad():
        outputs = net(batch)
    equities = -compute_equity(outputs)
    return equities.argmax().item()

def new_full_select_play():
    batch = torch.from_numpy(encode_board_batch(SUCCESSORS))
    with torch.no_grad():
        outputs = net(batch)
    equities = -compute_equity(outputs)
    return equities.argmax().item()

print("Running old: encode_board() x N + np.stack() ...", flush=True)
t_old_encode = timeit.timeit(old_encode_and_stack, number=REPS) / REPS * 1e6

print("Running new: encode_board_batch() ...", flush=True)
t_new_encode = timeit.timeit(new_encode_batch, number=REPS) / REPS * 1e6

print("Running network.forward(batch=N, CPU) ...", flush=True)
t_forward = timeit.timeit(forward_only, number=REPS) / REPS * 1e6

print("Running old full select_play() path ...", flush=True)
t_old_full = timeit.timeit(old_full_select_play, number=REPS) / REPS * 1e6

print("Running new full select_play() path ...", flush=True)
t_new_full = timeit.timeit(new_full_select_play, number=REPS) / REPS * 1e6

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print()
print(f"{'Phase':<40}  {'µs/call':>10}  {'':>16}")
print("-" * 72)
print(f"{'old: encode_board() x N + stack':<40}  {t_old_encode:>10.2f}")
print(f"{'new: encode_board_batch()':<40}  {t_new_encode:>10.2f}  "
      f"  {t_old_encode/t_new_encode:.1f}x faster")
print(f"{'network.forward(batch=N, CPU)':<40}  {t_forward:>10.2f}")
print("-" * 72)
print(f"{'old full select_play()':<40}  {t_old_full:>10.2f}")
print(f"{'new full select_play()':<40}  {t_new_full:>10.2f}  "
      f"  {t_old_full/t_new_full:.1f}x faster")
print()