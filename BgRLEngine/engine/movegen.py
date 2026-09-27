"""Interop wrapper for BgMoveGen NativeAOT shared library.

Two exports:
  generate_successor_states — move generation
  get_starting_position     — starting positions for all variants

BoardState contract:
  points[0..23]  int16  positive = on-roll player, negative = opponent
                        points[0] = 1-point, points[23] = 24-point
                        player moves high→low
  bar_player     int32
  bar_opponent   int32
  off_player     int32
  off_opponent   int32

All states are from the on-roll player's perspective.
generate_successor_states returns states already flipped to the next
player's perspective. get_starting_position returns the initial state
from the on-roll player's perspective (not flipped).
"""
from __future__ import annotations

import ctypes
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np

from engine.state import BoardState, NUM_POINTS


# ── Required DLL version ─────────────────────────────────────────────────────

# Bump this when BgRLEngine requires a new BgMoveGen capability.
# BgMoveGen must report this exact version via get_version() or load_movegen()
# will hard-fail with a clear message.
REQUIRED_MOVEGEN_VERSION: int = 101

# Default DLL location — relative to this file's package root.
# Checked into BgRLEngine/native/BgMoveGen.dll so the repo is self-contained.
# Override via load_movegen(dll_path=...) or the movegen.dll_path config key.
_DEFAULT_DLL_PATH = Path(__file__).parent.parent / "native" / "BgMoveGen.dll"


# ── Variant enum ────────────────────────────────────────────────────────────

class Variant(IntEnum):
    STANDARD   = 0
    NACKGAMMON = 1
    BG960      = 2


# ── Refusal statuses ────────────────────────────────────────────────────────

class _Status(IntEnum):
    """The negative codes a BgMoveGen export returns when it refuses a call.

    Mirrors BgMoveGen's Interop.Status; the one Python spelling of it.
    """
    INVALID_ARGUMENT = -1   # null pointer, negative capacity, die outside 1–6,
                            # unknown starting-position variant
    INVALID_POSITION = -2   # the input board is not a well-formed position
    FAILED           = -3   # anything else, caught inside BgMoveGen


# ── Blittable struct matching BgMoveGen's BgBoardState (C layout) ───────────

class _BgBoardState(ctypes.Structure):
    _fields_ = [
        ("points",       ctypes.c_int16 * NUM_POINTS),
        ("bar_player",   ctypes.c_int32),
        ("bar_opponent", ctypes.c_int32),
        ("off_player",   ctypes.c_int32),
        ("off_opponent", ctypes.c_int32),
    ]


# Starting capacity of the successor buffer. Not a bound: a call whose
# successors outnumber the buffer grows it (see generate_successor_states).
_INITIAL_SUCCESSOR_CAPACITY = 100

# Marshalling buffers, reused across calls. The successor buffer is
# reallocated only when a call's successors outnumber its capacity, and the
# larger buffer is kept, so it only ever grows. (The returned BoardStates are
# allocated per call; only the ctypes buffers are reused.)
_in_buf  = _BgBoardState()
_out_buf = (_BgBoardState * _INITIAL_SUCCESSOR_CAPACITY)()
_pos_buf = _BgBoardState()

_lib: ctypes.CDLL | None = None


# ── Library loading ──────────────────────────────────────────────────────────

def load_movegen(dll_path: str | Path | None = None) -> None:
    """Load the BgMoveGen shared library and verify its version.

    Args:
        dll_path: path to BgMoveGen.dll. Defaults to native/BgMoveGen.dll
                  relative to the repo root. Pass None to use the default.

    Raises:
        FileNotFoundError: DLL not found at the given path.
        RuntimeError:      DLL version does not match REQUIRED_MOVEGEN_VERSION.
    """
    global _lib

    path = Path(dll_path) if dll_path is not None else _DEFAULT_DLL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"BgMoveGen DLL not found: {path}\n"
            f"  If using the default path, copy BgMoveGen.dll to "
            f"native/BgMoveGen.dll in the repo root.\n"
            f"  If using a custom path, set movegen.dll_path in your config."
        )

    _lib = ctypes.CDLL(str(path))

    # ── Wire up exports ──────────────────────────────────────────────────────

    gv = _lib.get_version
    gv.restype  = ctypes.c_int
    gv.argtypes = []

    gss = _lib.generate_successor_states
    gss.restype  = ctypes.c_int
    gss.argtypes = [
        ctypes.POINTER(_BgBoardState),  # input
        ctypes.c_int,                   # die1
        ctypes.c_int,                   # die2
        ctypes.POINTER(_BgBoardState),  # outputBuffer
        ctypes.c_int,                   # bufferCapacity
    ]

    gsp = _lib.get_starting_position
    gsp.restype  = ctypes.c_int
    gsp.argtypes = [
        ctypes.c_int,                   # variant
        ctypes.c_int,                   # seed (-1 = no seed)
        ctypes.POINTER(_BgBoardState),  # output
    ]

    # ── Version check ────────────────────────────────────────────────────────

    actual = _lib.get_version()
    if actual != REQUIRED_MOVEGEN_VERSION:
        _lib = None
        raise RuntimeError(
            f"BgMoveGen version mismatch: "
            f"expected {REQUIRED_MOVEGEN_VERSION}, got {actual}.\n"
            f"  Republish BgMoveGen and copy the new DLL to {path}"
        )


# ── Marshal helpers ──────────────────────────────────────────────────────────

def _to_ctypes(state: BoardState) -> None:
    """Marshal BoardState into the pre-allocated input buffer in-place."""
    for i in range(NUM_POINTS):
        _in_buf.points[i] = int(state.points[i])
    _in_buf.bar_player   = state.bar_player
    _in_buf.bar_opponent = state.bar_opponent
    _in_buf.off_player   = state.off_player
    _in_buf.off_opponent = state.off_opponent


def _from_ctypes(src: _BgBoardState) -> BoardState:
    """Unmarshal a _BgBoardState into a BoardState."""
    s = BoardState()
    s.points       = np.array(src.points, dtype=np.int16)
    s.bar_player   = src.bar_player
    s.bar_opponent = src.bar_opponent
    s.off_player   = src.off_player
    s.off_opponent = src.off_opponent
    return s


# ── Successor-call helpers ───────────────────────────────────────────────────

def _call_generate_successor_states(die1: int, die2: int) -> int:
    """Call the export on _in_buf into _out_buf; return its successor count.

    The count may exceed the buffer's capacity, in which case nothing was
    written. A return below 1 raises (see generate_successor_states).
    """
    n = _lib.generate_successor_states(
        ctypes.byref(_in_buf),
        die1, die2,
        _out_buf,
        len(_out_buf),
    )
    if n < 1:
        raise _successor_refusal(n, die1, die2)
    return n


def _grow_successor_buffer(capacity: int) -> None:
    """Replace the successor buffer with one of the given capacity."""
    global _out_buf
    _out_buf = (_BgBoardState * capacity)()


def _successor_refusal(rc: int, die1: int, die2: int) -> Exception:
    """The exception for a generate_successor_states return below 1."""
    if rc == 0:
        return RuntimeError(
            "BgMoveGen: generate_successor_states returned 0, which its "
            "contract never does (a pass is one successor)"
        )
    try:
        status = _Status(rc)
    except ValueError:
        return RuntimeError(
            f"BgMoveGen: generate_successor_states returned {rc}, "
            f"a status its contract does not define"
        )
    refused = f"BgMoveGen refused generate_successor_states: {status.name} ({rc})"
    if status is _Status.INVALID_ARGUMENT:
        return ValueError(
            f"{refused}, an argument outside its domain (dice {die1}, {die2})"
        )
    if status is _Status.INVALID_POSITION:
        return ValueError(
            f"{refused}, the input board is not a well-formed position"
        )
    return RuntimeError(
        f"{refused}, a failure inside BgMoveGen that no argument explains"
    )


# ── Public API ───────────────────────────────────────────────────────────────

def generate_successor_states(
    state: BoardState,
    die1:  int,
    die2:  int,
) -> list[BoardState]:
    """Return every successor state, each from the perspective of the player now on roll.

    Pass is returned as a single successor (flipped current state, no moves applied).
    Always returns at least one state, and never a partial list: BgMoveGen
    returns the full successor count, writing nothing when that count exceeds
    the buffer's capacity, so this grows the buffer to the count and calls
    again.

    Raises:
        ValueError:   BgMoveGen refused the call's input — a die outside 1–6
                      (INVALID_ARGUMENT) or a board that is not a well-formed
                      position (INVALID_POSITION).
        RuntimeError: BgMoveGen failed internally (FAILED), or broke its
                      contract (a return of 0, an undefined status, or a
                      count that exceeds the grown buffer).
    """
    assert _lib is not None, "Call load_movegen() before use"

    _to_ctypes(state)
    n = _call_generate_successor_states(die1, die2)
    if n > len(_out_buf):
        _grow_successor_buffer(n)
        grown = len(_out_buf)
        n = _call_generate_successor_states(die1, die2)
        if n > grown:
            raise RuntimeError(
                f"BgMoveGen: generate_successor_states returned {n} with a "
                f"buffer grown to the {grown} it asked for; the count of one "
                f"input must not change between calls"
            )
    return [_from_ctypes(_out_buf[i]) for i in range(n)]


def get_starting_position(
    variant: Variant | int = Variant.STANDARD,
    seed:    Optional[int] = None,
) -> BoardState:
    """Return a starting BoardState for the requested variant.

    Args:
        variant: Variant.STANDARD, NACKGAMMON, or BG960.
        seed:    RNG seed for Bg960 reproducibility.
                 Ignored for standard and nackgammon.
                 None = no seed.

    Returns:
        BoardState from the on-roll player's perspective (not flipped).

    Raises:
        ValueError:   unknown variant.
        RuntimeError: BgMoveGen returned an error.
    """
    assert _lib is not None, "Call load_movegen() before use"

    rc = _lib.get_starting_position(
        int(variant),
        seed if seed is not None else -1,
        ctypes.byref(_pos_buf),
    )
    if rc == _Status.INVALID_ARGUMENT:
        raise ValueError(f"BgMoveGen: unknown variant {variant}")
    if rc != 0:
        raise RuntimeError(f"BgMoveGen: get_starting_position returned {rc}")

    return _from_ctypes(_pos_buf)