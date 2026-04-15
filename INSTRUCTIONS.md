# BgRLEngine — Subproject Instructions

> See [`../CLAUDE.md`](../CLAUDE.md) for session conventions.
> See [`../INSTRUCTIONS.md`](../INSTRUCTIONS.md) for cross-cutting status and the dependency graph.
> See [`../VISION.md`](../VISION.md) for mission and principles.

## Stack

Python 3.11 / PyTorch 2.5.1+cu121 (CUDA) / Visual Studio 2026 Python workload / Windows.

## Solution

`D:\Users\Hal\Documents\Visual Studio 2026\Projects\backgammon\BgRLEngine\BgRLEngine.slnx`

The `.slnx` wraps the `BgRLEngine.pyproj` Python project for VS tooling; all real work happens in the Python tree.

## Repo

https://github.com/halheinrich/BgRLEngine — branch `main`.

## Depends on

- **BgMoveGen** — move generation and all variant starting positions, consumed at runtime as a native DLL via ctypes (`engine/movegen.py`). No managed interop, no project reference.

## Directory tree

```
BgRLEngine/
├── BgRLEngine.slnx
├── INSTRUCTIONS.md
└── BgRLEngine/
    ├── BgRLEngine.pyproj
    ├── main.py                     entry point (CLI training driver)
    ├── compare_configs.py
    ├── profile_training.py
    ├── configs/
    │   ├── default.yaml            base hyperparameters
    │   ├── dmp.yaml                Standard DMP
    │   ├── nackgammon_dmp.yaml
    │   ├── bg960_dmp.yaml
    │   ├── money.yaml
    │   ├── gammon_seeking.yaml
    │   └── gammon_avoiding.yaml
    ├── engine/
    │   ├── state.py                BoardState, 303-feature encoding
    │   ├── movegen.py              BgMoveGen ctypes wrapper + version check
    │   ├── network.py              TDNetwork (PyTorch), equity computation
    │   ├── game.py                 self-play game simulation
    │   ├── setup_generator.py      Bg960 starting position generator
    │   └── dice.py                 pure-Python move generation (tests only)
    ├── training/
    │   └── td_trainer.py           TD(λ) loop, evaluation, SPRT, plateau detection
    ├── utils/
    │   └── sprt.py                 standalone SPRT implementation
    ├── tests/
    │   ├── test_core.py            pytest (requires torch)
    │   ├── run_tests.py            standalone (no torch)
    │   ├── bench_encode.py
    │   └── verify_bg960.py
    ├── native/                     published BgMoveGen DLL (gitignored)
    └── output/                     training artifacts (gitignored)
```

## Architecture

**TD-Gammon style neural network, self-play from scratch.** Feedforward, 2 hidden layers × 256 units, ReLU hidden / Sigmoid output, ~145K parameters. Trained via TD(λ), λ ≈ 0.7, PyTorch on CUDA. ε-greedy move selection during training, greedy at inference. One network plays both sides; weights update after each game.

**Input encoding.** 303 features: 24 points × 6-unit thermometer+overflow, bar, borne off, pip ratio, race flag, checker counts. No variant flags — Standard, Nackgammon, and Bg960 share identical rules (hitting, bar re-entry, bearing off); only starting positions differ, so the position itself is sufficient input. `encode_board_batch` is vectorized (7.5× encode / 2.5× select_play vs the per-position path).

**Output vector.** 6 values: `[P(win), P(win gammon), P(win backgammon), P(lose), P(lose gammon), P(lose backgammon)]`. All sub-engines share this shape so the routing tree can compose them uniformly.

**Two-level routing tree (design target — most branches not yet trained).**

```
Position
├── Checker play
│   ├── Race engine          mathematical, exhaustive
│   └── General engine       TD-Gammon NN  ← Phase 1 scope
└── Cube decision
    ├── Race cube engine     mathematical, pip count
    ├── Money play engine    single NN (gammon value fixed)
    └── Match play engine    four MET-classified sub-engines:
                             Seeking / Averse / Indifferent / Balanced
```

The race/contact seam is a hard boundary — zero contact = race, else general engine. New seams (containment, back game, prime-vs-prime) can be added as additional router branches, each a specialist NN with the same 6-value output.

**Variants.** Standard, Nackgammon, Bg960. BgMoveGen is the sole source of truth for every starting position — the Python side never constructs its own. Bg960 positions are symmetric, mirror-free, ≥2 checkers per occupied point, ≥1 point per quadrant, minimum pip count 100, weighted toward 4–5 made points.

**Training strategy.** Bg960 base model (diverse starts → general-purpose player) then fine-tune for Standard and Nackgammon (mid-game positions converge, so fine-tuning is cheap).

**SPRT promotion test.** Level N+1 = frozen checkpoint that beats Level N. Rolling evaluation win rate ≥ 0.70 triggers SPRT with H₀: p ≤ 0.70, H₁: p ≥ 0.76, α = 0.05, β = 0.10, hard cap 2000 games. Metric is game win rate, not point rate. α < β because false promotions corrupt the curriculum ladder; missed promotions just waste GPU and self-correct.

**Plateau detection.** Per-level self-play budget = 500K × 1.5^level. Evaluation every 5000 training games against the previous level (100-game match). Staleness signal: <2pp improvement in rolling win rate across any 200K-game window → early termination. Three failed SPRTs within one level → halve remaining budget.

**Gammon classification is rule-based.** MET lookup produces Seeking / Averse / Indifferent / Balanced — the MET is solved mathematics, so asking the NN to rediscover match-equity arithmetic wastes training capacity.

## Public API

The observable surface consumed outside `td_trainer.py` is small:

```python
# engine/state.py
class BoardState:  # 303-feature encoding, variant-agnostic
    def encode(self) -> np.ndarray
# engine/state.py
def encode_board_batch(states) -> torch.Tensor  # vectorized

# engine/movegen.py
REQUIRED_MOVEGEN_VERSION: int = 100
class MoveGen:  # ctypes wrapper over native BgMoveGen.dll
    def generate_states(...); def next_move(...); def get_version() -> int

# engine/network.py
class TDNetwork(nn.Module):
    def equity(state) -> torch.Tensor  # 6-value output

# training/td_trainer.py
class TdTrainer:
    def train(max_games: int | None)
    def evaluate_against(opponent: TDNetwork) -> float
# utils/sprt.py
class Sprt:  # p0, p1, alpha, beta, hard_cap; returns Accept / Reject / Continue
```

CLI entry point:

```
python main.py --config configs/<name>.yaml [--max-games N]
```

## Pitfalls

- **Two-network evaluation.** `play_game()` takes an optional `opponent` parameter. Without it, both sides use the same network (self-play for training). `evaluate_against` and `_run_sprt` **must** pass the opponent network — otherwise they silently test self-play and always report ~50%.
- **BgMoveGen DLL publish.** `native/` is gitignored; after cloning, publish explicitly from the BgMoveGen project, not the solution: `dotnet publish BgMoveGen/BgMoveGen.csproj -c Release -r win-x64 -o "<repo>/BgRLEngine/BgRLEngine/native"`. Solution-level publish is managed-only and won't emit the unmanaged DLL.
- **BgMoveGen version lock.** `REQUIRED_MOVEGEN_VERSION` in `engine/movegen.py` must match BgMoveGen's `get_version()` return value (currently **100**). Bump both sides together on any breaking interop change — the wrapper asserts on load.
- **UTF-8 on Windows.** Every `open()` for config/data files must pass `encoding="utf-8"`. Default `cp1252` chokes on YAML comments containing non-ASCII.
- **PyTorch CUDA query.** Use `torch.cuda.get_device_properties(0).total_memory` — there is no `total_mem` attribute.
- **PowerShell + `.bat`.** From PowerShell, run `cmd /c setup_env.bat`; `.\setup_env.bat` does not work. PowerShell also rejects `&&` as a statement separator — use two separate commands.
- **No variant flags in state encoding.** Tempting to add a one-hot "this is Nackgammon" feature when debugging variant-specific regressions; don't. Rules are identical across variants and the 303-feature encoding is load-bearing for cross-variant weight transfer.

## Subproject-internal next steps

- **Best-of-3 series promotion metric.** Level 4 is the empirical ceiling for every variant under the single-game win-rate metric — dice variance swamps skill signal at Level 4→5. Design questions are settled; implementation pending.
- **Config-specific promotion metrics.** Match win rate, equity error, gammon rate — the 75% per-game threshold is unreachable at higher levels and one metric does not fit all configs.
- **Multi-core parallelization of self-play.** Currently single-process; training throughput is the bottleneck for higher-level runs.
- **ONNX export of trained models.** Required by the planned BgInference consumer; not yet implemented.
