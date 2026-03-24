# BgRLEngine — Project Instructions

Part of the Backgammon tools ecosystem: https://github.com/halheinrich/backgammon
**After committing here, return to the Backgammon Umbrella project to update hashes and instructions doc.**

## Repo
https://github.com/halheinrich/BgRLEngine
**Branch:** main
**Current commit:** 5365dc7

## Stack
Python 3.11 / PyTorch 2.5.1+cu121 / Visual Studio 2026 (Python workload) / Windows

## Hardware
- CPU: Intel i9-12900K, 16 cores / 24 threads
- RAM: 64GB
- GPU: NVIDIA GeForce RTX 3060 12GB VRAM (CUDA 12.1)
- Cloud compute: optional for long training runs

## Purpose
A reinforcement learning engine that plays backgammon and its variants at or above XG/GNU strength. Trained via self-play from scratch (tabula rasa). Standalone — no ecosystem integration in Phase 1.

## Project structure
```
BgRLEngine/
├── BgRLEngine.pyproj          ← VS 2026 project file
├── BgRLEngine.slnx            ← VS 2026 solution (in parent dir)
├── main.py                    ← entry point
├── configs/
│   └── default.yaml           ← training hyperparameters
├── engine/
│   ├── state.py               ← BoardState, 303-feature encoding, flip_perspective
│   ├── setup_generator.py     ← Bg960 starting position generator
│   ├── network.py             ← TDNetwork (PyTorch), equity computation
│   ├── dice.py                ← dice rolling, legal move generation
│   └── game.py                ← self-play game simulation
├── training/
│   └── td_trainer.py          ← TD(λ) training loop, evaluation, SPRT, plateau detection
├── utils/
│   └── sprt.py                ← Sequential Probability Ratio Test (standalone)
├── tests/
│   ├── test_core.py           ← pytest tests (requires torch)
│   └── run_tests.py           ← standalone tests (no torch needed)
├── requirements.txt
├── pyproject.toml
└── setup_env.bat              ← Windows environment setup with CUDA
```

## Running
```
# From the BgRLEngine project directory:
python main.py --config configs/default.yaml --max-games 100    # smoke test
python main.py --config configs/default.yaml                     # train until plateau
```

---

## Architecture overview

### Two-level routing tree
```
Position
  ├── Checker play?
  │     ├── Race engine          ← mathematical, exhaustive
  │     └── General engine       ← TD-Gammon NN
  │
  └── Cube decision?
        ├── Race cube engine     ← mathematical, pip count based
        ├── Money play engine    ← single engine (fixed gammon value)
        └── Match play engine (pre-Crawford)
              ├── Gammon Seeking
              ├── Gammon Averse
              ├── Gammon Indifferent
              └── Gammon Balanced
```

### Phase 1: TD-Gammon style NN (IMPLEMENTED)
- Feedforward neural network trained via TD(λ), λ ≈ 0.7
- Self-play both sides; weights updated after each game
- Input: 303-feature board state vector (24 points × 6-unit thermometer+overflow encoding, bar, borne off, pip ratio, race flag, checker counts)
- Output: [P(win), P(win gammon), P(win backgammon), P(lose), P(lose gammon), P(lose backgammon)]
- Hidden layers: 2 layers, 256 units (configurable)
- Activation: ReLU (hidden), Sigmoid (output)
- Move selection training: ε-greedy; inference: greedy (best equity)
- Framework: PyTorch with CUDA (RTX 3060)
- Network size: ~145K parameters

### Phase 2: AlphaZero style NN + MCTS (deferred)

### Variants supported
- **Standard backgammon** — classic starting position
- **Nackgammon** — Nack Ballard variant starting position
- **Bg960** — random starting positions with constraints

All variants use identical rules (hitting, bar re-entry, bearing off). Only starting position differs.

---

## Bg960 setup constraints
- Symmetrical (opponent mirrors player)
- No checkers on bar or borne off at start
- At least 2 checkers on every occupied point (no blots)
- At least one occupied point per quadrant
- No mirror conflicts (point i and point 23-i never both occupied)
- Minimum pip count: 100
- Made-point distribution weighted toward 4–5 made points (configurable)

## Training strategy
- **Base model** trained on Bg960 (diverse starting positions)
- **Fine-tuned models** for standard backgammon and Nackgammon
- Warm-starting from Bg960 base weights

---

## Variant skill measurement framework

- **Level 0** — random legal move selection
- **Level N** — frozen checkpoint that beats Level N-1 at ~75% win rate
- **Adaptive training budget** — continue until progress plateaus
- **Termination** — stop when new level cannot reach threshold
- **Skill score** = levels reached + dominance margin (win rate of best vs second-best)

### SPRT promotion test (RESOLVED)
- Test: SPRT, one-sided
- H₀: p ≤ 0.70, H₁: p ≥ 0.76
- α = 0.05 (false promotion rate)
- β = 0.10 (missed promotion rate)
- Hard cap: 2,000 games → reject
- Metric: game win rate (not point rate)
- Gate threshold: rolling evaluation win rate ≥ 0.70 triggers SPRT

### Plateau detection (RESOLVED)
- Evaluation cadence: 100-game match every 5,000 training games
- Plateau budget: 500K × 1.5^(level) self-play games
- Staleness signal: <2pp improvement in rolling win rate over any 200K-game window → terminate early
- Three failed SPRTs within one level → halve remaining budget

### Gammon classification (RESOLVED)
- Rule-based from standard Match Equity Table (MET)
- Four categories: Seeking / Averse / Indifferent / Balanced
- Custom MET generation from trained models deferred

### Seam handling (RESOLVED)
- Hard boundary race detection: zero contact = race, else general engine
- New seams (containment, back game, prime-vs-prime) can be added later
- Router is a simple decision tree; adding a branch = define criterion + train specialist + insert node
- All sub-engines share the same 6-value output vector

---

## Key decisions
- Python / PyTorch for training; ONNX export for future C# inference
- Tabula rasa self-play — no supervised learning from XG data
- One model per variant, warm-started from Bg960 base
- Frozen checkpoints for level curriculum (not continuous snapshots)
- Adaptive training budget per level
- Race sub-engine is mathematical (no NN)
- Race cube engine is mathematical (no NN)
- Money play cube = single engine (gammon value is fixed)
- Match play cube classified by gammon dynamics via MET lookup
- Checker play sub-classification beyond Race vs General is deferred
- Standalone — no C# ecosystem integration in Phase 1
- State encoding: 303 features, no variant flags (rules are uniform)
- Bg960 base model with fine-tuning for specific starting positions

## Resolved questions
1. ✅ Win rate significance: SPRT with p₀=0.70, p₁=0.76, α=0.05, β=0.10, 2000-game cap
2. ✅ Plateau definition: 500K×1.5^level budget, staleness window, failed-SPRT acceleration
3. ✅ State representation: 303-feature uniform encoding, no variant flags, Bg960 generator
4. ✅ Gammon classification: rule-based from standard MET
5. ✅ Seam handling: hard boundary (zero contact), extensible router tree

## Known pitfalls
- **int16 for points array**: `BoardState.points` must be `np.int16`, not `int8`. Pip count calculations overflow int8 (max 127) since pip values reach 15×24=360. Same applies to `setup_generator._distribute_checkers`. All pip count methods must cast to `int()` before multiplication to avoid numpy scalar overflow.
- **Mirror conflicts in Bg960**: `setup_generator._select_points` must ensure point i and point (23-i) are never both selected. Otherwise `_mirror_setup` places opponent checkers on top of player checkers, corrupting the net count. The fix tracks a `blocked` set that excludes mirrors of every selected point.
- **Two-network evaluation**: `play_game()` takes an optional `opponent` parameter. Without it, both sides use the same network (self-play for training). `evaluate_against` and `_run_sprt` must pass the opponent network — otherwise they test self-play and always get ~50% win rate.
- **UTF-8 encoding on Windows**: All `open()` calls for config/data files must specify `encoding="utf-8"`. Windows defaults to cp1252 which chokes on YAML comments with special characters.
- **PyTorch API**: Use `torch.cuda.get_device_properties(0).total_memory` (not `total_mem`).
- **PowerShell and .bat files**: Use `cmd /c setup_env.bat` from PowerShell. `.\setup_env.bat` doesn't work.

## Design rationale (for future sessions)
- **SPRT over fixed-sample test**: SPRT naturally balances error rates against compute cost. Strong checkpoints promote in ~150 games; borderline ones spend ~1000 games in the indifference zone; clearly weak ones reject fast. Fixed-sample wastes games on obvious cases.
- **α=0.05, β=0.10**: False promotions (Type I) are worse than missed promotions (Type II). A false promotion weakens the entire curriculum ladder. A missed promotion just wastes some GPU time and self-corrects.
- **p₀=0.70, p₁=0.76**: 75% is the target; 70% is "clearly not ready," 76% is "clearly ready." The 6-point indifference zone is where spending extra games is justified.
- **Training strategy C (Bg960 base → fine-tune)**: Rules are identical across variants; only starting positions differ. Bg960 training produces a general-purpose player across diverse positions. Fine-tuning for standard/Nackgammon is cheap since mid-game positions converge. Strategy A (single model) was simpler but loses specialization. Strategy B (fully separate) wastes training.
- **No variant flags in encoding**: Since all variants share identical rules (hitting, bar re-entry, bearing off), the network doesn't need to know which variant is being played. The position itself is sufficient. This simplifies the encoding and means all variants share the same 303-feature input.
- **Rule-based gammon classification over learned**: The MET is solved mathematics. Asking the network to rediscover match equity arithmetic from scratch wastes training capacity on a problem with an exact analytical answer.
- **Hard boundary for race detection**: Zero contact = race, else general engine. The general engine handles near-race positions adequately. Soft blending adds complexity without clear benefit in Phase 1. New seams (containment, back game) can be added later as branches in the router tree.

## In progress
- Training validated: level 3 in 10K games at ~7 games/s across all equity configs
- Four equity weight configs validated: money, DMP, gammon-seeking, gammon-avoiding
- Long training run to find level ceiling (500K+ games)
- Verify Nackgammon starting position accuracy

## Deferred
- Config-specific promotion metrics (match win rate, equity error, gammon rate)
- Current 75% per-game threshold unreachable at higher levels due to dice variance
- Match-based metric: best-of-N win rate to filter luck
- Fix: SPRT failed counter not resetting after budget halving
- C# move generation integration (BgMoveGen)
- Multi-core parallelization of self-play

## Source files
All source is in the repo. Key files for reference:

- `engine/state.py` — BoardState class, feature encoding
- `engine/setup_generator.py` — Bg960 position generator
- `engine/network.py` — TDNetwork definition
- `engine/dice.py` — legal move generation
- `engine/game.py` — self-play simulation
- `training/td_trainer.py` — training loop orchestrator
- `utils/sprt.py` — SPRT implementation
- `configs/default.yaml` — all hyperparameters

## Shared rules
See `AGENTS.md` in the umbrella repo — applies to all sub-projects.
`https://raw.githubusercontent.com/halheinrich/backgammon/main/AGENTS.md`

## Session handoff
After committing:
1. `git rev-parse HEAD` in this subproject dir — note the short hash
2. Update commit hash in this doc and all raw URLs
3. Add URLs for any new files created
4. Update In progress / Deferred / Key decisions
5. Return to Backgammon Umbrella project — update umbrella instructions doc
