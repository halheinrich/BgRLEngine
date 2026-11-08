# BgRLEngine — Subproject Instructions

> Collaboration contract: [`../AGENTS.md`](../AGENTS.md)
> Umbrella status & dependency graph: [`../INSTRUCTIONS.md`](../INSTRUCTIONS.md)
> Mission & principles: [`../VISION.md`](../VISION.md)

## What it is for

BgRLEngine trains backgammon-playing neural networks from scratch by TD-Gammon-style self-play and exports them as ONNX models for the C# side of the tree. Its output is files, not a library: training checkpoints, the ONNX exports BgInference loads, and the committed parity fixtures that BgInference's and BgTournament's tests read in place. No other repository imports its Python. Phase 1 is the General engine for checker play; the routing tree under Architecture shows what else is planned.

**Stack:** Python 3.11, PyTorch (the local venv has 2.5.1+cu121), ONNX and ONNX Runtime, on Windows; a .NET SDK to publish the BgMoveGen DLL. `BgRLEngine.slnx` wraps `BgRLEngine.pyproj` for the Visual Studio 2026 Python workload and nothing else: there is no `dotnet build` here.

**Depends on:** BgMoveGen, for move generation and all variant starting positions, consumed at runtime as a native DLL via ctypes (`engine/movegen.py`). No managed interop, no project reference.

## Running it

Everything runs from the package directory, `BgRLEngine/BgRLEngine/`: it is the import root.

**Set up.** The venv is `env/` in the package directory:

```
python -m venv env
.\env\Scripts\Activate.ps1          # cmd: env\Scripts\activate.bat
```

The install lines are in `pyproject.toml`'s header, the canonical dependency record; the CUDA torch build needs its own index URL.

The BgMoveGen DLL is gitignored, so publish it after cloning. From the BgMoveGen repo root, with `<repo>` the umbrella checkout:

```
dotnet publish BgMoveGen/BgMoveGen.csproj -c Release -r win-x64 -o "<repo>/BgRLEngine/BgRLEngine/native"
```

Publish the project, never the solution; [BgMoveGen's "Publishing the native DLL"](../BgMoveGen/INSTRUCTIONS.md#publishing-the-native-dll) owns that rule and its reasons. `load_movegen()` reads `native/BgMoveGen.dll` unless `dll_path` or the `movegen.dll_path` config key says otherwise, and refuses a DLL of any version but `REQUIRED_MOVEGEN_VERSION`.

**Test.** No CI runs this suite; run it by hand, naming the interpreter:

```
./env/Scripts/python.exe -m pytest tests
```

`.gitignore`'s `env/` matches at any depth, so a stray venv elsewhere in the checkout does not show in `git status`, and one without onnx or onnxruntime fails at collection. `test_core.py` needs the DLL and errors without it. The parity gate needs only the committed fixtures and never skips. `test_checkpoint_round_trip` reads the gitignored `output/checkpoint_level4.pt` and skips loudly without it, so a fresh clone runs one test fewer than a machine with training output.

**Train, export, regenerate.**

```
python main.py [--config configs/<name>.yaml] [--max-games N] [--output-dir DIR]
python export_onnx.py CHECKPOINT [--output PATH] [--model-role ROLE]
python -m parity.generate_vectors        # regenerate the committed parity fixtures
```

The standalone tools under Layout carry their usage in their headers.

## Layout

One Python package, `BgRLEngine/`.

**`engine/`** — the engine proper. Five areas:

- **Position and encoding** — `state.py`: `BoardState`, the 303-feature encoding (`encode_board`, the vectorized `encode_board_batch`) and `ENCODING_VERSION`, the version every export and checkpoint is stamped with.
- **Move generation** — `movegen.py`: the ctypes wrapper over the BgMoveGen native DLL (successor states, every variant's starting position) and its `REQUIRED_MOVEGEN_VERSION` check; `halheinrich/backgammon#284` tracks the wrapper's untested defensive branches. `dice.py`: `roll_dice`, which self-play uses, beside the older pure-Python play generator (`generate_plays` and its helpers) that training no longer calls — the tests, `bench_encode.py` and `compare_configs.py` still import it.
- **The network** — `network.py`: `TDNetwork` (its architecture self-description and `from_state_dict`) and the equity computations.
- **Self-play** — `game.py`: one game's simulation, for training (self-play) or evaluation (against an opponent network), and `GameResult`, whose predicates (`is_win`, `is_loss`, `is_decisive`) are how an outcome is read; a game the move cap stops is `TRUNCATED`.
- **Persistence and export** — `checkpoint.py`, the training checkpoint format and its only reader and writer; `export.py`, ONNX export and the embedded `bgrl.*` metadata contract.

**`training/`** — `td_trainer.py`: the TD(λ) loop — ε-greedy self-play, evaluation against frozen level opponents, SPRT promotion, plateau detection, and checkpoint saves.

**`utils/`** — `sprt.py`: the standalone SPRT behind promotion.

**`configs/`** — complete, standalone training configurations, one per training target: DMP per variant (`dmp`, `nackgammon_dmp`, `bg960_dmp`), `money`, `gammon_seeking`, `gammon_avoiding`. `default.yaml` is `main.py`'s default `--config` and the file the tests load.

**`parity/`** — the committed cross-language parity fixtures (see Architecture): `generate_vectors.py`, their deterministic generator; `model.onnx`, a tiny parity model, *not* a trained one; `vectors.json`, the golden board→features→output cases.

**`tests/`** — the pytest suite is `test_core.py` (engine, training and checkpoint contracts; needs torch) and `test_parity.py` (PyTorch↔ONNX Runtime parity and the committed-fixture gate). Beside it, standalone scripts that are not collected: `run_tests.py`, a runner that needs neither pytest nor torch, though it loads the DLL; `bench_encode.py`, `encode_board_batch` correctness and speedup; `verify_bg960.py`, the Bg960 starting-position constraint checker.

**Scripts** — `main.py`, the training CLI; `export_onnx.py`, trainer checkpoint → ONNX; `profile_training.py`, the self-play vs. TD-update timing split; `compare_configs.py`, which play two equity-weight configs disagree on. That last one does not import: `engine.game._apply_play` is gone, so repairing it is a rewrite against the successor-state API, a repair-or-delete decision booked as `halheinrich/backgammon#168`. It stays out of `<Compile>` until that lands, since listing a module that cannot be imported would be worse than the omission; if repaired, the private `engine.dice._apply_move` it reaches for is promoted or avoided.

**Local, gitignored** — `native/` (the published BgMoveGen DLL), `output/` (training checkpoints and artifacts), `env/` (the venv).

## Architecture

**TD-Gammon-style neural network, self-play from scratch.** A compact feedforward network — 2 hidden layers × 256 units, ReLU hidden, sigmoid output — trained by TD(λ) in PyTorch. ε-greedy move selection during training, greedy at inference. One network plays both sides; weights update after each game.

**Training runs on CPU at this network size.** Self-play and evaluation always do, and `Trainer` also moves backprop to CPU for any network under 1M parameters (`td_trainer.py`, `Trainer.__init__`), which covers the 2×256 network, so a GPU changes nothing until the network grows. The threshold rests on the judgment that kernel launch and transfer overhead dominate at this size (415b6a5, 2026-03-21); that commit records a throughput but no CPU/GPU comparison, and `profile_training.py` times CPU only.

**Input encoding.** 303 features, board only; `engine/state.py`'s module docstring gives the layout. No variant flags: Standard, Nackgammon and Bg960 share identical rules (hitting, bar re-entry, bearing off) and differ only in starting position, so the position itself is sufficient input, and that is what lets weights transfer across variants. The cube and match features (`encode_full`) exist for the cube engines; no network consumes them yet.

**Output vector.** 6 values: `[P(win), P(win gammon), P(win backgammon), P(lose), P(lose gammon), P(lose backgammon)]`. All sub-engines share this shape so the routing tree can compose them uniformly. The six are exclusive outcomes, not cumulative ones: the training target is one-hot (`result_to_target`), so `P(win)` means a plain win. They are independent sigmoids, not a softmax, so nothing forces them to sum to 1.

**Two-level routing tree.**

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

The diagram is the design target; only the General engine exists. Phase 1 scope is the General engine for checker play. The race/contact seam is a hard boundary — zero contact = race, else general engine. New seams (containment, back game, prime-vs-prime) can be added as additional router branches, each a specialist NN with the same 6-value output.

**Gammon classification is rule-based** in that target: a MET lookup produces Seeking / Averse / Indifferent / Balanced, because the MET is solved mathematics and asking the NN to rediscover match-equity arithmetic wastes training capacity. Nothing implements the lookup yet; `compute_match_equity` takes the gammon values as arguments.

**Variants.** Standard, Nackgammon, Bg960. BgMoveGen is the sole source of truth for every starting position — the Python side never constructs its own. BgDataTypes_Lib's `BoardPosition.Bg960` owns the Bg960 constraints; `tests/verify_bg960.py` checks them from this side.

**Training strategy, planned.** Train a Bg960 base model (diverse starts → general-purpose player), then fine-tune it for Standard and Nackgammon. Nothing does this yet: `Trainer` always starts from a fresh network, and no path loads a checkpoint into it. That fine-tuning will be cheap because mid-game positions converge is the design's assumption (45619ec, 2026-03-23), not a result.

**Promotion.** Levels are frozen checkpoints: Level N+1 is the network that beat Level N. Evaluation plays a short match against the current level's frozen opponent; when that match's win rate reaches the gate, the SPRT runs and decides promotion. `rolling_win_rate` in the code is that latest match's rate, not an average. Level 0→1 is the gate alone, with no SPRT, against an untrained network. The metric is game win rate over decisive games, not point rate: a game the move cap truncates counts in neither the wins nor the games, in evaluation or in the SPRT, though it does count toward the SPRT's hard cap, which is on games attempted. An evaluation match with no decisive game raises and ends the run rather than report a rate. `p0` and `p1` bracket a 75% target, 70% being "clearly not ready" and 76% "clearly ready" (45619ec, 2026-03-23). α < β because false promotions corrupt the curriculum ladder, while missed promotions just waste training games and self-correct. A test that reaches the hard cap rejects.

**Plateau detection.** Each level has a self-play budget that grows geometrically with the level. A run ends when the budget runs out or training goes stale: stale means the latest evaluation's win rate improved by too little over the evaluation one staleness window of games earlier in the same level. Repeated failed SPRTs within a level halve what remains of its budget.

**The numbers live in the configs.** Evaluation cadence, match size, every SPRT parameter, the budget and the staleness window are the `evaluation`, `sprt` and `plateau` blocks of each `configs/*.yaml`. The fallback defaults in `Trainer` and `sprt_test` fill only a key a config omits, or a whole config when `main.py` finds no file at `--config`, and they are not kept in step: `cadence_games` already differs.

**ONNX export & the cross-language contract.** Trained checkpoints become consumable outside Python via `engine/export.py`, whose module docstring owns the graph and metadata contract. The contract travels inside the `.onnx` file's `metadata_props` — no sidecar to drift; ONNX Runtime exposes it as `ModelMetadata.CustomMetadataMap`. One network per ONNX file; the portfolio-of-experts export shape (fused graph vs. router + experts) is deliberately deferred until more than the General engine exists, and `bgrl.model_role` keeps exports self-describing for that future.

What a reader outside Python needs:

| Contract | Owner | Held by |
| --- | --- | --- |
| Board→feature encoding version | `ENCODING_VERSION`, `engine/state.py` | Stamped as `bgrl.encoding_version`; the consumer holds its own required version and fails fast, mirroring `REQUIRED_MOVEGEN_VERSION` |
| Graph: input `features` float32 `[batch, input_size]` (dynamic batch), output `probabilities` float32 `[batch, 6]`, opset 17 | `INPUT_NAME`, `OUTPUT_NAME`, `ONNX_OPSET`, `engine/export.py` | `test_parity.py` |
| Model metadata: `bgrl.*` structural keys, plus checkpoint provenance (source file, sha256, games, level) on checkpoint exports | `engine/export.py` | `test_parity.py`; BgInference's handshake |
| Native interop version | `REQUIRED_MOVEGEN_VERSION`, `engine/movegen.py`, equal to BgMoveGen's `Interop.Version` | `load_movegen()` refuses any other version |
| Checkpoint self-description | `CHECKPOINT_ARCHITECTURE_KEY`, `CHECKPOINT_ENCODING_VERSION_KEY`, `engine/checkpoint.py` | `load_checkpoint` |
| Encoding and inference parity | `parity/model.onnx`, `parity/vectors.json` | `test_parity.py` here; BgInference's parity gate there |

**The checkpoint contract has one owner: `engine/checkpoint.py`,** whose module docstring states it in full. The `.pt` file's shape is that module's alone: `save_checkpoint` writes it, `load_checkpoint` is the one path by which a checkpoint becomes a network, and no caller builds or indexes the dict. Every save stamps the architecture and the encoding version beside the weights, the same no-sidecar principle the `bgrl.*` metadata applies to exported models, one level up the pipeline (`halheinrich/backgammon#167`, `halheinrich/backgammon#169`). A checkpoint from another encoding version refuses to load, because weights trained against one encoding evaluate silently wrong under another of the same input size. One whose embedded architecture disagrees with its weight shapes refuses too, since the file is corrupt or hand-edited and neither side can be trusted. Two older shapes stay readable on purpose: an unstamped checkpoint is read as encoding 1 with a warning, and one without the architecture key infers it from the weight shapes, a permanent fallback because `output/` is gitignored and such files cannot be regenerated.

**Parity fixtures are the encoding's cross-language SSOT mitigation.** The 303-feature encoding must be reimplemented in C# and cannot be single-sourced, so `parity/` commits the executable contract: a tiny deterministic parity model (`model.onnx`, 303→16→16→6, seeded-NumPy weights, no timestamp — not a trained model) plus `vectors.json`, golden board→features→output triples sha256-paired to the model. Regeneration was byte-identical when 0a58949 checked it (2026-07-02); `vectors.json`'s `generator` block names the toolchain behind the committed pair. The consumer gate reads its tolerances from the fixture (encoding pin bit-exact, inference pin abs 1e-5). Committed here rather than the umbrella's gitignored `TestData/` so a fresh checkout fails loud, never no-ops.

## Pitfalls

- **Two-network evaluation.** `play_game()` takes an optional `opponent` parameter. Without it, both sides use the same network (self-play for training). `evaluate_against()` and `_run_sprt()` **must** pass the opponent network — otherwise they silently test self-play and report ~50% whatever the network's strength.
- **A game at the move cap is truncated, not won.** `play_game` stops a game after `max_moves` turns (1000) and returns `GameResult.TRUNCATED`; until `halheinrich/backgammon#534` it returned `WIN` for the initial player, which trained the network toward an outcome nobody reached and scored the game as a win for whichever side started. A truncated game teaches nothing (`Trainer.train` skips its update; `td_lambda_update` and `result_to_target` refuse it) and decides nothing (evaluation and the SPRT leave it out), and it is counted: `games_truncated` in the progress line, with the evaluation and SPRT counts in the final summary. A count that is not near zero, or that rises, says games are failing to finish, which is a defect to find, not a statistic to tune around. The frequency is unmeasured until the first run that reports it. Read an outcome through the `GameResult` predicates, never its sign: `GameResult` is a plain `Enum`, so comparing a result with a number raises, and `TRUNCATED`'s value is `None`, so a sign test on `.value` raises for exactly the game it would misread.
- **The BgMoveGen DLL can be stale at the right version.** It is published and gitignored, so it is invisible to csproj audits and to git history. After any BgMoveGen behaviour change, republish it (see Running it) and run pytest: `REQUIRED_MOVEGEN_VERSION` guards the interop contract, not a bug fixed within it. A breaking interop change bumps both sides together; [BgMoveGen's publishing section](../BgMoveGen/INSTRUCTIONS.md#publishing-the-native-dll) owns that rule.
- **UTF-8 on Windows.** Every `open()` for config/data files must pass `encoding="utf-8"`. Default `cp1252` chokes on YAML comments containing non-ASCII. **Console output is the same trap:** a standalone script that prints a non-ASCII status marker dies on its own success message under the default `cp1252` stdout, after the work it was measuring already succeeded. `sys.stdout.reconfigure(encoding="utf-8")` at the top of such scripts.
- **PyTorch CUDA memory query.** The attribute is `total_memory`, not `total_mem` — use `torch.cuda.get_device_properties(0).total_memory`.
- **No variant flags in state encoding.** Tempting to add a one-hot "this is Nackgammon" feature when debugging variant-specific regressions; don't. Input encoding, under Architecture, says why.
- **`.gitignore` excludes `*.txt` — never add a `requirements.txt`.** It would be silently invisible to version control. `pyproject.toml` is the canonical dependency record; if a new dep is installed, record it there.
- **Encoding changes require a version bump + fixture regeneration.** Any change to the feature layout, sizes, or arithmetic in `engine/state.py` must bump `ENCODING_VERSION` and regenerate the parity fixtures (`python -m parity.generate_vectors`) in the same commit — `model.onnx` and `vectors.json` are sha256-paired and must always move together. A bump also strands every earlier training checkpoint: `load_checkpoint` refuses it and nothing migrates.
- **Never commit an exported trained model.** Only `parity/model.onnx` belongs in git; trained exports are deployment artifacts identified by their embedded `bgrl.checkpoint_sha256` and regenerated locally with `export_onnx.py`.
