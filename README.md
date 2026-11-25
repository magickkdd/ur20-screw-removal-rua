# Guarded Action-Stream Control for UR20 Screw Removal

Research code for a MuJoCo UR20 benchmark that uses an LLM tool-calling policy to align a hex socket, engage an M6 screw, and extract it. This repository studies guarded action-stream fusion: grouping up to four consecutive free-space waypoints into one policy action while returning control at contact, low visual confidence, or an XY alignment phase change.

The implementation preserves the existing per-waypoint robot motion limits and task-specific contact, seating, and extraction controllers. The valid action-stream result in this repository is the paired archived-decision replay; the retained online recordings are diagnostic runs, not a valid fusion ablation.

## 最新真实 LLM 对照：RUA 完整适配

最新 `rua_full` 档案在 state 0 上两次成功，均为 5 次 LLM 调用：动作放大系数 3.0 时总墙钟时间 319.379 s，系数 5.0 时 297.062 s；两次都达到 20.7 mm 螺钉抬升并以 `screw_extracted` 停止。state 1 压力回合在 291.583 s 触发 `spiral_force_guard`，该记录保留用于说明力保护边界。

完整对照见 [`results/rua_full_comparison_20261126.json`](results/rua_full_comparison_20261126.json)，运行档案见 `outputs/ur20_luna_state00_20261126_024659/`、`outputs/ur20_luna_state00_20261126_025757/` 和 `outputs/ur20_luna_state01_20261126_025240/`。

## Results

### Live API diagnostic runs

The retained online runs used the same `state_id=0`, model (`gpt-6-luna`), contact controls, real-time pacing, and 21 mm extraction criterion, and both completed successfully. However, the run launched with `--fused-action-stream` issued zero `move_sequence` calls; all policy turns remained ordinary `move_to` calls until `spiral_search` and `turn_socket`. The 34 versus 36 turn difference is therefore an uncontrolled model trajectory difference, not evidence of fusion or acceleration. The files are retained for audit only.

The recorded videos retain the measured API waits by holding the latest camera frame during each response interval:

- Native: `outputs/live_native_fixed_20261124/rollout.mp4`
- Fused: `outputs/live_fused_fixed_20261124_run04/rollout.mp4`

Machine-readable values and the invalidation reason are in [`results/live_api_video_comparison_fixed.json`](results/live_api_video_comparison_fixed.json). The complete summaries remain alongside each video.

### Verified live pair (2026-11-25)

The replacement pair forces and records one real `move_sequence` call in fused mode. Both runs completed the same 21 mm extraction task and both videos include measured API waits and every internal spiral-search increment. Native used 34 policy turns and fused used 32. In this single live pair, fused took 549.57 s versus 545.71 s for native because its measured API response time was 3.82 s higher; this is a valid execution comparison but not evidence of a speedup under variable provider latency.

- Native video: `outputs/live_native_verified_20261125_run02/rollout.mp4`
- Fused video: `outputs/live_fused_verified_20261125/rollout.mp4`
- Verified comparison: [`results/live_api_video_comparison_verified_20261125.json`](results/live_api_video_comparison_verified_20261125.json)

### Archived paired replay results

| Metric | Native | Fused | Change |
|---|---:|---:|---:|
| Policy turns | 56 | 26 | -53.6% |
| Archived policy response latency | 1397.2 s | 561.9 s | -59.8% |
| Fast-forward simulation replay wall time | 158.4 s | 156.6 s | -1.1% |
| Latency-reconstructed task wall time | 1555.6 s | 718.5 s | -53.8% |
| Task successes | 1/2 | 1/2 | unchanged in these traces |

Both modes executed the same 53 robot actions, 2071 environment steps, and 135410 physics steps. Fast-forward replay keeps the MuJoCo timestep and force/contact updates unchanged while removing simulated-time sleeps. Internal spiral and extraction controllers render observations only periodically because the policy does not receive those intermediate frames. The reconstructed task time adds archived model-response latency to replay runtime; it estimates the effect of fewer model round trips, but is not a measured online LLM run. Both source traces share `state_id=0` and replay holds their policy decisions fixed, so this result does not establish population-level success-rate non-inferiority.

The simulator also writes spiral force diagnostics once per 25 physics steps while keeping force control, tactile state updates, and force guards at every physics step. In a repeated replay of the successful wristview trace, this reduced diagnostic output from 86.1 MB to 3.56 MB and replay wall time from 156.3 s to 94.2 s (native) and 154.5 s to 95.4 s (fused). The task remained successful in both modes with identical physics-step count and extraction distance. This engineering optimization is measured on one archived trace; it is separate from the policy-turn reduction above.

Per-trace values are in [`results/action_stream_fast_replay.json`](results/action_stream_fast_replay.json); the logging optimization comparison is in [`results/action_stream_trace_optimization.json`](results/action_stream_trace_optimization.json). The earlier real-time replay is retained in [`results/action_stream_replay.json`](results/action_stream_replay.json). The full study, including protocol, definitions, analysis, and limitations, is in [`docs/action_stream_ur20_study.md`](docs/action_stream_ur20_study.md). Broader historical benchmark records and their interpretation are in [`RESULTS.md`](RESULTS.md).

## Paper

The complete manuscript is available as [Markdown](docs/action_stream_ur20_study.md) and [Word](docs/action_stream_ur20_study.docx). The five publication figures are stored in `docs/figures/` as PNG, SVG, and PDF. The supplied figures are embedded by the manuscript; the plotting script can generate JSON-derived analysis figures in a separate output directory:

```bash
python -m pip install matplotlib
python scripts/evaluation/plot_paper_figures.py --output-dir outputs/paper_figures
pandoc docs/action_stream_ur20_study.md --resource-path=docs --standalone \
  --metadata author='CZX (magickkdd)' -o docs/action_stream_ur20_study.docx
```

## Method

The optional `move_sequence` policy tool accepts one to four absolute Cartesian waypoints. Each waypoint is mapped and executed separately through the existing robot adapter. The adapter ends the sequence when axial contact is detected, visual confidence falls below 0.7, or the XY alignment lock changes. Contact search, socket seating, and screw turning remain separate task controllers and are not batched into waypoint sequences.

The benchmark task uses a 10 mm hex socket and an M6 screw in a MuJoCo scene. Success in the paired replay requires verified engagement and at least 21 mm of measured axial extraction from the start of socket turning. The scene uses an ideal helical constraint and does not represent manufactured thread friction or physical UR20 performance.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Configure the policy API key and base URL in `.env` only when running a live policy episode. On headless Linux, set `MUJOCO_GL=egl`.

## Reproduce the paired replay

The replay uses the two committed source summaries and makes no policy API calls:

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/evaluation/replay_action_stream.py \
  --fast-simulation \
  --output outputs/action_stream_replay_run1 \
  --traces archive/rollouts/20261109_183017_agentview_primary/summary.json \
           archive/rollouts/20261109_191326_wristview_primary/summary.json
```

The output path must not already exist. The runner writes per-mode summaries, trajectory and force-trace artifacts, and an aggregate `results.json`. The tracked aggregate used in this repository is a compact copy of that result file.

## Record wall-clock comparison videos

The live runner records one native and one fused episode with the same initial state and real-time pacing. It measures each policy response interval in `video_waits.jsonl` and holds the last camera frame for that measured interval when encoding `rollout.mp4`; therefore the video timeline includes API waiting rather than compressing it away. The summary records `policy_response_wall_s`, `rollout_wall_s`, and `video_includes_api_waits`.

Run the two modes into separate output directories after configuring the selected provider profile in `.env`:

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --output outputs/video_native --execute-paid-rollouts

MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --output outputs/video_fused --fused-action-stream --execute-paid-rollouts
```

Compare `outputs/video_native/rollout.mp4` and `outputs/video_fused/rollout.mp4` using their `summary.json` files. These are live API runs and incur provider cost; archived replay latency is not substituted for a live response.

## Run the simulator

Run a local mock and policy contract check:

```bash
MUJOCO_GL=egl python scripts/rollout/run_mock.py
python scripts/rollout/check_policy.py
```

To run one live policy episode, the API call and rollout cost require explicit opt-in:

```bash
MUJOCO_GL=egl python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --execute-paid-rollouts --fused-action-stream
```

The rollout runner writes timestamped records under `outputs/`. The `--fused-action-stream` flag enables guarded waypoint grouping; native mode remains the default. Force feedback, compliant approach, spiral seating, and threaded extraction are separate opt-in controls.

Use `--fast-simulation` to run the MuJoCo episode without real-time pacing. The episode summary records `rollout_wall_s` and measured `policy_response_wall_s`, so a live native/fused comparison can separate actual model waiting from controller and simulator runtime.

## Repository map

- `src/ur20_screw/simulation/`: MuJoCo worker, robot adapter, and simulation timing.
- `src/ur20_screw/vision/`: image-based screw and tool estimates.
- `src/ur20_screw/control/`: action mapping and task-specific control contracts.
- `src/ur20_screw/policy/`: tool schema, policy client, and API budget accounting.
- `scripts/evaluation/`: archived-decision replay and evaluation utilities.
- `scripts/rollout/`: live and mocked rollout entry points.
- `assets/ur20_screw_station/`: UR20 scene, robot model, meshes, and textures.
- `archive/rollouts/`: source rollout records used by the paired replay.

## Documentation

- [Risk-adaptive action-stream paper](docs/risk_adaptive_action_stream_paper.md)
- [Action-stream fusion study](docs/action_stream_ur20_study.md)
- [System technical report](docs/technical_report.md)
- [Historical benchmark results](RESULTS.md)

## License and hardware scope

This repository does not declare a root-level license. Check the licenses of bundled assets, including `assets/ur20_screw_station/ur20/LICENSE`, before redistribution. The simulation force limits, clearances, and contact parameters are not validated operating limits for a physical UR20. No physical-robot evaluation is reported.

## Current RUA local-stream result

The latest verified real-LLM run applies guarded local action-stream execution to the UR20 task.
After visual XY alignment, the local controller performs bounded compliant approach steps and
returns control at axial contact; the LLM then calls `spiral_search` and `turn_socket`. The run
used `gpt-6-luna`, state 0, force feedback, compliant control, spiral search, threaded extraction,
low reasoning effort, and no viewer.

It completed successfully in 10 LLM calls and 560.493 seconds (9.342 minutes) of wall-clock time,
including LLM waiting. The maximum screw lift was 0.020705 m and the stop reason was
`screw_extracted`. Full records are in
[`archive/rollouts/20261125_rua_local_stream_success/`](archive/rollouts/20261125_rua_local_stream_success/).

- [Simulation replay video](archive/rollouts/20261125_rua_local_stream_success/rollout.mp4)
- [Full wall-clock video with phase annotations](archive/rollouts/20261125_rua_local_stream_success/rollout_realtime_annotated.mp4)
