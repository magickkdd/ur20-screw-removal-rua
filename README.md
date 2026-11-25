# UR20 风险自适应事件触发动作流

基于 MuJoCo 的 UR20 M6 螺钉拆解仿真。项目实现 LLM 高层决策、局部动作流、力反馈顺应控制、螺旋坐实和螺纹旋出，并对比逐步基线与风险自适应方法。

## 最新真实 LLM 对照

模型为 `openai/gpt-6-luna`，`state_id=0`，两组都启用力反馈、顺应控制、螺旋搜索和坐实后旋出，均成功完成拆解：

| 指标 | 基线 | 风险自适应 |
|---|---:|---:|
| LLM 调用次数 | 32 | 5 |
| 总墙钟时间 | 1002.860 s | 342.353 s |
| API 等待时间 | 692.468 s | 82.725 s |
| 动作流段数 | 0 | 3 |
| 内部动作路点 | 0 | 8 |
| 最大螺钉抬升量 | 20.705 mm | 20.709 mm |
| 停止原因 | `screw_extracted` | `screw_extracted` |

机器可读结果见 [`results/risk_adaptive_comparison_20261126.json`](results/risk_adaptive_comparison_20261126.json)。

## 安装

需要 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

无显示器 Linux 环境设置 `MUJOCO_GL=egl`，并在 `.env` 中填写 Luna API 配置。

## 运行

本地检查：

```bash
MUJOCO_GL=egl python scripts/rollout/run_mock.py
python scripts/rollout/check_policy.py
```

基线回合：

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --method baseline --no-mujoco-viewer --force-feedback \
  --compliant-control --spiral-search --twist-after-seating \
  --reasoning-effort low --execute-paid-rollouts
```

风险自适应回合：

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --method risk_adaptive --no-mujoco-viewer --force-feedback \
  --compliant-control --spiral-search --twist-after-seating \
  --reasoning-effort low \
  --prompt-cache-key ur20-risk-adaptive-state00-gpt6-luna \
  --prompt-cache-retention in-memory --execute-paid-rollouts
```

## 方法

风险自适应方法只在视觉置信度高、无轴向接触且处于自由空间时融合内部路点；接触、视觉置信度下降或 XY 锁定状态改变时立即中断动作流并恢复 LLM 决策。接近、螺旋坐实和螺纹旋出分别由局部控制器完成，不跨越阶段边界。

## 论文与记录

- [完整中文论文 Markdown](docs/risk_adaptive_action_stream_paper.md)
- [完整中文论文 DOCX](docs/risk_adaptive_action_stream_paper.docx)
- [对照实验 JSON](results/risk_adaptive_comparison_20261126.json)
- [基线回合记录](outputs/ur20_luna_state00_20261126_004036/)
- [风险自适应回合记录](outputs/ur20_luna_state00_20261126_010253/)

视频位于 GitHub Release：

- `baseline_rollout.mp4`
- `baseline_realtime_annotated.mp4`
- `risk_adaptive_rollout.mp4`
- `risk_adaptive_realtime_annotated.mp4`

## 目录

- `src/ur20_screw/`：策略、仿真、控制、视觉和评估代码。
- `scripts/`：回合、场景、视觉和评估脚本。
- `assets/ur20_screw_station/`：UR20 场景、网格和纹理。
- `tests/`：策略、传感器、视觉和视频产物测试。

当前实验只覆盖一个初始状态的一对真实 LLM 回合，结果是初步在线证据，不代表跨状态统计结论。
