# RUA-UR20：面向 LLM 控制螺钉拆解的风险自适应双时间尺度闭环

**作者：** CZX（magickkdd）  
**实验日期：** 2026 年 11 月 26 日

## 摘要

视觉语言模型可以通过工具调用控制机器人，但在需要连续微动作的任务中，每个动作都等待一次模型响应，会使策略服务延迟成为主要任务开销。本文针对 MuJoCo UR20 M6 螺钉拆解任务，提出 RUA-UR20 风险自适应事件触发动作流。方法将 LLM 保留在任务阶段和异常处理层，将低风险自由空间运动交给局部控制器；局部动作只有在视觉置信度、横向误差和轴向力满足条件时才连续执行，并在接触、置信度下降或对齐状态改变时立即中断。完整适配档案同时使用有限历史、阶段自适应视角、提示缓存、有界重试和低推理强度，以减少上下文和模型等待。

在同一初始状态、同一 `gpt-6-luna` 模型、同一控制配置下进行的真实 LLM 配对实验中，基线和风险自适应方法均成功完成螺钉拆解。基线需要 32 次 LLM 调用和 1002.860 s 墙钟时间；风险自适应方法需要 5 次调用和 342.353 s。进一步的 RUA-UR20 完整适配档案在相同状态下两次成功，分别耗时 319.379 s 和 297.062 s，均以 `screw_extracted` 停止并达到约 20.7 mm 抬升。一个额外的 state-1 压力回合在 291.583 s 触发 `spiral_force_guard`，显示方法仍然服从力安全边界。当前数据支持单任务、单状态下的效率可行性结论，但尚不能声明跨状态统计优势。代码、原始记录和视频随仓库提供。

**关键词：** LLM 机器人控制；动作流；事件触发控制；风险门控；UR20；螺钉拆解；MuJoCo

## 1 引言

LLM 控制机器人通常采用“观察—模型决策—执行一个动作—再次观察”的闭环。该方式易于审计，但当任务包含大量自由空间微动作时，模型服务等待会被重复累积。对 UR20 螺钉拆解而言，横向对准和接近阶段需要连续运动，而接触、套筒坐实和螺纹旋出又具有明显的阶段边界。把所有动作交给 LLM 会产生大量往返；把整个任务交给一段预先生成的程序，又可能在接触或视觉失效后执行过期动作。

本文研究的问题是：**如何在不放松接触安全约束的情况下，只在低风险阶段减少 LLM 往返，并在风险事件出现时恢复模型决策？**

本文的核心方法是风险自适应事件触发动作流。它不是简单地把多个动作拼接起来，而是让风险状态同时决定动作段长度和 LLM 通信边界。方法包含三个要素：

1. 用视觉置信度、XY 误差、轴向力和任务阶段定义动作流是否可以继续；
2. 在低风险自由空间中执行多个有界内部路点，在接触或状态变化时立即停止；
3. 将图像数量、历史长度和模型推理开销随阶段变化，并保留接触和提取阶段的专用控制器。

本文贡献如下：

1. 提出面向螺钉拆解的风险门控双时间尺度控制架构；
2. 将动作流、局部顺应接近、力反馈门控、历史压缩、观测调度、提示缓存和阶段安全中断统一为一个可审计的 UR20 适配档案；
3. 实现 `baseline`、`risk_adaptive` 和 `rua_full` 的可复现实验开关，并报告真实 LLM 墙钟时间、API 等待、调用次数、动作流内部路点和任务终态；
4. 给出同一状态的真实在线配对和两次完整档案复现，同时报告一个跨状态力保护失败案例，明确效率结论和鲁棒性结论的证据边界。

## 2 基线方法

基线采用分层但逐动作的 LLM 控制流程。LLM 通过 `move_to`、`spiral_search` 和 `turn_socket` 工具操作 UR20。每次 `move_to` 只提交一个绝对笛卡尔目标，执行完成后重新发送完整观测并等待下一次模型响应。基线不使用局部动作流、历史观测压缩或自适应单视角输入；力反馈、顺应控制、螺旋搜索和坐实后旋出保持开启，以保证两组任务能力一致。

基线的任务流程为：

1. 从 `agentview` 和 `wristview` 获取图像及结构化状态；
2. 使用固定 ROI 和 HSV/连通域视觉估计螺钉横向位置；
3. 通过多次 `move_to` 完成 XY 对准和轴向下降；
4. 在检测到轴向接触后调用 `spiral_search`；
5. 在稳定坐实并通过视觉复核后调用 `turn_socket`。

基线的主要瓶颈是高层决策频率。自由空间对准和接近阶段的每一个小步都产生一次 API 往返，即使相邻动作之间没有新的任务级信息。

## 3 风险自适应事件触发动作流

### 3.1 风险状态

在每个内部动作点后更新状态：视觉置信度 (gamma_t)、横向误差 (e_t)、补偿轴向力 (f_t)、XY 锁定状态 (a_t) 和任务阶段 (s_t)。自由空间动作流只在以下条件同时满足时继续：

\[
\gamma_t \ge 0.7, \qquad e_t > 2\text{ mm}, \qquad |f_t| < 1\text{ N},
\]

且目标运动仍处于自由空间横向运动。动作流在以下任一事件发生时中断：

- 轴向力达到接触门限；
- 视觉置信度低于 0.7；
- XY 误差进入锁定范围或锁定状态改变；
- 任务阶段从自由空间接近转入接触处理；
- 仿真器返回终止或安全错误。

这使动作段长度由状态风险决定，而不是固定地跳过观测。

### 3.2 双时间尺度执行

LLM 负责低频任务级决策。局部控制器负责高频内部点，单个模型动作最多追加三个受限横向内部点；每个内部点仍经过机器人适配器和仿真器，保留笛卡尔位移限制、力反馈和视觉门控。接近阶段可以由 `approach_until_contact` 执行局部顺应下降；接触后动作流被挂起，只允许 `spiral_search`；稳定坐实后由 `turn_socket` 完成整个螺纹提取。

因此，方法的控制边界为：

```text
LLM: 选择阶段和目标
  -> 低风险自由空间: 局部动作流
  -> 接触事件: 立即交回 LLM
  -> 螺旋坐实: 阻塞式力控搜索
  -> 稳定坐实: 阻塞式螺纹旋出
```

### 3.3 RUA-UR20 适配档案

RUA 原始实现面向 Python 程序单元、结构化事实和可预生成分支；UR20 策略使用 Responses API 函数调用，不能直接执行 RUA 的 Python 单元解释器。因此本文把可迁移机制映射到同一个安全接口，而不是引入不适用于 UR20 的程序运行时。`rua_full` 档案包含以下组件：

| RUA 机制 | UR20 中的实现 | 可观测记录 |
|---|---|---|
| 动作流提交 | `move_to` 后追加最多三个受限内部横向路点 | `action_stream_segments`、`action_stream_points` |
| 风险中断 | 接触、低置信度、XY 锁定变化或终止立即停止 | `risk_events`、阶段时间线 |
| 局部执行交接 | `approach_until_contact`、螺旋搜索和旋出由局部控制器阻塞执行 | 工具调用与物理步数 |
| 观测窗口 | 保留最近两轮图像，稳定阶段优先 wristview | 活动图像数、视角记录 |
| 历史压缩 | `rua_full` 保留 3 轮观测，其余历史替换为占位文本 | `history_compactions` |
| 前缀缓存 | 固定提示和工具 schema 使用 `prompt_cache_key` | API usage 中的 cached tokens |
| 有界重试 | 对允许的瞬时响应错误进行有限重试 | API usage 事件 |

`rua_full` 不把接触、坐实或旋出纳入动作流。这样保留了 RUA “稳定区间减少高层往返、事件发生时恢复高层控制”的核心思想，同时保持 UR20 的力控安全边界。该映射是任务适配，不声称复现 RUA 在其他 benchmark 上的全部运行时语义。

### 3.4 观测与上下文调度

风险自适应模式使用有限观测历史。满足高置信度、低误差、无轴向力时优先发送 wristview；其他阶段发送两路视角。超过窗口的历史图像和观测被替换为占位文本。固定 system prompt 和工具 schema 使用 `prompt_cache_key` 与 `prompt_cache_retention`，模型使用低 reasoning effort。上述机制不改变任务状态机，只减少稳定阶段的输入规模和推理负担。

### 3.5 任务安全约束

`spiral_search` 仍要求观测到至少 1 N 的轴向接触；`turn_socket` 仍要求触觉坐实和视觉确认。所有横向动作继续经过现有 XY 对齐锁定和单步位移限制。动作流不得跨越接触、坐实或旋出阶段。

## 4 实验设置

### 4.1 环境与任务

实验使用 MuJoCo UR20 场景、固定安装的六角套筒和 M6 螺钉。螺钉由理想螺旋约束耦合旋转和轴向位移。成功条件为：检测到有效套筒接合，稳定坐实，并通过 `turn_socket` 达到至少 21 mm 的轴向提取量。

### 4.2 配置

所有在线回合使用相同模型和控制器能力；方法档案决定高层通信和动作流配置：

| 项目 | 设置 |
|---|---|
| 模型 | `openai/gpt-6-luna` |
| profile | `luna` |
| 初始状态 | state 0 配对与复现；另有 state 1 压力回合 |
| 力反馈 | 开启 |
| 顺应控制 | 开启 |
| 螺旋搜索 | 开启 |
| 坐实后旋出 | 开启 |
| reasoning effort | `low` |
| viewer | 关闭 |
| 最大调用数 | 50 |
| 最大仿真步数 | 7500 |
| 动作放大 | 3.0；完整档案另测 5.0 |

主要配对的唯一方法变量是 `--method baseline` 与 `--method risk_adaptive`。基线不启用动作流、历史压缩或自适应观测；风险自适应方法启用这些组件。新增 `--method rua_full` 作为 RUA-UR20 完整适配档案，使用更短的 3 轮历史窗口，并显式记录所有启用组件。由于模型服务延迟随时间变化，`rua_full` 的两次 state-0 回合用于复现和工作点比较，不替代严格的多状态消融。

### 4.3 测量

每个回合写入 `summary.json`、`wall_clock.json`、`actions.jsonl`、`transcript.jsonl`、`api/api_usage.jsonl`、阶段时间线和两个视频。总墙钟时间从仿真创建后开始，包含 LLM 等待；API 等待由每个响应事件的 `latency_s` 求和。任务终态由环境成功标志、停止原因和最大螺钉抬升量共同判断。

## 5 实验结果

### 5.1 真实在线配对结果

| 指标 | 基线 | 风险自适应 | 变化 |
|---|---:|---:|---:|
| 任务状态 | 成功 | 成功 | 不变 |
| LLM 调用次数 | 32 | 5 | -84.4% |
| 动作流段数 | 0 | 3 | — |
| 动作流内部路点 | 0 | 8 | — |
| 总墙钟时间 | 1002.860 s | 342.353 s | -65.9% |
| API 响应等待 | 692.468 s | 82.725 s | -88.0% |
| 输入 token | 988,490 | 51,328 | -94.8% |
| 输出 token | 3,210 | 481 | -85.0% |
| 缓存输入 token | 231,808 | 24,832 | 口径不同，非主要结论 |
| 估算费用 | $0.07959 | $0.00314 | -96.1% |
| 最大螺钉抬升量 | 20.705 mm | 20.709 mm | 近似不变 |
| 螺旋内部增量 | 2 | 1 | — |
| 停止原因 | `screw_extracted` | `screw_extracted` | 不变 |

风险自适应回合在 3 个低风险横向动作段中执行了 8 个内部路点；接触后没有继续执行自由空间动作流，而是交回 LLM 调用 `spiral_search` 和 `turn_socket`。两组均达到任务成功条件，改进组没有以减少旋出过程或降低物理仿真步数换取成功。

### 5.2 结果解释

墙钟时间下降主要来自两部分：LLM 调用从 32 次减少到 5 次，以及每个请求上下文从完整历史变为有限窗口。API 等待下降幅度大于总墙钟时间下降幅度，说明模型服务等待是本次基线的主要瓶颈。风险自适应组的物理步数为 48,525，基线为 49,397；两者数量同一量级，不能把主要收益解释为跳过物理控制。

本实验同时验证了风险门控的行为：自由空间动作流在视觉和无接触条件下运行；接触后动作流停止，后续由螺旋搜索和坐实后旋出控制器执行。没有出现因动作流跨越接触阶段而导致的失败。

### 5.3 RUA-UR20 完整适配回合

为检验组合档案在不同运行条件下的表现，进一步运行了两个 state-0 完整档案回合和一个 state-1 压力回合。结果如下：

| 回合 | 方法 | 状态 | 放大系数 | 状态 | 总墙钟时间 | API 等待 | 调用 | 动作流路点 | 终态 |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `rua_full` | 0 | 3.0 | 成功 | 319.379 s | 64.424 s | 5 | 8 | `screw_extracted` |
| 2 | `rua_full` | 0 | 5.0 | 成功 | 297.062 s | 44.486 s | 5 | 8 | `screw_extracted` |
| 3 | `rua_full` | 1 | 3.0 | 失败 | 291.583 s | 281.554 s | 25 | 1 | `spiral_force_guard` |

两个 state-0 回合均保持约 20.7 mm 的最大螺钉抬升量。相对 state-0 基线，3.0 和 5.0 放大系数的完整档案分别减少 68.1% 和 70.4% 的总墙钟时间；相对此前 342.353 s 的风险自适应回合，分别减少 6.7% 和 13.2%。这两个差异应解释为当前服务时段和完整档案配置下的工作点结果，而不是放大系数或单个 RUA 组件的独立因果效应。

state-1 回合在进入有效套筒接合前触发了 `spiral_force_guard`，补偿力约为 66.16 N，螺钉抬升量为 0。该失败没有被动作流绕过，说明力保护仍然优先于完成率；同时它表明 state-0 的成功结果不能直接推广到全部初始状态。完整机器可读记录见 `results/rua_full_comparison_20261126.json`。

## 6 证据归因与后续验证计划

当前真实在线实验包含一组严格的 state-0 基线配对和两次完整档案复现，但尚未完成各组件独立消融。因而论文可以支持“组合方法减少了本任务的高层往返”，不能把全部收益归因于动作流本身。为关闭方法归因，后续实验应保持同一 `state_id` 集合和同一模型，增加以下变体：

1. 基线：逐步 `move_to`、完整历史、双视角；
2. 仅动作流：关闭历史压缩和缓存；
3. 动作流 + 风险门控：加入接触/置信度/对齐中断；
4. 历史压缩、图像窗口和 prompt cache：逐项加入通信优化；
5. 完整方法：再加入所有 RUA-UR20 组件。

每个变体至少运行多个初始状态并重复，报告成功率、墙钟时间中位数和 P90、LLM 等待占比、调用次数、输入 token、缓存命中、动作流中断次数和失败类型。运行顺序应交错安排，以减少服务时段造成的偏差。只有在多状态重复实验中同时保持成功率不下降，才能将本次 state-0 结果升级为总体效率结论。

## 7 相关工作与定位

本文受到 Code as Policies、HexaShell 和 RUA 动作流思想启发。与单纯的工具调用或固定动作合并不同，本文将动作段长度与接触风险绑定：低风险阶段允许局部连续动作，高风险事件触发模型重新决策。RUA 的程序流式提交、有限上下文和分支预生成提供了通信优化的设计参照；本文只迁移其中能由 UR20 函数调用和物理状态检查实现的部分，不声称复现其 Python 程序 IR 或跨基准性能。本文聚焦 UR20 螺钉拆解中的风险门控通信和阶段控制，不声称提出通用机器人 API。

## 8 局限性

1. 严格的 baseline/risk-adaptive 配对只覆盖一个初始状态、每种方法一个回合；新增完整档案虽有两次 state-0 成功和一次 state-1 力保护失败，仍不能代表跨状态成功率。
2. 完整档案同时启用了动作流、历史压缩、图像窗口和 prompt cache，当前结果不能单独归因于某一组件。
3. API 响应延迟随服务时段变化，单次墙钟差异需要多次交错重复才能稳定估计。
4. state-1 回合触发了 `spiral_force_guard`，说明当前完整档案的跨状态鲁棒性尚未闭合。
5. 场景采用理想螺旋约束，不模拟真实螺纹摩擦、材料变形或实体 UR20 的控制延迟。
6. 当前 reasoning effort 固定为 `low`；真正按风险升级推理强度的策略尚未完成。

## 9 结论

本文实现了 RUA-UR20 风险自适应事件触发动作流，将 LLM 决策限制在阶段变化和高风险事件附近，并由局部控制器完成低风险自由空间运动。在 state-0 真实在线实验中，风险自适应配对将调用次数从 32 降至 5，将总墙钟时间从 1002.860 s 降至 342.353 s；完整适配档案的两次复现进一步达到 319.379 s 和 297.062 s，同时保持螺钉提取成功。state-1 压力回合触发力保护，说明安全约束仍然有效但跨状态鲁棒性尚未充分验证。当前证据支持“风险门控动作流能够显著减少本任务中的模型往返”这一初步结论；多状态重复、交错延迟控制和逐组件消融仍是形成统计性结论所必需的后续工作。

## 数据与代码

- 基线回合：`outputs/ur20_luna_state00_20261126_004036/`
- 风险自适应回合：`outputs/ur20_luna_state00_20261126_010253/`
- 完整档案回合：`outputs/ur20_luna_state00_20261126_024659/`、`outputs/ur20_luna_state00_20261126_025757/`
- state-1 力保护压力回合：`outputs/ur20_luna_state01_20261126_025240/`
- 完整档案汇总：`results/rua_full_comparison_20261126.json`
- 实验代码：`scripts/rollout/run_episode.py`、`src/ur20_screw/policy/client.py`
- 原有动作流研究：[`docs/action_stream_ur20_study.md`](action_stream_ur20_study.md)

复现实验命令：

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --method baseline --no-mujoco-viewer --force-feedback \
  --compliant-control --spiral-search --twist-after-seating \
  --reasoning-effort low --execute-paid-rollouts

MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --method risk_adaptive --no-mujoco-viewer --force-feedback \
  --compliant-control --spiral-search --twist-after-seating \
  --reasoning-effort low \
  --prompt-cache-key ur20-risk-adaptive-state00-gpt6-luna \
  --prompt-cache-retention in-memory --execute-paid-rollouts

MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --method rua_full --no-mujoco-viewer --force-feedback \
  --compliant-control --spiral-search --twist-after-seating \
  --reasoning-effort low --action-amplification 3.0 \
  --prompt-cache-key ur20-rua-full-state00-gpt6-luna-a3 \
  --prompt-cache-retention in-memory --execute-paid-rollouts
```

## 参考文献

1. Liang J, Huang W, Xia F, et al. Code as Policies: Language Model Programs for Embodied Control. ICRA, 2023.
2. Gao H, Zeng J, Zhou A, et al. HexaShell: A Universal Harness for Robot Coding Agents in Simulation and the Real World. arXiv:2610.23479, 2026.
3. Todorov E, Erez T, Tassa Y. MuJoCo: A physics engine for model-based control. IROS, 2012.
