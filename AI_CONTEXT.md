# GCBF-BPTT 单/多无人机智能飞行项目 - CLAUDE.md

## 核心科研目标 (Core Research Goal)
本项目旨在融合**基于几何的控制屏障函数 (GCBF+)** 和**通过时间反向传播 (BPTT) 的可微分物理**，开发出能够在复杂环境中执行任务的、兼具**极致安全**与**高效机动**的无人机自主控制策略。当前阶段的重点是**完美实现单机智能**。

## 技术栈 (Technology Stack)
* **核心框架:** PyTorch (最新稳定版)
* **科学计算:** NumPy
* **环境模拟:** 可能是基于PyTorch的自定义2D/3D环境
* **数据可视化:** Matplotlib, Seaborn
* **配置管理:** YAML (`.yaml` or `.yml` files)
* **包管理:** pip (通过 `requirements.txt`)
* **潜在需求:** 自定义 CUDA 核函数 (用于高效的可微分动力学计算)

## 项目架构 (Project Architecture)
项目遵循模块化的研究代码结构：
* `train.py`: 训练和评估流程的主入口脚本。
* `main.py`: (如果存在) 可能用于协调不同的运行模式（训练、可视化等）。
* `/configs`: 存放所有实验的`.yaml`配置文件。每个文件定义一组完整的超参数，用于确保实验的可复现性。
* `/src` (或 `/core`): 存放核心源代码。
    * `/envs`: 自定义无人机模拟环境。定义了状态空间、动作空间、奖励/损失函数、障碍物生成逻辑。
    * `/models`: 存放所有神经网络模型定义。
        * `policy_net.py`: 端到端策略网络。
        * `gcbf_module.py`: GCBF+安全模块的实现。
        * `prob_shield.py`: **“概率性安全护盾”**模型的实现。
    * `/losses`: 定义项目中使用的复杂损失函数，如目标达成、碰撞惩罚、平滑度(Jerk)惩罚、安全门控正则化项等。
    * `/utils`: 存放通用工具函数，如日志记录、数据处理、坐标变换、可视化辅助函数等。
* `/outputs`: 存放训练日志、模型权重、生成的图表和数据。
* `/scripts`: 存放用于特定任务的脚本，如数据后处理、批量可视化等。

## 代码风格与规范 (Coding Style & Conventions)
* **语言:** Python 3.8+ with Type Hinting。
* **格式化:** 遵循 PEP 8 规范。
* **命名约定:**
    * 类名: `PascalCase` (e.g., `DroneEnv`, `PolicyNetwork`)
    * 函数/变量名: `snake_case` (e.g., `compute_jerk_loss`)
    * 配置文件: `descriptive-name.yaml` (e.g., `rebalance-c.yaml`, `prob-shield-v1.yaml`)
* **PyTorch规范:**
    * 模型继承自 `torch.nn.Module`。
    * 在 `forward` 方法中定义计算图。
    * 清晰地使用 `.to(device)` 来管理CPU/GPU。
* **配置驱动开发:** **严禁**在代码中硬编码超参数。所有可调参数（学习率、损失权重、模型结构等）必须在`.yaml`配置文件中定义，并通过`train.py`加载。

## Git 工作流 (Git Workflow)
* **分支:** 为每个新实验或功能创建新分支 (e.g., `feature/probabilistic-shield`, `fix/nan-in-distance`)。
* **提交信息:** 使用清晰的提交信息，描述本次修改的内容。

## 重要提醒与核心逻辑 (Key Reminders & Core Logic)
1.  **安全是第一原则:** GCBF+的约束是硬性的。在“概率性安全护盾”模型中，当安全置信度 `alpha_safety` 趋近于0时，系统必须以 `GCBF安全动作` 为主导。
2.  **端到端可微是关键:** 整个计算图，从**策略网络输出 -> 动力学模拟一步 -> 新状态 -> 损失计算**，必须是完全可微分的。这是通过BPTT优化策略的根本。要特别注意任何可能中断梯度流的操作（如`.detach()`的误用）。
3.  **损失函数是灵魂:** 项目的性能直接取决于损失函数的设计和权重平衡。当前的关键权重包括：
    * `goal_weight`: 鼓励无人机飞向目标。
    * `jerk_loss_weight`: 惩罚加加速度，确保轨迹平滑。
    * `alpha_reg_weight`: (在安全门控模型中) 鼓励模型在安全区飞得更大胆。
4.  **当前对比实验:** 目前的核心任务是严格对比 **`Rebalance C` (Baseline)** 和 **`Probabilistic Shield` (Candidate)**。必须使用**完全相同的随机种子**来复现失败场景，并进行逐帧对比分析。
5.  **单机优先:** 在单机模型取得突破性进展前，暂停多机协同的开发。
6.  **诊断 NaN 问题:** `Avg.Min. Safe Distance` 为 `NaN` 很可能意味着在某些回合中没有探测到任何障碍物（距离为无穷大），或者发生了导致计算错误的碰撞。在代码中需要有安全检查（e.g., `torch.isnan`）和合理的默认值。