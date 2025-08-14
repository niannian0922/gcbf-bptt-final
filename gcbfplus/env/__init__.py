from typing import Optional

# 暴露重构后的环境模块
from .base_env import BaseEnv, EnvState, StepResult
from .multi_agent_env import MultiAgentEnv, MultiAgentState
from .double_integrator import DoubleIntegratorEnv, DoubleIntegratorState
from .gcbf_safety_layer import GCBFSafetyLayer

# 注意：专注于双积分器BPTT研究，已移除其他环境实现
