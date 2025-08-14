from flax import core, struct
from jaxtyping import Float, Int
from typing import Any
import jax.numpy as jnp
import jax.random as jr

# jax类型
Array = jnp.ndarray
PRNGKey = Array  # JAX PRNGKey is just an Array
Action = Float[Array, "batch action_dim"]
Obs = Float[Array, "batch obs_dim"]
State = Float[Array, "batch state_dim"]

# 环境类型
Pos = Float[Array, "2"]
Vel = Float[Array, "2"]

Pos2d = Float[Array, "2"]
Pos3d = Float[Array, "3"]

Vector2d = Float[Array, "2"]
Vector3d = Float[Array, "3"]

# 批处理位置和速度
BatchPos = Float[Array, "batch 2"]
BatchVel = Float[Array, "batch 2"]
BatchPos2d = Float[Array, "batch 2"]
BatchPos3d = Float[Array, "batch 3"]

BatchVector2d = Float[Array, "batch 2"]
BatchVector3d = Float[Array, "batch 3"]

# 神经网络类型
PolicyParams = Any
CBFParams = Any

# 障碍物
Obstacle = Float[Array, "n_obs obstacle_dim"]
