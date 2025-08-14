import jax.tree_util as jtu
import numpy as np

from abc import ABC, abstractproperty, abstractmethod
from .data import Rollout
from .utils import jax2np, np2jax
from ..utils.utils import tree_merge
from ..utils.typing import Array


class Buffer(ABC):
    """
    抽象基类：用于存储和采样轨迹数据的缓冲区。
    
    为不同类型的经验缓冲区提供统一接口。
    """

    def __init__(self, size: int):
        """
        初始化缓冲区。
        
        参数:
            size: 缓冲区最大容量
        """
        self._size = size

    @abstractmethod
    def append(self, rollout: Rollout):
        """
        向缓冲区添加新的轨迹数据。
        
        参数:
            rollout: 要添加的轨迹数据
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Rollout:
        """
        从缓冲区中采样批次数据。
        
        参数:
            batch_size: 采样的批次大小
            
        返回:
            采样的轨迹数据
        """
        pass

    @abstractproperty
    def length(self) -> int:
        """
        获取缓冲区当前存储的数据量。
        
        返回:
            当前数据量
        """
        pass


class ReplayBuffer(Buffer):
    """
    标准经验重放缓冲区实现。
    
    采用先进先出（FIFO）策略存储轨迹数据，
    支持随机采样训练批次。
    """

    def __init__(self, size: int):
        """
        初始化重放缓冲区。
        
        参数:
            size: 缓冲区最大容量
        """
        super(ReplayBuffer, self).__init__(size)
        self._buffer = None

    def append(self, rollout: Rollout):
        """
        向缓冲区添加新轨迹，超出容量时删除最旧数据。
        
        参数:
            rollout: 新的轨迹数据
        """
        if self._buffer is None:
            self._buffer = jax2np(rollout)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)

    def sample(self, batch_size: int) -> Rollout:
        """
        随机采样训练批次。
        
        参数:
            batch_size: 采样的批次大小
            
        返回:
            随机采样的轨迹数据批次
        """
        idx = np.random.randint(0, self._buffer.length, batch_size)
        return np2jax(self.get_data(idx))

    def get_data(self, idx: np.ndarray) -> Rollout:
        """
        根据索引获取数据。
        
        参数:
            idx: 数据索引数组
            
        返回:
            对应索引的轨迹数据
        """
        return jtu.tree_map(lambda x: x[idx], self._buffer)

    @property
    def length(self) -> int:
        """
        获取缓冲区当前数据量。
        
        返回:
            当前存储的轨迹数量
        """
        if self._buffer is None:
            return 0
        return self._buffer.n_data


class MaskedReplayBuffer:
    """
    带掩码的重放缓冲区，支持按安全性分类存储轨迹。
    
    除了存储轨迹数据外，还维护安全掩码和不安全掩码，
    用于区分不同类型的经验数据。
    """

    def __init__(self, size: int):
        """
        初始化掩码重放缓冲区。
        
        参数:
            size: 缓冲区最大容量
        """
        self._size = size
        # (b, T) 维度的缓冲区
        self._buffer = None
        self._safe_mask = None
        self._unsafe_mask = None

    def append(self, rollout: Rollout, safe_mask: Array, unsafe_mask: Array):
        """
        向缓冲区添加带掩码的轨迹数据。
        
        参数:
            rollout: 轨迹数据
            safe_mask: 安全区域掩码
            unsafe_mask: 不安全区域掩码
        """
        if self._buffer is None:
            self._buffer = jax2np(rollout)
            self._safe_mask = jax2np(safe_mask)
            self._unsafe_mask = jax2np(unsafe_mask)
            # self._mid_mask = jax2np(mid_mask)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
            self._safe_mask = tree_merge([self._safe_mask, jax2np(safe_mask)])
            self._unsafe_mask = tree_merge([self._unsafe_mask, jax2np(unsafe_mask)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)
            self._safe_mask = jtu.tree_map(lambda x: x[-self._size:], self._safe_mask)
            self._unsafe_mask = jtu.tree_map(lambda x: x[-self._size:], self._unsafe_mask)

    def sample(self, batch_size: int) -> [Rollout, Array, Array]:
        """
        随机采样带掩码的训练批次。
        
        参数:
            batch_size: 采样的批次大小
            
        返回:
            包含轨迹数据、安全掩码和不安全掩码的元组
        """
        idx = np.random.randint(0, self._buffer.length, batch_size)
        rollout, safe_mask, unsafe_mask = self.get_data(idx)
        return rollout, safe_mask, unsafe_mask

    def get_data(self, idx: np.ndarray) -> [Rollout, Array, Array]:
        """
        根据索引获取带掩码的数据。
        
        参数:
            idx: 数据索引数组
            
        返回:
            对应索引的轨迹数据、安全掩码和不安全掩码
        """
        return jtu.tree_map(lambda x: x[idx], self._buffer), self._safe_mask[idx], self._unsafe_mask[idx]

    @property
    def length(self) -> int:
        """
        获取缓冲区当前数据量。
        
        返回:
            当前存储的轨迹数量
        """
        if self._buffer is None:
            return 0
        return self._buffer.n_data
