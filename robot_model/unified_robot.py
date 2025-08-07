import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseRobot(nn.Module, ABC):
    """
    统一的机器人基类，支持不同的运动学表示方法
    """
    
    def __init__(self, name: str = "Robot", robot_type: str = "unknown"):
        super(BaseRobot, self).__init__()
        self.name = name
        self.robot_type = robot_type  # "mdh"
        self.n = 0  # 关节数量
        
        # 统一的手眼参数 (所有机器人都有)
        self.tx = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.ty = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.tz = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.rx = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.ry = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.rz = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
        # 基座变换矩阵 (所有机器人都有)
        self.register_buffer('base', torch.eye(4, dtype=torch.float64))
    
    @property
    def tool(self):
        """动态计算手眼变换矩阵"""
        return self._get_hand_eye_transform()
    
    def _get_hand_eye_transform(self):
        """
        使用当前的手眼参数构建4x4变换矩阵
        """
        # 平移部分
        translation = torch.stack([self.tx, self.ty, self.tz])
        
        # 旋转部分(使用欧拉角ZYX顺序)
        cx = torch.cos(self.rx)
        sx = torch.sin(self.rx)
        cy = torch.cos(self.ry)
        sy = torch.sin(self.ry)
        cz = torch.cos(self.rz)
        sz = torch.sin(self.rz)
        
        # ZYX欧拉角旋转矩阵
        R = torch.stack([
            torch.stack([cy*cz, -cy*sz, sy]),
            torch.stack([sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy]),
            torch.stack([-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy])
        ])
        
        # 构建4x4变换矩阵
        T_tool = torch.eye(4, dtype=torch.float64)
        T_tool[:3, :3] = R
        T_tool[:3, 3] = translation
        
        return T_tool
    
    @abstractmethod
    def fkine(self, q):
        """
        前向运动学计算 (子类必须实现)
        
        :param q: 关节角度向量
        :return: 末端位姿变换矩阵(包含手眼变换)
        """
        pass
    
    def fkine_batch(self, q_batch):
        """
        批量计算前向运动学
        
        :param q_batch: 批量关节角度 [batch_size, n_joints]
        :return: 批量末端位姿变换矩阵 [batch_size, 4, 4]
        """
        if q_batch.dim() != 2:
            raise ValueError(f"Expected 2D tensor [batch_size, n_joints], got {q_batch.shape}")
        
        batch_size, n_joints = q_batch.shape
        if n_joints != self.n:
            raise ValueError(f"Expected {self.n} joint angles, got {n_joints}")
        
        # 使用torch.vmap实现批量计算，如果不支持则回退到循环方式
        try:
            return torch.vmap(self.fkine)(q_batch)
        except (RuntimeError, AttributeError):
            # 回退到手动批量处理（兼容老版本PyTorch）
            results = []
            for i in range(batch_size):
                results.append(self.fkine(q_batch[i]))
            return torch.stack(results)
    
    def forward(self, q):
        """
        前向传播，用于nn.Module
        """
        if q.dim() == 1:
            return self.fkine(q)
        else:
            return self.fkine_batch(q)
    
    def set_hand_eye_params(self, tx=None, ty=None, tz=None, rx=None, ry=None, rz=None):
        """
        设置手眼参数的便捷方法
        """
        if tx is not None:
            self.tx.data = torch.tensor(tx, dtype=torch.float64)
        if ty is not None:
            self.ty.data = torch.tensor(ty, dtype=torch.float64)
        if tz is not None:
            self.tz.data = torch.tensor(tz, dtype=torch.float64)
        if rx is not None:
            self.rx.data = torch.tensor(rx, dtype=torch.float64)
        if ry is not None:
            self.ry.data = torch.tensor(ry, dtype=torch.float64)
        if rz is not None:
            self.rz.data = torch.tensor(rz, dtype=torch.float64)
    
    def get_hand_eye_params(self):
        """
        获取当前手眼参数
        """
        return {
            'tx': self.tx.item(),
            'ty': self.ty.item(),
            'tz': self.tz.item(),
            'rx': self.rx.item(),
            'ry': self.ry.item(),
            'rz': self.rz.item()
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.robot_type}', n={self.n})"
