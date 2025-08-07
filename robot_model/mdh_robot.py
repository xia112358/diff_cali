import torch
import torch.nn as nn
from .unified_robot import BaseRobot

class DHLinkWithBeta(nn.Module):
    """
    基于torch的DH参数连杆,支持beta参数,适合微分运动学。
    :param a: 连杆长度
    :param alpha: 连杆扭角 
    :param d: 连杆偏置
    :param theta: 关节变量(旋转关节)
    :param beta: beta参数(平行关节奇异性补偿)
    :param sigma: 关节类型 (0:旋转, 1:移动)
    :param offset: 关节变量偏移 
    :param flip: 关节反向运动
    :param qlim: 关节变量限制 [min, max]    
    :param name: 连杆名称
    """

    def __init__(
        self,
        a=0.0,
        alpha=0.0,
        d=0.0,
        theta=0.0,
        beta=0.0,
        sigma=0,      # 0:旋转, 1:移动
        offset=0.0,
        flip=False,
        qlim=None,
        name=None,
    ):
        super(DHLinkWithBeta, self).__init__()
        
        # 使用nn.Parameter注册参数，以便梯度优化
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float64))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float64))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float64))
        
        # 非可训练参数
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.int))
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float64))
        self.flip = bool(flip)
        self.qlim = qlim
        self.name = name
    
    def A(self, q):
        """
        计算当前关节变量q下的齐次变换矩阵:
        始终使用带beta的完整DH变换矩阵
        """
        if self.flip:
            q_val = -q + self.offset
        else:
            q_val = q + self.offset

        if self.sigma == 0:  # 旋转关节
            theta = q_val
            d_val = self.d
        else:  # 移动关节
            theta = self.theta
            d_val = q_val

        a = self.a
        alpha = self.alpha
        beta = self.beta
        
        # 计算三角函数
        sa = torch.sin(alpha)
        ca = torch.cos(alpha)
        sb = torch.sin(beta)
        cb = torch.cos(beta)
        st = torch.sin(theta)
        ct = torch.cos(theta)
        
        # 构建4x4变换矩阵
        # 使用torch.stack构建矩阵，确保可以处理梯度
        row0 = torch.stack([ct*cb - st*sa*sb, -st*ca, ct*sb + st*sa*cb, a*ct])
        row1 = torch.stack([st*cb + ct*sa*sb,  ct*ca, st*sb - ct*sa*cb, a*st])
        row2 = torch.stack([-ca*sb,            sa,    ca*cb,            d_val])
        row3 = torch.stack([
            torch.zeros_like(ct),
            torch.zeros_like(ct),
            torch.zeros_like(ct),
            torch.ones_like(ct)
        ])
        T = torch.stack([row0, row1, row2, row3])
        return T
    
    def __repr__(self):
        return (f"DHLinkWithBeta(a={self.a.item()}, alpha={self.alpha.item()}, d={self.d.item()}, "
                f"theta={self.theta.item()}, beta={self.beta.item()}, sigma={self.sigma.item()}, "
                f"offset={self.offset.item()}, flip={self.flip}, name={self.name})")


class RevoluteDHWithBeta(DHLinkWithBeta):
    """
    基于torch的旋转关节,使用标准DH约定,支持beta参数。
    
    :param d: 连杆偏置
    :param a: 连杆长度
    :param alpha: 连杆扭角
    :param beta: beta参数
    :param offset: 关节变量偏移
    :param qlim: 关节变量限制 [min, max]
    :param flip: 关节反向运动
    :param name: 连杆名称
    """
    
    def __init__(
        self, d=0.0, a=0.0, alpha=0.0, beta=0.0, offset=0.0, qlim=None, flip=False, name=None
    ):
        theta = 0.0
        sigma = 0  # 旋转关节
        
        super().__init__(
            d=d,
            a=a,
            alpha=alpha,
            theta=theta,
            beta=beta,
            sigma=sigma,
            offset=offset,
            qlim=qlim,
            flip=flip,
            name=name,
        )


class DHRobotWithBeta(BaseRobot):
    """
    基于torch的DH机器人,支持beta参数,继承自BaseRobot统一接口
    """

    def __init__(self, links, name="DHRobotWithBeta", base=None):
        super(DHRobotWithBeta, self).__init__(name=name, robot_type="mdh")
        
        if not isinstance(links, list):
            raise TypeError("The links must be stored in a list.")
        
        # 使用ModuleList存储连杆
        self.links = nn.ModuleList(links)
        self.n = len(links)
        
        # 设置base变换
        if base is not None:
            self.base.data = base.clone()
    
    def fkine(self, q):
        """
        前向运动学，包含手眼变换矩阵
        
        :param q: 关节角度向量
        :return: 末端位姿变换矩阵(包含手眼变换)
        """
        if len(q) != self.n:
            raise ValueError(f"Expected {self.n} joint angles, got {len(q)}")
        
        T = self.base.clone()
        for i, (link, qi) in enumerate(zip(self.links, q)):
            T = T @ link.A(qi)
        T = T @ self.tool  # tool矩阵即为手眼变换
        
        return T


# 创建工厂函数以保持向后兼容性
def create_robot_mdh(links, name="DHRobotWithBeta", base=None):
    """创建MDH机器人的工厂函数"""
    return DHRobotWithBeta(links, name, base)
