import torch
import torch.nn as nn


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


class DHRobotWithBeta(nn.Module):
    """
    基于torch的DH机器人,支持beta参数,集成统一的机器人接口
    """

    def __init__(self, links, name="DHRobotWithBeta", base=None):
        super(DHRobotWithBeta, self).__init__()
        
        # 基本属性
        self.name = name
        self.robot_type = "mdh"
        
        if not isinstance(links, list):
            raise TypeError("The links must be stored in a list.")
        
        # 使用ModuleList存储连杆
        self.links = nn.ModuleList(links)
        self.n = len(links)
        
        # 统一的手眼参数 (所有机器人都有)
        self.tx = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.ty = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.tz = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.rx = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.ry = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.rz = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
        # 基座变换矩阵 (所有机器人都有)
        self.register_buffer('base', torch.eye(4, dtype=torch.float64))
        
        # 设置base变换
        if base is not None:
            self.base.data = base.clone()
    
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



