import torch

class DHLinkWithBeta:
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
        self._a = torch.tensor(a, dtype=torch.float64)
        self._alpha = torch.tensor(alpha, dtype=torch.float64)
        self._d = torch.tensor(d, dtype=torch.float64)
        self._theta = torch.tensor(theta, dtype=torch.float64)
        self._beta = torch.tensor(beta, dtype=torch.float64)
        self._sigma = int(sigma)
        self._offset = torch.tensor(offset, dtype=torch.float64)
        self._flip = bool(flip)
        self.qlim = qlim
        self.name = name
    
    
    
    
    # ----------------- DH参数属性 -----------------
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, value):
        if isinstance(value, torch.Tensor):
            self._a = value.detach().clone().requires_grad_(True)
        else:
            self._a = torch.tensor(value, dtype=torch.float64)

    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, value):
        if isinstance(value, torch.Tensor):
            self._alpha = value.detach().clone().requires_grad_(True)
        else:
            self._alpha = torch.tensor(value, dtype=torch.float64)

    @property
    def d(self):
        return self._d
    @d.setter
    def d(self, value):
        if isinstance(value, torch.Tensor):
            self._d = value.detach().clone().requires_grad_(True)
        else:
            self._d = torch.tensor(value, dtype=torch.float64)

    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        if isinstance(value, torch.Tensor):
            self._theta = value.detach().clone().requires_grad_(True)
        else:
            self._theta = torch.tensor(value, dtype=torch.float64)

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        if isinstance(value, torch.Tensor):
            self._beta = value.detach().clone().requires_grad_(True)
        else:
            self._beta = torch.tensor(value, dtype=torch.float64)

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = int(value)

    @property
    def offset(self):
        return self._offset
    @offset.setter
    def offset(self, value):
        if isinstance(value, torch.Tensor):
            self._offset = value.detach().clone().requires_grad_(True)
        else:
            self._offset = torch.tensor(value, dtype=torch.float64)

    @property
    def flip(self):
        return self._flip
    @flip.setter
    def flip(self, value):
        self._flip = bool(value)
        
        
        
        

    # ----------------- 变换矩阵 -----------------
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

        # 始终使用包含beta参数的完整变换矩阵
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
    
    
    # ----------------- 字符串表示 -----------------
    def __repr__(self):
        return (f"DHLinkWithBeta(a={self.a.item()}, alpha={self.alpha.item()}, d={self.d.item()}, "
                f"theta={self.theta.item()}, beta={self.beta.item()}, sigma={self.sigma}, "
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




class DHRobotWithBeta:
    """
    基于torch的DH机器人,支持beta参数,tool矩阵代表手眼变换,适合微分运动学。
    """

    def __init__(self, links, name="DHRobotWithBeta", base=None, tool=None):
        if not isinstance(links, list):
            raise TypeError("The links must be stored in a list.")
        
        self.links = links
        self.n = len(links)
        self.name = name
        
        # 设置base变换
        if base is None:
            self.base = torch.eye(4, dtype=torch.float64)
        else:
            self.base = base
        
        # 初始化手眼参数(无论是否提供自定义tool)
        self._tx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self._ty = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self._tz = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self._rx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self._ry = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self._rz = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            
        # 设置tool变换
        if tool is None:
            # 使用参数化的tool变换
            self._tool = None  # 将动态计算
            self._use_custom_tool = False
        else:
            # 使用提供的固定tool变换
            self._tool = tool
            self._use_custom_tool = True
    
    @property
    def tool(self):
        """动态计算tool变换矩阵"""
        if self._use_custom_tool:
            # 使用固定的tool变换
            return self._tool
        else:
            # 使用参数化的tool变换
            return self._get_tool_transform()
    
    # ----------------- 手眼参数属性 -----------------
    @property
    def tx(self):
        return self._tx
    @tx.setter
    def tx(self, value):
        if isinstance(value, torch.Tensor):
            self._tx = value.detach().clone().requires_grad_(True)
        else:
            self._tx = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    
    @property
    def ty(self):
        return self._ty
    @ty.setter
    def ty(self, value):
        if isinstance(value, torch.Tensor):
            self._ty = value.detach().clone().requires_grad_(True)
        else:
            self._ty = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    
    @property
    def tz(self):
        return self._tz
    @tz.setter
    def tz(self, value):
        if isinstance(value, torch.Tensor):
            self._tz = value.detach().clone().requires_grad_(True)
        else:
            self._tz = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    
    @property
    def rx(self):
        return self._rx
    @rx.setter
    def rx(self, value):
        if isinstance(value, torch.Tensor):
            self._rx = value.detach().clone().requires_grad_(True)
        else:
            self._rx = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    
    @property
    def ry(self):
        return self._ry
    @ry.setter
    def ry(self, value):
        if isinstance(value, torch.Tensor):
            self._ry = value.detach().clone().requires_grad_(True)
        else:
            self._ry = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    
    @property
    def rz(self):
        return self._rz
    @rz.setter
    def rz(self, value):
        if isinstance(value, torch.Tensor):
            self._rz = value.detach().clone().requires_grad_(True)
        else:
            self._rz = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    
    
    
    
    def _get_tool_transform(self):
        """
        使用当前的手眼参数(tx, ty, tz, rx, ry, rz)构建4x4变换矩阵
        
        :return: 4x4变换矩阵
        """
        # 平移部分
        translation = torch.stack([self._tx, self._ty, self._tz])
        
        # 旋转部分(使用欧拉角ZYX顺序)
        cx = torch.cos(self._rx)
        sx = torch.sin(self._rx)
        cy = torch.cos(self._ry)
        sy = torch.sin(self._ry)
        cz = torch.cos(self._rz)
        sz = torch.sin(self._rz)
        
        # ZYX欧拉角旋转矩阵
        R = torch.stack([
            torch.stack([cy*cz, -cy*sz, sy]),
            torch.stack([sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy]),
            torch.stack([-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy])
        ])
        
        # 构建4x4变换矩阵
        T_tool = torch.zeros(4, 4, dtype=torch.float64)
        T_tool[:3, :3] = R
        T_tool[:3, 3] = translation
        T_tool[3, 3] = 1.0
        
        return T_tool

    def fkine(self, q):
        """
        前向运动学,tool变换矩阵包含手眼变换
        
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


    def __repr__(self):
        return f"DHRobotWithBeta(name='{self.name}', n={self.n})"

    def __str__(self):
        s = f"DHRobotWithBeta: {self.name}, {self.n} joints\n"
        for i, link in enumerate(self.links):
            s += f"  Link {i+1}: {link}\n"
        return s




