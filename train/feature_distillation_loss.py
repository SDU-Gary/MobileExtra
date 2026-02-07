#!/usr/bin/env python3
"""
特征蒸馏损失模块
用于Teacher→Student知识蒸馏的中间特征对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """
    五层特征蒸馏损失

    在U-Net的5个关键层进行特征对齐：
    - encoder3: 浅层纹理特征
    - encoder5: 深层语义特征
    - bottleneck: 最深层表示
    - decoder2: 中层重建特征
    - decoder4: 输出层特征

    使用1×1卷积将Teacher特征投影到Student通道数，然后计算MSE损失
    """

    def __init__(
        self,
        teacher_channels: dict,
        student_channels: dict,
        layer_weights: dict = None,
        use_l1: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            teacher_channels: Teacher各层通道数，例如:
                {'encoder3': 128, 'encoder5': 256, 'bottleneck': 256, 'decoder2': 128, 'decoder4': 64}
            student_channels: Student各层通道数，例如:
                {'encoder3': 40, 'encoder5': 80, 'bottleneck': 80, 'decoder2': 40, 'decoder4': 20}
            layer_weights: 各层权重，例如:
                {'encoder3': 0.1, 'encoder5': 0.15, 'bottleneck': 0.2, 'decoder2': 0.1, 'decoder4': 0.05}
                如果为None，则使用默认权重
            use_l1: 是否使用L1损失（默认MSE）
            normalize: 是否对特征进行L2归一化
        """
        super().__init__()

        # 默认权重（bottleneck最重要，输出层次之）
        if layer_weights is None:
            layer_weights = {
                'encoder3': 0.1,
                'encoder5': 0.15,
                'bottleneck': 0.2,
                'decoder2': 0.1,
                'decoder4': 0.05,
            }

        self.layer_weights = layer_weights
        self.use_l1 = use_l1
        self.normalize = normalize

        # 创建投影层（1×1卷积）
        self.projections = nn.ModuleDict()
        for layer_name in teacher_channels.keys():
            if layer_name not in student_channels:
                continue

            t_ch = teacher_channels[layer_name]
            s_ch = student_channels[layer_name]

            # 1×1卷积投影 Teacher → Student通道数
            self.projections[layer_name] = nn.Conv2d(
                t_ch, s_ch, kernel_size=1, stride=1, padding=0, bias=False
            )

        # 初始化投影层权重
        self._initialize_projections()

    def _initialize_projections(self):
        """初始化投影层权重（Xavier均匀初始化）"""
        for proj in self.projections.values():
            nn.init.xavier_uniform_(proj.weight)

    def _normalize_features(self, feat: torch.Tensor) -> torch.Tensor:
        """L2归一化特征图（沿通道维度）"""
        return F.normalize(feat, p=2, dim=1)

    def _compute_layer_loss(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        计算单层蒸馏损失

        Args:
            teacher_feat: Teacher特征 [B, C_t, H, W]
            student_feat: Student特征 [B, C_s, H, W]
            layer_name: 层名称

        Returns:
            loss: 标量损失
        """
        # 投影Teacher特征
        proj = self.projections[layer_name]
        teacher_proj = proj(teacher_feat)  # [B, C_s, H, W]

        # 可选：L2归一化
        if self.normalize:
            teacher_proj = self._normalize_features(teacher_proj)
            student_feat = self._normalize_features(student_feat)

        # 计算损失
        if self.use_l1:
            loss = F.l1_loss(student_feat, teacher_proj, reduction='mean')
        else:
            loss = F.mse_loss(student_feat, teacher_proj, reduction='mean')

        return loss

    def forward(
        self,
        teacher_features: dict,
        student_features: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        前向传播

        Args:
            teacher_features: Teacher中间特征字典，例如:
                {'encoder3': [B,128,H/4,W/4], 'encoder5': [B,256,H/16,W/16], ...}
            student_features: Student中间特征字典，例如:
                {'encoder3': [B,40,H/4,W/4], 'encoder5': [B,80,H/16,W/16], ...}

        Returns:
            total_loss: 加权总损失
            layer_losses: 各层损失字典（用于监控）
        """
        total_loss = 0.0
        layer_losses = {}

        for layer_name, weight in self.layer_weights.items():
            if layer_name not in teacher_features or layer_name not in student_features:
                continue

            t_feat = teacher_features[layer_name]
            s_feat = student_features[layer_name]

            # 检查空间尺寸是否匹配
            if t_feat.shape[2:] != s_feat.shape[2:]:
                # 如果尺寸不匹配，插值对齐
                s_feat = F.interpolate(
                    s_feat,
                    size=t_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

            # 计算单层损失
            layer_loss = self._compute_layer_loss(t_feat, s_feat, layer_name)

            # 加权累加
            total_loss += weight * layer_loss
            layer_losses[f'distill_{layer_name}'] = layer_loss.item()

        return total_loss, layer_losses

    def get_projection_stats(self) -> dict:
        """
        获取投影层权重统计信息（用于调试）

        Returns:
            stats: 各层投影权重的均值和标准差
        """
        stats = {}
        for layer_name, proj in self.projections.items():
            weight = proj.weight.data
            stats[layer_name] = {
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'min': weight.min().item(),
                'max': weight.max().item(),
            }
        return stats


class AdaptiveFeatureDistillationLoss(FeatureDistillationLoss):
    """
    自适应特征蒸馏损失

    在基础特征蒸馏的基础上，添加自适应权重调整：
    - 根据各层损失的相对大小动态调整权重
    - 避免某些层的损失dominate整体优化
    """

    def __init__(
        self,
        teacher_channels: dict,
        student_channels: dict,
        layer_weights: dict = None,
        use_l1: bool = False,
        normalize: bool = True,
        adaptive_factor: float = 0.1,
    ):
        """
        Args:
            adaptive_factor: 自适应调整因子（0.0表示不调整，1.0表示完全自适应）
        """
        super().__init__(
            teacher_channels=teacher_channels,
            student_channels=student_channels,
            layer_weights=layer_weights,
            use_l1=use_l1,
            normalize=normalize,
        )
        self.adaptive_factor = adaptive_factor
        self.register_buffer('loss_ema', torch.ones(len(layer_weights)))
        self.layer_names = list(layer_weights.keys())

    def forward(
        self,
        teacher_features: dict,
        student_features: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        前向传播（自适应版本）
        """
        layer_losses_raw = []
        layer_losses_dict = {}

        # 计算各层原始损失
        for layer_name in self.layer_names:
            if layer_name not in teacher_features or layer_name not in student_features:
                continue

            t_feat = teacher_features[layer_name]
            s_feat = student_features[layer_name]

            if t_feat.shape[2:] != s_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[2:],
                    mode='bilinear', align_corners=False,
                )

            layer_loss = self._compute_layer_loss(t_feat, s_feat, layer_name)
            layer_losses_raw.append(layer_loss)
            layer_losses_dict[f'distill_{layer_name}'] = layer_loss.item()

        if len(layer_losses_raw) == 0:
            return torch.tensor(0.0, device=teacher_features[self.layer_names[0]].device), {}

        # 计算自适应权重
        layer_losses_tensor = torch.stack(layer_losses_raw)

        # 更新EMA
        self.loss_ema = (
            (1 - self.adaptive_factor) * self.loss_ema +
            self.adaptive_factor * layer_losses_tensor.detach()
        )

        # 归一化权重（损失越大，权重越小）
        adaptive_weights = 1.0 / (self.loss_ema + 1e-6)
        adaptive_weights = adaptive_weights / adaptive_weights.sum()

        # 加权总损失
        total_loss = (layer_losses_tensor * adaptive_weights).sum()

        # 记录自适应权重
        for i, layer_name in enumerate(self.layer_names):
            if i < len(adaptive_weights):
                layer_losses_dict[f'weight_{layer_name}'] = adaptive_weights[i].item()

        return total_loss, layer_losses_dict


def create_feature_distillation_loss(
    teacher_base_channels: int = 64,
    student_base_channels: int = 20,
    adaptive: bool = False,
    **kwargs,
) -> FeatureDistillationLoss:
    """
    工厂函数：创建特征蒸馏损失模块

    Args:
        teacher_base_channels: Teacher基础通道数（例如64）
        student_base_channels: Student基础通道数（例如20）
        adaptive: 是否使用自适应版本
        **kwargs: 其他参数传递给损失模块

    Returns:
        loss_module: 特征蒸馏损失模块
    """
    # 根据base_channels计算各层通道数
    teacher_channels = {
        'encoder3': teacher_base_channels * 2,      # 128 (base=64)
        'encoder5': teacher_base_channels * 4,      # 256
        'bottleneck': teacher_base_channels * 4,    # 256
        'decoder2': teacher_base_channels * 2,      # 128
        'decoder4': teacher_base_channels * 1,      # 64
    }

    student_channels = {
        'encoder3': student_base_channels * 2,      # 40 (base=20)
        'encoder5': student_base_channels * 4,      # 80
        'bottleneck': student_base_channels * 4,    # 80
        'decoder2': student_base_channels * 2,      # 40
        'decoder4': student_base_channels * 1,      # 20
    }

    if adaptive:
        return AdaptiveFeatureDistillationLoss(
            teacher_channels=teacher_channels,
            student_channels=student_channels,
            **kwargs,
        )
    else:
        return FeatureDistillationLoss(
            teacher_channels=teacher_channels,
            student_channels=student_channels,
            **kwargs,
        )
