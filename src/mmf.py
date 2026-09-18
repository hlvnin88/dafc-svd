"""
MMAF — Multi-Metric Adaptive Fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (MMAF_FUSION_DIM, MMAF_ATTN_HIDDEN,
                     MMAF_CLS_HIDDEN, MMAF_DROPOUT)


class MMAF(nn.Module):
    """
    Multi-Metric Adaptive Fusion model.

    Args:
        in_dims: List[int] — số chiều của từng nhóm trong tổ hợp.
                 Theo Phụ lục A, tất cả phải = 256
                 (đã qua projection ở Bước 3).
        fusion_dim: Số chiều không gian hợp nhất (mặc định 256).
        attn_hidden: Số chiều ẩn của attention (mặc định 128).
        cls_hidden: Số chiều ẩn của classifier (mặc định 128).
        dropout: Tỉ lệ dropout (mặc định 0.3).
    """

    def __init__(self, in_dims, fusion_dim=MMAF_FUSION_DIM,
                 attn_hidden=MMAF_ATTN_HIDDEN,
                 cls_hidden=MMAF_CLS_HIDDEN,
                 dropout=MMAF_DROPOUT):
        super().__init__()

        # === KIỂM TRA: mọi input phải = fusion_dim (256) ===
        # Theo Phụ lục A: "Mỗi biểu diễn đầu vào được đưa về không gian 256
        # chiều. Sau đó, mỗi nhóm đặc trưng sử dụng một lớp Linear 256–256."
        for d in in_dims:
            if d != fusion_dim:
                raise ValueError(
                    f"MMAF requires all input dims = {fusion_dim}, got {d}. "
                    f"Run `extract_features.py` first to project F1–F10 "
                    f"to {fusion_dim}-dim.")

        # === (1) PROJECTION: Linear 256–256 → ReLU → Dropout ===
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in in_dims
        ])

        # === (2) ADAPTIVE WEIGHTING: 256–128 → tanh → 128–1 ===
        self.attn = nn.Sequential(
            nn.Linear(fusion_dim, attn_hidden),   # 256 → 128
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),            # 128 → 1
        )

        # === (3) CLASSIFIER: 256–128 → ReLU → Dropout → 128–1 → Sigmoid ===
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cls_hidden),    # 256 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1),             # 128 → 1
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.projections:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for module in self.attn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_list):
        """
        Args:
            x_list: List[Tensor] — mỗi tensor shape (batch_size, 256).

        Returns:
            Tensor shape (batch_size, 1) — xác suất dự đoán.
        """
        # (1) Projection per group
        h = [p(x) for p, x in zip(self.projections, x_list)]  # list (B, 256)

        # Stack: (B, m, 256)
        stacked = torch.stack(h, dim=1)

        # (2) Adaptive attention weights per sample
        scores = self.attn(stacked).squeeze(-1)                # (B, m)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)         # (B, m, 1)

        # (3) Weighted fusion
        fused = (stacked * alpha).sum(dim=1)                   # (B, 256)

        # (4) Classifier
        return self.classifier(fused)                          # (B, 1)


def count_parameters(model: MMAF) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mmaf_for_combo(combo: tuple, feature_dims: dict, **kwargs) -> MMAF:
    dims = [feature_dims[g] for g in combo]
    return MMAF(dims, **kwargs)