# ============================================================
# src/explain/gradcam.py | v2.1 CPU & macOS 안전 버전 (2025-11)
# ------------------------------------------------------------
# ✅ 주요 변경사항
# 1. x.requires_grad_(True) → CPU에서도 gradient 추적 확실히 설정
# 2. self.gradients / activations None 방지용 초기화
# 3. backward() 블로킹 방지 (retain_graph=False로 단순화)
# 4. _normalize_cam() 보강
# ============================================================

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, use_cuda: bool | None = None):
        """
        Grad-CAM 클래스: 대상 모델과 layer를 연결하여
        입력 이미지에 대한 class activation map을 계산한다.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.device = torch.device(
            "cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu"
        )
        self.model.to(self.device)

        # forward / backward hook 등록
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        """hook 제거"""
        for h in [getattr(self, "fwd_handle", None), getattr(self, "bwd_handle", None)]:
            try:
                if h:
                    h.remove()
            except Exception:
                pass

    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """CAM을 0~1 사이로 정규화"""
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) < 1e-6:
            return torch.zeros_like(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        x: (1, C, H, W)
        return: (HxW CAM tensor, predicted_class)
        """
        x = x.to(self.device)
        x.requires_grad_(True)  # ✅ CPU에서도 gradient 추적 강제 활성화

        # Forward pass
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass
        loss = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        loss.backward()  # retain_graph 제거 → macOS CPU 블로킹 방지

        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations properly.")

        # 가중치 계산 및 CAM 생성
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dimensions
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam = self._normalize_cam(cam)

        return cam.detach().cpu(), class_idx
