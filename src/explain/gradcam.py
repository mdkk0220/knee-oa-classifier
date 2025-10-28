# src/explain/gradcam.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, use_cuda: bool | None = None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.device = torch.device("cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)

        # forward & backward hook 등록
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        for h in [getattr(self, "fwd_handle", None), getattr(self, "bwd_handle", None)]:
            try:
                if h:
                    h.remove()
            except Exception:
                pass

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        x: (1,C,H,W)
        return: (HxW CAM tensor, predicted_class)
        """
        x = x.to(self.device)
        x.requires_grad = True

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        loss = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)

        grads = self.gradients  # dY/dA
        acts = self.activations  # A
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (N,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # (H,W)
        cam = self._normalize_cam(cam)
        return cam.detach().cpu(), class_idx
