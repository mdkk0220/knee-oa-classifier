# src/explain/viz_utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def visualize_cam(img_path, cam, save_path=None, show=False):
    """Save or display Grad-CAM result."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = overlay_heatmap(img, cam)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close()
    return overlay
