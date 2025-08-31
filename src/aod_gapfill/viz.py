from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt




def save_tile(img_in, img_truth, img_pred, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.axis("off"); plt.title("Input")
    plt.imshow(img_in, cmap="gray")
    plt.subplot(1,3,2); plt.axis("off"); plt.title("Truth")
    plt.imshow(img_truth, cmap="gray")
    plt.subplot(1,3,3); plt.axis("off"); plt.title(title)
    plt.imshow(img_pred, cmap="gray")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)