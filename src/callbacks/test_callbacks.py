# src/callbacks/test_callbacks.py
from __future__ import annotations

import csv
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
from pathlib import Path

import torch
from PIL import Image


from datetime import datetime

class TestVisCallback(pl.Callback):
    def __init__(self, save_dir: str = "outputs", dataset_name: str = "",
                 model_name: str = "", num_samples: int = float("inf"), enabled: bool = False,
                 overlay_on_image: bool = False):
        super().__init__()
        self.enabled = enabled

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}-{dataset_name}-{model_name}"
        self.save_dir = Path(save_dir) / dir_name
        self.img_dir = self.save_dir / "plots"
        self.csv_path = self.save_dir / "per_sample_metrics.csv"

        self.max_samples = num_samples
        self.samples_count = 0
        self.test_metrics = None
        self.alpha = 0.4
        self.overlay_on_image = overlay_on_image


    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return

        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.test_metrics = pl_module.build_metrics().to(pl_module.device)

        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            metric_names = list(self.test_metrics.keys())
            header = ["sample_idx"] + metric_names
            writer.writerow(header)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                          outputs: Dict[str, torch.Tensor], batch: Any,
                          batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.enabled:
            return

        preds = outputs["preds"].to(pl_module.device)
        masks = outputs["mask"].to(pl_module.device)
        imgs = outputs["image"]
        filenames = outputs["filenames"]

        batch_size = preds.shape[0]

        for i in range(batch_size):
            p = preds[i:i + 1]
            m = masks[i:i + 1]
            img_tensor = imgs[i]
            fname = str(filenames[i])

            self.test_metrics.reset()
            self.test_metrics.update(p, m)
            results = self.test_metrics.compute()

            row = {"sample_idx": fname}
            for k, v in results.items():
                row[k] = f"{v.item():.4f}"
            self._write_to_csv(row)

            if self.samples_count < self.max_samples:
                self._save_overlay(img_tensor, m, p, fname)
                self.samples_count += 1

    def _write_to_csv(self, row_dict: dict) -> None:
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _save_overlay(self, img: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, filename: str) -> None:
        # (C, H, W) -> (H, W, C) -> uint8 [0, 255]
        img_np = img.detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC

        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        m = mask.detach().cpu().squeeze().numpy().astype(bool)
        p_prob = pred.detach().cpu().squeeze().numpy()
        p = (p_prob > 0.5).astype(bool)

        tp_mask = m & p
        fn_mask = m & (~p)
        fp_mask = (~m) & p

        overlay = np.zeros_like(img_np)
        overlay[tp_mask] = [0, 255, 0]  # green TP (Both True)
        overlay[fn_mask] = [255, 0, 0]  # red FN (GT True, Pred False)
        overlay[fp_mask] = [0, 0, 255]  # blue FP (GT False, Pred True)

        if self.overlay_on_image:
            roi = tp_mask | fn_mask | fp_mask
            final_img = img_np.copy()

            if roi.any():
                final_img[roi] = (
                        img_np[roi] * (1 - self.alpha) +
                        overlay[roi] * self.alpha
                ).astype(np.uint8)
        else:
            final_img = overlay

        save_path = self.img_dir / f"{filename}.png"
        Image.fromarray(final_img).save(save_path)