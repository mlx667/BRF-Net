# src/callbacks/earlystop_print.py
from __future__ import annotations

import pytorch_lightning as pl

class EarlyStopPrinter(pl.Callback):

    def __init__(self, monitor: str, mode: str = "min"):
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        val = metrics[self.monitor]
        try:
            val = float(val.detach().cpu().item() if hasattr(val, "detach") else val)
        except Exception:
            return

        improved = False
        if self.best is None:
            improved = True
        else:
            improved = (val < self.best) if self.mode == "min" else (val > self.best)

        if improved:
            self.best = val
            trainer.print(f"[EarlyStop] {self.monitor} improved -> best={self.best:.4f}")
