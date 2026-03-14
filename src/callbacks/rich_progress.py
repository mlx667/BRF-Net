# src/callbacks/rich_progress.py
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class _LRColumn(ProgressColumn):
    def __init__(self, trainer_getter, label="lr", fmt="{:.2e}"):
        super().__init__()
        self._get_trainer = trainer_getter
        self.label = label
        self.fmt = fmt

    def render(self, task):
        trainer = self._get_trainer()
        lr = None
        try:
            if trainer and trainer.optimizers:
                lr = trainer.optimizers[0].param_groups[0].get("lr", None)
        except Exception:
            lr = None
        return f"{self.label}={self.fmt.format(lr)}" if lr is not None else f"{self.label}=—"


class _VRAMColumn(ProgressColumn):
    def __init__(self, label="vram"):
        super().__init__()
        self.label = label

    def render(self, task):
        if not torch.cuda.is_available():
            return f"{self.label}=—"
        mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return f"{self.label}={mb:.0f}MB"


class CustomRichProgressBar(RichProgressBar):

    def __init__(self, bar_width: int = 20):
        theme = RichProgressBarTheme(
            description="white",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="green1",
            batch_progress="cyan",
            time="grey70",
            processing_speed="grey70",
            metrics="magenta",
        )
        super().__init__(theme=theme, leave=True)
        self._trainer: Optional[pl.Trainer] = None
        self.bar_width = int(bar_width)

    def _get_trainer(self) -> Optional[pl.Trainer]:
        return self._trainer

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._trainer = trainer
        return super().on_fit_start(trainer, pl_module)

    def configure_columns(self, trainer: pl.Trainer):
        return [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=self.bar_width),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("|"),
            _LRColumn(self._get_trainer),
            TextColumn("•"),
            _VRAMColumn(),
            TextColumn("|"),
        ]
