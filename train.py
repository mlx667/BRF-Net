# train.py
from __future__ import annotations

import argparse

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src.callbacks.earlystop_print import EarlyStopPrinter
from src.data.datamodule import MultiDatasetDataModule
from src.models.registry import build_model, get_active_model
from src.lightning.seg_module import SegLitModule
from src.callbacks.visualize import SegVisCallback
from src.callbacks.rich_progress import CustomRichProgressBar

torch.set_float32_matmul_precision('medium')


def main():
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(int(cfg.seed), workers=True)

    # data
    dm = MultiDatasetDataModule(cfg)
    dm.setup("fit")  # 新增：提前构建 split
    has_val = dm.val_ds is not None and len(dm.val_ds) > 0  # 新增：是否有验证集
    print(f"[Train] has_val={has_val}")

    # model
    active_model_cfg = get_active_model(cfg)
    model = build_model(active_model_cfg["name"], **active_model_cfg.get("kwargs", {}))
    lit = SegLitModule(model=model, cfg=cfg)

    # logging
    logger = TensorBoardLogger(save_dir=cfg.logging.save_dir, name=cfg.experiment_name, default_hp_metric=False)
    print(f"[Logger] Tensorboard is supported, try to use by running `tensorboard --logdir={cfg.logging.save_dir}`")

    # callbacks
    ckpt = ModelCheckpoint(               ############Synapse
        monitor="val/Dice",
        mode = "max",
        save_top_k=1,
        filename = "{epoch:03d}-{val_Dice:.4f}",
        auto_insert_metric_name=False,
    )
    lr_mon = LearningRateMonitor(logging_interval="step")

    vis_cb = SegVisCallback(
        fixed_ids=None,
        fixed_first_n=4,
        topk=cfg.logging.topk,
        threshold=cfg.metrics.threshold,
    )

    progress = CustomRichProgressBar(bar_width=15)

    early_stop = EarlyStopping(                       ############Synapse
        monitor=cfg.trainer.early_stopping.monitor,
        mode=cfg.trainer.early_stopping.mode,
        patience=cfg.trainer.early_stopping.patience,
        min_delta=cfg.trainer.early_stopping.min_delta,
        verbose=False,
    )
    # EarlyStopPrinter(monitor="train/loss", mode="min"),


    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        # limit_val_batches=0,  ##加
        # num_sanity_val_steps=0, ##加
        logger=logger,
        callbacks=[
            ckpt,
            early_stop,
            EarlyStopPrinter(
                monitor=cfg.trainer.early_stopping.monitor,
                mode=cfg.trainer.early_stopping.mode),
            # EarlyStopPrinter(
            #     monitor="train/loss",  # 你现在监控 train/loss
            #     mode="min",
            # ),
            lr_mon,
            progress,
            vis_cb,
        ],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        fast_dev_run=False
    )

    trainer.fit(lit, datamodule=dm)

    if ckpt.best_model_path:
        trainer.test(lit, datamodule=dm, ckpt_path=ckpt.best_model_path, weights_only=False)
    else:
        trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ETIS-LaribPolypDB.yaml")
    args = parser.parse_args()
    main()
