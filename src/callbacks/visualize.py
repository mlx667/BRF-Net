# src/callbacks/visualize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter


def _minmax01_per_image(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x: (B,1,H,W) -> (B,1,H,W) in [0,1]"""
    x = x.detach().float().cpu()
    out = []
    for i in range(x.shape[0]):
        xi = x[i]
        vmin, vmax = xi.min(), xi.max()
        xi = (xi - vmin) / (vmax - vmin + eps)
        out.append(xi)
    return torch.stack(out, 0)


def _to_rgb(gray01: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) -> (B,3,H,W)"""
    return gray01.repeat(1, 3, 1, 1)


def _overlay(base_rgb: torch.Tensor, mask: torch.Tensor, color: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    base_rgb: (B,3,H,W) in [0,1]
    mask: (B,1,H,W) in {0,1} or [0,1]
    color: (3,) in [0,1]
    """
    if mask.ndim == 3:
        mask = mask[:, None, ...]
    mask = mask.detach().float().cpu().clamp(0, 1)
    color = color.view(1, 3, 1, 1).float()
    a = float(alpha) * mask
    out = base_rgb * (1.0 - a) + color * a
    return out.clamp(0, 1)


def _dice_per_sample_binary(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """pred/gt: (B,1,H,W) -> dice: (B,)"""
    pred = (pred >= thr).float()
    gt = (gt >= 0.5).float()
    pred_f = pred.flatten(1)
    gt_f = gt.flatten(1)
    inter = (pred_f * gt_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + gt_f.sum(dim=1)
    return (2 * inter + eps) / (denom + eps)


def _dice_per_sample_multiclass(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, ignore_index: Optional[int] = 0, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/gt: (B,H,W) long
    返回 macro dice (按类平均) 的每样本分数: (B,)
    """
    pred = pred.detach()
    gt = gt.detach()
    b = pred.shape[0]
    dices = []

    classes = list(range(num_classes))
    if ignore_index is not None and ignore_index in classes:
        classes.remove(ignore_index)

    for i in range(b):
        pi = pred[i]
        gi = gt[i]
        per_cls = []
        for c in classes:
            p = (pi == c).float()
            g = (gi == c).float()
            inter = (p * g).sum()
            denom = p.sum() + g.sum()
            if denom.item() == 0:
                per_cls.append(torch.tensor(1.0, device=p.device))
            else:
                per_cls.append((2 * inter + eps) / (denom + eps))
        dices.append(torch.stack(per_cls, dim=0).mean())
    return torch.stack(dices, 0)



def _default_colormap(num_classes: int) -> torch.Tensor:
    """
    固定可复现的 colormap: (C,3) in [0,1]
    class 0（背景）默认不画/或颜色很淡
    """
    palette = torch.tensor([
        [0.0, 0.0, 0.0],  # 0 background (unused)
        [1.0, 0.0, 0.0],  # 1 red
        [0.0, 1.0, 0.0],  # 2 green
        [0.0, 0.0, 1.0],  # 3 blue
        [1.0, 1.0, 0.0],  # 4 yellow
        [1.0, 0.0, 1.0],  # 5 magenta
        [0.0, 1.0, 1.0],  # 6 cyan
        [1.0, 0.5, 0.0],  # 7 orange
        [0.6, 0.0, 1.0],  # 8 purple
        [0.0, 0.6, 0.3],  # 9 teal
    ], dtype=torch.float32)

    if num_classes <= palette.shape[0]:
        return palette[:num_classes]
    reps = (num_classes + palette.shape[0] - 1) // palette.shape[0]
    cm = palette.repeat(reps, 1)[:num_classes]
    cm[0] = torch.tensor([0.0, 0.0, 0.0])
    return cm


def _label_to_rgb(label: torch.Tensor, colormap: torch.Tensor) -> torch.Tensor:
    """
    label: (B,H,W) long
    colormap: (C,3)
    -> (B,3,H,W) float
    """
    label = label.detach().long().cpu().clamp(0, colormap.shape[0] - 1)
    b, h, w = label.shape
    rgb = colormap[label.view(-1)].view(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
    return rgb


@dataclass
class _VisPack:
    sid: str
    overlay: torch.Tensor   # (3,H,W)
    prob: torch.Tensor      # (1,H,W)  binary: prob; multiclass: confidence
    err: torch.Tensor       # (3,H,W)  binary: tpfpfn; multiclass: correct/incorrect
    focus_tpfpfn: Optional[torch.Tensor] = None


class SegVisCallback(pl.Callback):
    def __init__(
        self,
        threshold: float = 0.5,
        alpha_pred: float = 0.45,
        alpha_gt: float = 0.45,
        alpha_diff: float = 0.55,
        alpha_err: float = 0.55,
        fixed_ids: Optional[Sequence[str]] = None,
        fixed_first_n: int = 4,
        topk: int = 4,
        log_every_n_epochs: int = 1,
        tag_prefix: str = "val_vis",
        # multiclass settings
        task: Optional[str] = None,
        num_classes: Optional[int] = None,
        include_background: bool = False,
        focus_class: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.alpha_pred = float(alpha_pred)
        self.alpha_gt = float(alpha_gt)
        self.alpha_diff = float(alpha_diff)
        self.alpha_err = float(alpha_err)

        self.fixed_ids = list(fixed_ids) if fixed_ids is not None else None
        self.fixed_first_n = int(fixed_first_n)
        self.topk = int(topk)
        self.log_every_n_epochs = int(log_every_n_epochs)
        self.tag_prefix = str(tag_prefix)

        self.task = None if task is None else str(task).lower()
        self.num_classes = num_classes
        self.include_background = bool(include_background)
        self.focus_class = focus_class

        # binary fixed colors
        self.c_pred = torch.tensor([1.0, 0.0, 0.0])  # red
        self.c_gt = torch.tensor([0.0, 1.0, 0.0])    # green
        self.c_diff = torch.tensor([0.0, 0.0, 1.0])  # blue

        self.c_correct = torch.tensor([0.0, 1.0, 0.0])  # green
        self.c_wrong = torch.tensor([1.0, 0.0, 0.0])    # red
        self.c_diff_mc = torch.tensor([1.0, 1.0, 1.0])  # white

        # epoch cache
        self._worst: List[Tuple[float, _VisPack]] = []
        self._fixed_collected: Dict[str, _VisPack] = {}

        # lazy init
        self._cm: Optional[torch.Tensor] = None

    def _infer_task(self, batch_mask: torch.Tensor, pl_module: pl.LightningModule) -> str:
        if self.task is not None:
            return self.task
        if batch_mask.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            return "multiclass"
        try:
            t = str(pl_module.cfg.data.get("task", "binary")).lower()  # type: ignore
            if t in ("binary", "multiclass"):
                return t
        except Exception:
            pass
        return "binary"

    def _get_num_classes(self, pl_module: pl.LightningModule) -> int:
        if self.num_classes is not None:
            return int(self.num_classes)
        try:
            return int(pl_module.cfg.data.num_classes)  # type: ignore
        except Exception:
            raise ValueError("num_classes is required for multiclass visualization (pass num_classes or set cfg.data.num_classes).")

    @staticmethod
    def _get_writer(trainer: pl.Trainer) -> Optional[SummaryWriter]:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return None
        exp = logger.experiment
        if isinstance(exp, SummaryWriter):
            return exp
        try:
            return logger.experiment  # type: ignore
        except Exception:
            return None

    def _maybe_push_worst(self, score: float, pack: _VisPack):
        self._worst.append((score, pack))
        self._worst.sort(key=lambda x: x[0])  # ascending: smaller is worse
        if len(self._worst) > self.topk:
            self._worst = self._worst[: self.topk]

    # ----------------- build packs -----------------
    @torch.no_grad()
    def _build_pack_binary(self, img: torch.Tensor, gt: torch.Tensor, prob: torch.Tensor, sid: str) -> _VisPack:
        gray01 = _minmax01_per_image(img[None, ...])  # (1,1,H,W)
        base = _to_rgb(gray01)                        # (1,3,H,W)

        gt_bin = (gt >= 0.5).float().cpu()
        pred_bin = (prob >= self.threshold).float().cpu()

        out = base
        out = _overlay(out, gt_bin, self.c_gt, self.alpha_gt)
        out = _overlay(out, pred_bin, self.c_pred, self.alpha_pred)
        diff = (pred_bin != gt_bin).float()
        out = _overlay(out, diff, self.c_diff, self.alpha_diff)

        tp = (pred_bin * gt_bin).float()
        fp = (pred_bin * (1 - gt_bin)).float()
        fn = ((1 - pred_bin) * gt_bin).float()

        err = base
        err = _overlay(err, tp, torch.tensor([0.0, 1.0, 0.0]), self.alpha_err)
        err = _overlay(err, fp, torch.tensor([1.0, 0.0, 0.0]), self.alpha_err)
        err = _overlay(err, fn, torch.tensor([0.0, 0.0, 1.0]), self.alpha_err)

        prob01 = prob.detach().float().cpu().clamp(0, 1)
        return _VisPack(sid=sid, overlay=out[0], prob=prob01, err=err[0], focus_tpfpfn=None)

    @torch.no_grad()
    def _build_pack_multiclass(
        self,
        img: torch.Tensor,         # (1,H,W)
        gt: torch.Tensor,          # (H,W) long
        pred: torch.Tensor,        # (H,W) long
        conf: torch.Tensor,        # (1,H,W) float [0,1]
        sid: str,
        num_classes: int,
        ignore_bg: bool,
    ) -> _VisPack:
        gray01 = _minmax01_per_image(img[None, ...])  # (1,1,H,W)
        base = _to_rgb(gray01)                        # (1,3,H,W)

        if self._cm is None or self._cm.shape[0] != num_classes:
            self._cm = _default_colormap(num_classes)

        gt_rgb = _label_to_rgb(gt[None, ...], self._cm)     # (1,3,H,W)
        pred_rgb = _label_to_rgb(pred[None, ...], self._cm) # (1,3,H,W)

        if ignore_bg:
            gt_m = (gt[None, None, ...] != 0).float()
            pr_m = (pred[None, None, ...] != 0).float()
        else:
            gt_m = torch.ones_like(gt[None, None, ...]).float()
            pr_m = torch.ones_like(pred[None, None, ...]).float()

        out = base
        out = out * (1.0 - self.alpha_gt * gt_m.cpu()) + gt_rgb * (self.alpha_gt * gt_m.cpu())
        out = out * (1.0 - self.alpha_pred * pr_m.cpu()) + pred_rgb * (self.alpha_pred * pr_m.cpu())

        wrong = (pred != gt).float()  # (H,W)
        if ignore_bg:
            wrong = wrong * (gt != 0).float()
        out = _overlay(out, wrong[None, ...], self.c_diff_mc, self.alpha_diff)

        correct = (pred == gt).float()
        if ignore_bg:
            fg = (gt != 0).float()
            correct = correct * fg
            wrong2 = (pred != gt).float() * fg
        else:
            wrong2 = (pred != gt).float()

        err = base
        err = _overlay(err, correct[None, ...], self.c_correct, 0.30)
        err = _overlay(err, wrong2[None, ...], self.c_wrong, self.alpha_err)

        focus_img = None
        if self.focus_class is not None:
            c = int(self.focus_class)
            gt_bin = (gt == c).float()
            pr_bin = (pred == c).float()
            tp = (gt_bin * pr_bin)
            fp = (pr_bin * (1 - gt_bin))
            fn = ((1 - pr_bin) * gt_bin)
            focus_img = base
            focus_img = _overlay(focus_img, tp[None, ...], torch.tensor([0.0, 1.0, 0.0]), self.alpha_err)
            focus_img = _overlay(focus_img, fp[None, ...], torch.tensor([1.0, 0.0, 0.0]), self.alpha_err)
            focus_img = _overlay(focus_img, fn[None, ...], torch.tensor([0.0, 0.0, 1.0]), self.alpha_err)
            focus_img = focus_img[0]

        return _VisPack(
            sid=sid,
            overlay=out[0],
            prob=conf.detach().float().cpu().clamp(0, 1),
            err=err[0],
            focus_tpfpfn=focus_img,
        )

    # ----------------- lightning hooks -----------------
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._worst = []
        self._fixed_collected = {}

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return

        img = batch["image"].to(pl_module.device)  # (B,C,H,W)
        gt = batch["mask"].to(pl_module.device)    # binary: (B,1,H,W) float; multiclass: (B,H,W) long
        sids = batch.get("id", None)

        task = self._infer_task(gt, pl_module)

        logits = pl_module(img)

        if task == "binary":
            prob = torch.sigmoid(logits)  # (B,1,H,W)
            dice = _dice_per_sample_binary(prob, gt, thr=self.threshold)  # (B,)
            dice_cpu = dice.detach().cpu()

            b = img.shape[0]
            for i in range(b):
                sid = str(sids[i]) if sids is not None else f"idx_{batch_idx:04d}_{i:02d}"
                img_i = img[i, :1]
                gt_i = gt[i]      # (1,H,W)
                prob_i = prob[i]  # (1,H,W)
                pack = self._build_pack_binary(img_i, gt_i, prob_i, sid=sid)

                if self.fixed_ids is not None:
                    if sid in self.fixed_ids and sid not in self._fixed_collected:
                        self._fixed_collected[sid] = pack
                else:
                    if len(self._fixed_collected) < self.fixed_first_n:
                        self._fixed_collected[sid] = pack

                self._maybe_push_worst(float(dice_cpu[i].item()), pack)

        else:
            C = self._get_num_classes(pl_module)
            ignore_bg = (not self.include_background)

            # probs/conf/pred
            probs = F.softmax(logits, dim=1)          # (B,C,H,W)
            conf, pred = torch.max(probs, dim=1)      # conf: (B,H,W), pred: (B,H,W)
            conf = conf[:, None, ...]                 # (B,1,H,W)

            dice = _dice_per_sample_multiclass(pred, gt.long(), num_classes=C, ignore_index=(0 if ignore_bg else None))
            dice_cpu = dice.detach().cpu()

            b = img.shape[0]
            for i in range(b):
                sid = str(sids[i]) if sids is not None else f"idx_{batch_idx:04d}_{i:02d}"
                img_i = img[i, :1]          # (1,H,W)
                gt_i = gt[i].long()         # (H,W)
                pred_i = pred[i].long()     # (H,W)
                conf_i = conf[i]            # (1,H,W)

                pack = self._build_pack_multiclass(
                    img=img_i,
                    gt=gt_i,
                    pred=pred_i,
                    conf=conf_i,
                    sid=sid,
                    num_classes=C,
                    ignore_bg=ignore_bg,
                )

                if self.fixed_ids is not None:
                    if sid in self.fixed_ids and sid not in self._fixed_collected:
                        self._fixed_collected[sid] = pack
                else:
                    if len(self._fixed_collected) < self.fixed_first_n:
                        self._fixed_collected[sid] = pack

                self._maybe_push_worst(float(dice_cpu[i].item()), pack)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return

        writer = self._get_writer(trainer)
        if writer is None:
            return

        global_step = trainer.global_step

        fixed_packs = list(self._fixed_collected.values())
        if len(fixed_packs) > 0:
            overlay = torch.stack([p.overlay for p in fixed_packs], dim=0)  # (N,3,H,W)
            err = torch.stack([p.err for p in fixed_packs], dim=0)          # (N,3,H,W)
            prob = torch.stack([p.prob for p in fixed_packs], dim=0)        # (N,1,H,W)

            writer.add_images(f"{self.tag_prefix}/fixed_overlay", overlay, global_step=global_step)
            writer.add_images(f"{self.tag_prefix}/fixed_err", err, global_step=global_step)
            writer.add_images(f"{self.tag_prefix}/fixed_prob", prob, global_step=global_step)

            if any(p.focus_tpfpfn is not None for p in fixed_packs):
                focus = torch.stack([p.focus_tpfpfn if p.focus_tpfpfn is not None else p.err for p in fixed_packs], dim=0)
                writer.add_images(f"{self.tag_prefix}/fixed_focus_tpfpfn", focus, global_step=global_step)

        if self.topk > 0 and len(self._worst) > 0:
            worst = self._worst[: self.topk]
            overlay = torch.stack([p.overlay for _, p in worst], dim=0)
            err = torch.stack([p.err for _, p in worst], dim=0)
            prob = torch.stack([p.prob for _, p in worst], dim=0)

            text_lines = [f"{i+1}) {p.sid}: dice={d:.4f}" for i, (d, p) in enumerate(worst)]
            writer.add_text(f"{self.tag_prefix}/topk_worst_list", "\n".join(text_lines), global_step=global_step)

            writer.add_images(f"{self.tag_prefix}/topk_worst_overlay", overlay, global_step=global_step)
            writer.add_images(f"{self.tag_prefix}/topk_worst_err", err, global_step=global_step)
            writer.add_images(f"{self.tag_prefix}/topk_worst_prob", prob, global_step=global_step)

            if any(p.focus_tpfpfn is not None for _, p in worst):
                focus = torch.stack([p.focus_tpfpfn if p.focus_tpfpfn is not None else p.err for _, p in worst], dim=0)
                writer.add_images(f"{self.tag_prefix}/topk_worst_focus_tpfpfn", focus, global_step=global_step)
