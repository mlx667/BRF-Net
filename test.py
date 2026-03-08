# export_per_sample_metrics.py  (FULL, runnable, Windows local, server-path remap, ONLY ACDC exact test-json filtering)
from __future__ import annotations

import os
import re
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F

# ====== PyTorch 2.6+ safe globals for omegaconf ckpt ======
import torch.serialization
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig
from omegaconf.base import ContainerMetadata

torch.serialization.add_safe_globals(
    [OmegaDictConfig, OmegaListConfig, DictConfig, ListConfig, ContainerMetadata]
)

import pytorch_lightning as pl

from src.data.datamodule import MultiDatasetDataModule
from src.models.registry import build_model, get_active_model
from src.lightning.seg_module import SegLitModule


# ======================================================================================
# 0) YOU ONLY EDIT HERE
# ======================================================================================
LOCAL_PROJECT_ROOT = r"D:\CV\Medical Image Segmentation"
MODELS_ROOT = Path(r"D:\CV\Medical Image Segmentation\model_version")
OUT_CSV = Path(r"D:\CV\Medical Image Segmentation\per_sample_metrics_acdc.csv")

SERVER_PROJECT_ROOT = "/root/Medical Image Segmentation"

DATASETS = [
    # "CVC-ClinicDB",
    # "ETIS-LaribPolypDB",
    # "USG",
    # "ISIC2017",
    # "ISIC2018",
    # "DSB18",
    # "Synapse",
    "ACDC",
]

MODELS = [
    #"BRFNet",         # Ours
    "res34_swin_ms_h2",
    "HiFormer",
    "MADGNet",
    "nnFormer",
    "nnUNet",
    "nnWNet",
    "PraNet",
    "PVT-EMCAD",
    "RollingUNet",
    "Swin-UNet",
    "TransUNet",
    "U-KAN",
    "UNETR",
]

THR_BIN = 0.5

# DataLoader workers（你的 DataModule 里 persistent_workers=True，因此 num_workers 必须 >0）
EVAL_NUM_WORKERS = 2
EVAL_BATCH_SIZE = 1

# 距离指标空集策略： "max" / "nan"
EMPTY_STRATEGY = "max"


# ======================================================================================
# 1) utilities
# ======================================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def safe_id(x: Any) -> str:
    s = str(x)
    s = s.strip().lstrip("\\/").replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".npy", "")
    return s

def apply_path_remap(cfg: DictConfig) -> DictConfig:
    """Remap any string path starting with SERVER_PROJECT_ROOT -> LOCAL_PROJECT_ROOT."""
    server = str(SERVER_PROJECT_ROOT).replace("\\", "/").rstrip("/")
    local = str(LOCAL_PROJECT_ROOT).replace("\\", "/").rstrip("/")

    def _remap_str(s: str) -> str:
        if not isinstance(s, str):
            return s
        s0 = s
        ss = s.replace("\\", "/")
        if ss == server or ss.startswith(server + "/"):
            tail = ss[len(server):]
            merged = local + tail
            return os.path.normpath(merged)
        return s0

    def _rec(obj: Any) -> Any:
        if isinstance(obj, str):
            return _remap_str(obj)
        if isinstance(obj, dict):
            return {k: _rec(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = [_rec(v) for v in obj]
            return type(obj)(out) if isinstance(obj, tuple) else out
        return obj

    raw = OmegaConf.to_container(cfg, resolve=False)
    remapped = _rec(raw)
    return OmegaConf.create(remapped)

def load_cfg_from_ckpt(ckpt_path: Path) -> DictConfig:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "cfg" in hp:
        cfg_obj = hp["cfg"]
        cfg = cfg_obj if isinstance(cfg_obj, DictConfig) else OmegaConf.create(cfg_obj)
    else:
        cfg = OmegaConf.create(hp)
    return apply_path_remap(cfg)

def warmup_registry() -> None:
    """Try import all possible model modules so registry has entries."""
    import importlib
    candidates = [
        "src.models.emcadnet",
        "src.models.hgnet",
        "src.models.hiformer",
        "src.models.mfmsnet",
        "src.models.nnformer",
        "src.models.nnunet_2d",
        "src.models.pranet",
        "src.models.pvtv2_gcascade_b2",
        "src.models.res34_swin_ms_h2",
        "src.models.res34_swin_ms",
        "src.models.rolling_unet_s",
        "src.models.rolling_unet_m",
        "src.models.rolling_unet_l",
        "src.models.swin_unet",
        "src.models.transunet",
        "src.models.u_rwkv_unet",
        "src.models.ukan",
        "src.models.unet_small",
        "src.models.unetr2d",
        "src.models.vmunet",
        "src.models.wnet2d",
    ]
    for m in candidates:
        try:
            importlib.import_module(m)
        except Exception:
            pass

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unwrap_logits(outputs: Any, prefer_largest_hw: bool = True) -> Optional[torch.Tensor]:
    if outputs is None:
        return None
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        for k in ["logits", "preds", "out", "y", "pred"]:
            v = outputs.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v
        return None
    if isinstance(outputs, (list, tuple)):
        ts = [t for t in outputs if isinstance(t, torch.Tensor)]
        if not ts:
            return None
        if prefer_largest_hw:
            cand = [t for t in ts if t.ndim == 4]
            if cand:
                cand.sort(key=lambda t: int(t.shape[-1] * t.shape[-2]), reverse=True)
                return cand[0]
        return ts[-1]
    return None

def tweak_cfg_for_eval(cfg: DictConfig, dataset: str) -> DictConfig:
    """Only tweak cfg in script: batch_size / num_workers / activate_dataset."""
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    try:
        cfg2.data.activate_dataset = dataset
    except Exception:
        pass
    try:
        cfg2.data.batch_size = int(EVAL_BATCH_SIZE)
    except Exception:
        pass
    try:
        cfg2.data.num_workers = int(EVAL_NUM_WORKERS)
    except Exception:
        pass
    try:
        cfg2.data.pin_memory = bool(torch.cuda.is_available())
    except Exception:
        pass
    return cfg2


# ======================================================================================
# 2) ONLY ACDC exact test-json filtering when split_json mismatch happens
# ======================================================================================
def _get_active_dataset_cfg(cfg: DictConfig) -> Tuple[str, Optional[DictConfig]]:
    act = str(getattr(cfg.data, "activate_dataset", ""))
    ds_cfg = None
    try:
        for ds in cfg.data.datasets:
            name = str(ds.name) if hasattr(ds, "name") else str(ds.get("name"))
            if name == act:
                ds_cfg = ds
                break
    except Exception:
        ds_cfg = None
    return act, ds_cfg

def _get_split_json_path(ds_cfg: Optional[DictConfig]) -> Optional[Path]:
    if ds_cfg is None:
        return None
    try:
        p = ds_cfg.split_json if hasattr(ds_cfg, "split_json") else ds_cfg.get("split_json", None)
    except Exception:
        p = None
    if not p:
        return None
    pp = Path(str(p))
    return pp if pp.exists() else None

def _load_split_json(p: Path) -> Optional[Dict[str, List[str]]]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        out = {}
        for k in ["train", "val", "test"]:
            v = obj.get(k, [])
            out[k] = [safe_id(x) for x in v] if isinstance(v, list) else []
        return out
    except Exception:
        return None

def build_acdc_test_filter_exact(cfg: DictConfig) -> Optional[Callable[[str], bool]]:
    """
    ONLY for ACDC:
      - read split_json['test']
      - exact match sid in test_set
    """
    act, ds_cfg = _get_active_dataset_cfg(cfg)
    if act != "ACDC":
        return None

    split_p = _get_split_json_path(ds_cfg)
    if split_p is None:
        return None

    split_obj = _load_split_json(split_p)
    if not split_obj or "test" not in split_obj:
        return None

    test_set = set(safe_id(x) for x in split_obj.get("test", []))

    if len(test_set) == 0:
        return None

    def _fn(sid: str) -> bool:
        return safe_id(sid) in test_set

    return _fn

def build_datamodule_with_possible_filter(cfg: DictConfig) -> Tuple[MultiDatasetDataModule, str, Optional[Callable[[str], bool]]]:
    """
    Normal: dm.setup("test") OK -> no filter.
    If split_json mismatch:
      - build all_test dataloader
      - ONLY if dataset == ACDC: apply exact test-json filter
      - for other datasets: no filter (as you requested)
    """
    dm = MultiDatasetDataModule(cfg)
    try:
        dm.setup("test")
        return dm, "split_json_or_cfg_split", None
    except ValueError as e:
        msg = str(e)
        if "split_json contains ids not found in dataset" not in msg:
            raise

        act, _ = _get_active_dataset_cfg(cfg)
        test_filter = build_acdc_test_filter_exact(cfg)  # ONLY ACDC exact

        print(f"[WARN] split_json mismatch for {act}. Using all_test dataloader. ACDC_test_filter={test_filter is not None}")

        # clone cfg and force all_test
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

        # disable split_json for active dataset (so datamodule can build)
        try:
            act2, ds_cfg2 = _get_active_dataset_cfg(cfg2)
            if ds_cfg2 is not None:
                if hasattr(ds_cfg2, "split_json"):
                    ds_cfg2.split_json = None
                else:
                    ds_cfg2["split_json"] = None
        except Exception:
            pass

        # force split all test
        try:
            if not hasattr(cfg2.data, "split") or cfg2.data.split is None:
                cfg2.data.split = OmegaConf.create({"train": 0.0, "val": 0.0, "test": 1.0})
            else:
                cfg2.data.split.train = 0.0
                cfg2.data.split.val = 0.0
                cfg2.data.split.test = 1.0
        except Exception:
            pass

        dm2 = MultiDatasetDataModule(cfg2)
        dm2.setup("test")
        return dm2, "all_test_fallback", test_filter


# ======================================================================================
# 3) ckpt scanning / matching
# ======================================================================================
def scan_all_ckpts(models_root: Path) -> List[Path]:
    ckpts = list(Path(models_root).rglob("*.ckpt"))
    print(f"[SCAN] MODELS_ROOT={models_root}  ckpt_files={len(ckpts)}")
    return ckpts

def find_best_ckpt(candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    best = [p for p in candidates if "best" in p.name.lower()]
    if best:
        candidates = best
    else:
        last = [p for p in candidates if "last" in p.name.lower()]
        if last:
            candidates = last
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

MODEL_ALIASES: Dict[str, List[str]] = {
    "BRFNet": ["brfnet", "brf-net", "brf_net", "ours", "res34_swin_ms"],
    "res34_swin_ms_h2": ["res34_swin_ms_h2", "h2former", "h2"],
    "HiFormer": ["hiformer"],
    "MADGNet": ["madgnet", "mfmsnet"],
    "nnFormer": ["nnformer"],
    "nnUNet": ["nnunet", "nnunet2d", "nnunet_2d", "nnunet-2d"],
    "nnWNet": ["wnet2d", "nnwnet"],
    "PraNet": ["pranet"],
    "PVT-EMCAD": ["emcad", "emcadnet", "pvt-emcad", "pvt_emcad"],
    "RollingUNet": ["rolling_unet_s", "rollingunet", "rolling-unet"],
    "Swin-UNet": ["swin_unet", "swinunet"],
    "TransUNet": ["transunet", "trans_unet", "trans-unet"],
    "U-KAN": ["ukan", "u-kan", "u_kan"],
    "UNETR": ["unetr", "unetr2d"],
}

DATASET_ALIASES: Dict[str, List[str]] = {
    "CVC-ClinicDB": ["cvc-clinicdb", "cvcclinicdb", "clinicdb"],
    "ETIS-LaribPolypDB": ["etis", "etis-laribpolypdb", "larib"],
    "USG": ["usg"],
    "ISIC2017": ["isic2017"],
    "ISIC2018": ["isic2018"],
    "DSB18": ["dsb18", "dsb2018", "dsb2018cell", "cell"],
    "Synapse": ["synapse"],
    "ACDC": ["acdc"],
}

def match_path_for(ds: str, model: str, ckpt_path: Path) -> bool:
    s = norm_key(str(ckpt_path))
    ds_hits = any(norm_key(a) in s for a in DATASET_ALIASES.get(ds, [ds]))
    m_hits = any(norm_key(a) in s for a in MODEL_ALIASES.get(model, [model]))
    return ds_hits and m_hits

def find_ckpt_for(ds: str, model: str, ckpts_all: List[Path]) -> Optional[Path]:
    matched = [p for p in ckpts_all if match_path_for(ds, model, p)]
    return find_best_ckpt(matched)


# ======================================================================================
# 4) metrics
# ======================================================================================
def _binary_confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    return tp, fp, fn, tn

def dice_iou_precision_recall_binary(pred01: np.ndarray, gt01: np.ndarray) -> Tuple[float, float, float, float]:
    tp, fp, fn, tn = _binary_confusion(pred01, gt01)
    pred_sum = tp + fp
    gt_sum = tp + fn

    denom = pred_sum + gt_sum
    dice = 1.0 if denom == 0 else (2.0 * tp + 1e-6) / (denom + 1e-6)

    union = pred_sum + gt_sum - tp
    iou = 1.0 if union == 0 else (tp + 1e-6) / (union + 1e-6)

    if pred_sum == 0:
        precision = 1.0 if gt_sum == 0 else 0.0
    else:
        precision = (tp + 1e-6) / (pred_sum + 1e-6)

    if gt_sum == 0:
        recall = 1.0 if pred_sum == 0 else 0.0
    else:
        recall = (tp + 1e-6) / (gt_sum + 1e-6)

    return float(dice), float(iou), float(precision), float(recall)

def _surface_distances_binary(pred: np.ndarray, gt: np.ndarray) -> Optional[np.ndarray]:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except Exception:
        return None

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float32)

    if pred.any() and not gt.any():
        return None
    if gt.any() and not pred.any():
        return None

    pred_surf = np.logical_xor(pred, binary_erosion(pred))
    gt_surf = np.logical_xor(gt, binary_erosion(gt))

    if not pred_surf.any() or not gt_surf.any():
        return None

    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    d1 = dt_gt[pred_surf]
    d2 = dt_pred[gt_surf]
    return np.concatenate([d1, d2]).astype(np.float32)

def hd95_asd_binary(pred01: np.ndarray, gt01: np.ndarray) -> Tuple[float, float]:
    H, W = pred01.shape
    max_dist = float(np.sqrt(H * H + W * W))

    d = _surface_distances_binary(pred01, gt01)
    if d is None:
        if EMPTY_STRATEGY == "nan":
            return float("nan"), float("nan")
        return max_dist, max_dist

    asd = float(np.mean(d))
    hd95 = float(np.percentile(d, 95))
    return hd95, asd

def dice_iou_multiclass(pred_lab: np.ndarray, gt_lab: np.ndarray, num_classes: int, ignore_bg: bool = True) -> Dict[str, float]:
    out: Dict[str, float] = {}
    classes = list(range(1, num_classes)) if ignore_bg else list(range(num_classes))

    dices, ious, dices_ie, ious_ie = [], [], [], []
    for c in classes:
        p = (pred_lab == c)
        g = (gt_lab == c)

        inter = int(np.logical_and(p, g).sum())
        ps = int(p.sum())
        gs = int(g.sum())

        denom = ps + gs
        dice = 1.0 if denom == 0 else (2.0 * inter + 1e-6) / (denom + 1e-6)

        union = ps + gs - inter
        iou = 1.0 if union == 0 else (inter + 1e-6) / (union + 1e-6)

        out[f"dice_c{c}"] = float(dice)
        out[f"iou_c{c}"] = float(iou)

        dices.append(float(dice))
        ious.append(float(iou))
        if ps > 0 or gs > 0:
            dices_ie.append(float(dice))
            ious_ie.append(float(iou))

    out["dice_macro"] = float(np.mean(dices)) if len(dices) else 1.0
    out["iou_macro"] = float(np.mean(ious)) if len(ious) else 1.0
    out["dice_macro_ignore_empty"] = float(np.mean(dices_ie)) if len(dices_ie) else 1.0
    out["iou_macro_ignore_empty"] = float(np.mean(ious_ie)) if len(ious_ie) else 1.0
    return out


# ======================================================================================
# 5) evaluation
# ======================================================================================
@dataclass
class RunPack:
    dataset: str
    model_tag: str
    ckpt: Path
    cfg: DictConfig
    arch_name: str
    lit: SegLitModule
    split_mode: str
    test_filter: Optional[Callable[[str], bool]]  # ONLY ACDC exact test ids

def load_runpack(dataset: str, model_tag: str, ckpt: Path, device: torch.device) -> Tuple[RunPack, MultiDatasetDataModule]:
    cfg = load_cfg_from_ckpt(ckpt)
    cfg = tweak_cfg_for_eval(cfg, dataset)

    pl.seed_everything(int(getattr(cfg, "seed", 3407)), workers=True)

    warmup_registry()
    active_model_cfg = get_active_model(cfg)
    arch_name = str(active_model_cfg["name"])
    model_arch = build_model(active_model_cfg["name"], **active_model_cfg.get("kwargs", {}))

    lit = SegLitModule.load_from_checkpoint(
        checkpoint_path=str(ckpt),
        model=model_arch,
        cfg=cfg,
        weights_only=False,
        strict=False,
    )
    lit.to(device)
    lit.eval()

    dm, split_mode, test_filter = build_datamodule_with_possible_filter(cfg)

    rp = RunPack(
        dataset=dataset,
        model_tag=model_tag,
        ckpt=ckpt,
        cfg=cfg,
        arch_name=arch_name,
        lit=lit,
        split_mode=split_mode,
        test_filter=test_filter,
    )
    return rp, dm

@torch.no_grad()
def evaluate_one(rp: RunPack, dm: MultiDatasetDataModule, device: torch.device, rows_out: List[Dict[str, Any]]) -> None:
    print(f"\n[EVAL] dataset={rp.dataset} model={rp.model_tag} arch={rp.arch_name}")
    print(f"       ckpt={rp.ckpt}")
    print(f"       split_mode={rp.split_mode}  ACDC_test_filter={rp.test_filter is not None}")

    dl = dm.test_dataloader()
    for batch in dl:
        if not isinstance(batch, dict):
            continue

        imgs = batch.get("image", batch.get("images", None))
        masks = batch.get("mask", batch.get("masks", None))
        ids = batch.get("id", batch.get("ids", batch.get("filenames", batch.get("filename", None))))

        if imgs is None or masks is None:
            continue

        imgs = imgs.to(device, non_blocking=True)
        masks_t = masks.to(device, non_blocking=True)

        out = rp.lit.model(imgs)
        logits = unwrap_logits(out)
        if logits is None or logits.ndim != 4:
            continue

        gt_h, gt_w = int(masks_t.shape[-2]), int(masks_t.shape[-1])
        if int(logits.shape[-2]) != gt_h or int(logits.shape[-1]) != gt_w:
            logits = F.interpolate(logits, size=(gt_h, gt_w), mode="bilinear", align_corners=False)

        B, C, H, W = logits.shape

        for i in range(B):
            sid = safe_id(ids[i] if ids is not None else f"b{len(rows_out)}")

            # ONLY ACDC: exact test list filtering
            if rp.test_filter is not None and (not rp.test_filter(sid)):
                continue

            row: Dict[str, Any] = {
                "dataset": rp.dataset,
                "model": rp.model_tag,
                "arch": rp.arch_name,
                "ckpt": str(rp.ckpt),
                "split_mode": rp.split_mode,
                "id": sid,
                "H": H,
                "W": W,
                "num_classes": C,
            }

            if C <= 2:
                if C == 1:
                    prob = torch.sigmoid(logits[i, 0])
                    pred01 = (prob > THR_BIN)
                else:
                    prob = torch.softmax(logits[i:i+1], dim=1)[0, 1]
                    pred01 = (prob > THR_BIN)

                gt = masks_t[i]
                if gt.ndim == 3 and gt.shape[0] == 1:
                    gt01 = (gt[0] > 0.5)
                elif gt.ndim == 2:
                    gt01 = (gt > 0.5)
                else:
                    gt01 = (gt.squeeze() > 0.5)

                pred_np = pred01.detach().cpu().numpy().astype(np.uint8)
                gt_np = gt01.detach().cpu().numpy().astype(np.uint8)

                dice, iou, prec, rec = dice_iou_precision_recall_binary(pred_np, gt_np)
                hd95, asd = hd95_asd_binary(pred_np, gt_np)

                row.update({
                    "task": "binary",
                    "dice": dice,
                    "iou": iou,
                    "precision": prec,
                    "recall": rec,
                    "hd95": hd95,
                    "asd": asd,
                    "pred_fg": int(pred_np.sum()),
                    "gt_fg": int(gt_np.sum()),
                })
            else:
                pred_lab = torch.argmax(logits[i], dim=0).detach().cpu().numpy().astype(np.int32)
                gt = masks_t[i]
                if gt.ndim == 3 and gt.shape[0] == 1:
                    gt_lab = gt[0].detach().cpu().numpy().astype(np.int32)
                else:
                    gt_lab = gt.detach().cpu().numpy().astype(np.int32)

                m = dice_iou_multiclass(pred_lab, gt_lab, num_classes=C, ignore_bg=True)
                row.update({
                    "task": "multiclass",
                    "pred_fg": int((pred_lab > 0).sum()),
                    "gt_fg": int((gt_lab > 0).sum()),
                })
                row.update(m)

            rows_out.append(row)

def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)

    base_cols = [
        "dataset", "model", "arch", "ckpt", "split_mode", "id",
        "task", "num_classes", "H", "W",
        "pred_fg", "gt_fg",
        "dice", "iou", "precision", "recall", "hd95", "asd",
        "dice_macro", "iou_macro", "dice_macro_ignore_empty", "iou_macro_ignore_empty",
    ]
    class_cols = []
    for c in range(1, 20):
        class_cols.append(f"dice_c{c}")
        class_cols.append(f"iou_c{c}")

    fieldnames = base_cols + class_cols

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k, "") for k in fieldnames}
            w.writerow(rr)

    print(f"\n[CSV] saved -> {out_csv}  rows={len(rows)}")


# ======================================================================================
# 6) main
# ======================================================================================
def main():
    device = get_device()
    print(f"[DEVICE] {device}")

    ckpts_all = scan_all_ckpts(MODELS_ROOT)
    all_rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        for model in MODELS:
            ckpt = find_ckpt_for(ds, model, ckpts_all)
            if ckpt is None:
                print(f"[SKIP] no ckpt for dataset={ds} model={model}")
                continue

            rp = None
            dm = None
            try:
                rp, dm = load_runpack(ds, model, ckpt, device)
                evaluate_one(rp, dm, device, all_rows)
            except Exception as e:
                print(f"[FAIL] dataset={ds} model={model} -> {type(e).__name__}: {e}")
            finally:
                try:
                    del rp
                    del dm
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_csv(all_rows, OUT_CSV)

if __name__ == "__main__":
    main()