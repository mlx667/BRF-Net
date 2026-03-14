# src/data/datasets.py
from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from ..utils.misc import list_image_files, stem


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")

def _normalize_mask_stem(s: str) -> str:
    for suf in ("_segmentation", "_tumor", "_mask", "-segmentation", "-tumor", "-mask"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s



import re

def _mask_base_id(mask_stem: str) -> str:

    s = mask_stem.lower()
    s = re.sub(r"\s*\(\d+\)$", "", s)  # 去掉 " (1)" 这种
    s = re.sub(r"([_-](segmentation|mask|tumor|anno))$", "", s)
    s = re.sub(r"([_-]other\d+)$", "", s)
    return s



def _load_2d(path: str, image_channels: int) -> np.ndarray:
    """Load image or mask as 2D numpy array.

    - For image: returns float32 array (H, W) or (H, W, C)
    - For mask : returns float32 array (H, W)
    """
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            # (C,H,W) -> (H,W,C)
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)

    img = Image.open(path)
    arr = np.array(img)
    return arr.astype(np.float32)


def _ensure_channels(arr: np.ndarray, image_channels: int) -> np.ndarray:
    if arr.ndim == 2:
        if image_channels == 1:
            return arr[..., None]  # (H,W,1)
        return np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3:
        if arr.shape[-1] == image_channels:
            return arr
        if image_channels == 1:
            return arr.mean(axis=-1, keepdims=True)
        if arr.shape[-1] == 1 and image_channels == 3:
            return np.repeat(arr, 3, axis=-1)
    raise ValueError(f"Unsupported image shape {arr.shape} for image_channels={image_channels}")


@dataclass
class SamplePaths:
    image_path: str
    mask_paths: Sequence[str]



class ImageMaskDataset(Dataset):
    """Generic 2D segmentation dataset supporting binary and multiclass label maps.

    Pairing rule: match by basename *stem* (without extension).

    Modes:
      - task="binary": returns mask float tensor (1,H,W) in {0,1}
      - task="multiclass": returns mask long tensor (H,W) with values in [0, num_classes-1]
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_channels: int = 1,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_threshold: float = 0.5,
        task: str = "binary",
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_channels = int(image_channels)
        self.transform = transform
        self.mask_threshold = float(mask_threshold)

        self.task = str(task).lower()
        if self.task not in ("binary", "multiclass"):
            raise ValueError(f"Unknown task='{task}'. Use 'binary' or 'multiclass'.")
        self.num_classes = int(num_classes) if num_classes is not None else None
        if self.task == "multiclass" and (self.num_classes is None or self.num_classes < 2):
            raise ValueError("For task='multiclass', num_classes must be provided and >= 2.")

        img_files = list_image_files(images_dir, IMG_EXTS)
        msk_files = list_image_files(masks_dir, IMG_EXTS)

        img_map = {stem(p).lower(): p for p in img_files}

        msk_map: dict[str, List[str]] = {}
        for p in msk_files:
            s = _mask_base_id(stem(p))
            msk_map.setdefault(s, []).append(p)

        keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
        print(f"[ImageMaskDataset] images={len(img_map)} masks={len(msk_files)} matched={len(keys)}")

        if len(keys) == 0:
            raise RuntimeError(
                f"No matched image-mask pairs found. images_dir={images_dir}, masks_dir={masks_dir}\n"
                "Hint: make sure basenames match, e.g. case001.png <-> case001.png"
            )

        self.samples = [SamplePaths(img_map[k], msk_map[k]) for k in keys]

    def __len__(self) -> int:
        return len(self.samples)

    def _postprocess_mask_binary(self, msk: np.ndarray) -> torch.Tensor:
        # msk: (H,W) or (H,W,1/3)
        m = msk
        if m.ndim == 3:
            m = m[..., 0]

        mf = m.astype(np.float32)
        mx = float(mf.max()) if mf.size else 0.0

        u = np.unique(mf[::8, ::8])
        u_n = int(u.size)

        if mx <= 20 and u_n <= 64:
            bin_m = (mf > 0).astype(np.float32)
            return torch.from_numpy(bin_m[None, ...]).float()  # (1,H,W)

        if mx > 1.5:
            mf = mf / 255.0

        msk_t = torch.from_numpy(mf[None, ...]).float()
        msk_t = (msk_t >= self.mask_threshold).float()
        return msk_t

    def _postprocess_mask_multiclass(self, msk: np.ndarray) -> torch.Tensor:
        C = self.num_classes  # type: ignore
        m = msk
        if m.dtype != np.int32 and m.dtype != np.int64:
            # float 或 uint8 都先转 float 方便判断
            m = m.astype(np.float32)

        mmax = float(m.max()) if m.size > 0 else 0.0
        if mmax <= (C - 1) + 1e-6:
            lab = np.rint(m).astype(np.int64)
        else:
            lab = np.rint((m / 255.0) * (C - 1)).astype(np.int64)

        lab = np.clip(lab, 0, C - 1).astype(np.int64)
        return torch.from_numpy(lab).long()  # (H,W)

    def __getitem__(self, idx: int) -> dict:
        sp = self.samples[idx]
        img = _load_2d(sp.image_path, self.image_channels)

        msks = []
        for mp in sp.mask_paths:
            m = _load_2d(mp, 1)
            if m.ndim == 3:
                m = m[..., 0]
            msks.append(m)

        msk = msks[0] if len(msks) == 1 else np.maximum.reduce(msks)

        img = _ensure_channels(img, self.image_channels)  # (H,W,C)

        if img.max() > 1.5:
            img = img / 255.0

        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        if self.task == "binary":
            msk_t = self._postprocess_mask_binary(msk)
        else:
            msk_t = self._postprocess_mask_multiclass(msk)

        if self.transform is not None:
            img_t, msk_t = self.transform(img_t, msk_t)

        return dict(
            image=img_t,
            mask=msk_t,
            id=os.path.splitext(os.path.basename(sp.image_path))[0],
        )

