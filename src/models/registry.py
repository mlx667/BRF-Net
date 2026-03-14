# src/models/registry.py
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    def deco(fn: Callable[..., nn.Module]):
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = fn
        return fn
    return deco

def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)

def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())

def get_active_model(cfg):
    mcfg = cfg.model
    models = mcfg.models
    active = mcfg.get("active_model", 0)

    if isinstance(active, int):
        if active >= len(models):
            available = list(range(len(models)))
            raise KeyError(f"Index out of range. active_model={active} but available indices: {available}")
        spec = models[active]
    else:
        try:
            spec = next(m for m in models if m.get("alias", m.name) == active)
        except StopIteration:
            available = [m.get("alias", m.name) for m in models]
            raise KeyError(f"Unknown active_model='{active}'. Available: {available}")

    if active is None:
        raise KeyError("cfg.model.active_model is required when cfg.model.models is provided.")

    return spec

