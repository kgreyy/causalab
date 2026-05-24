"""Loaders for pipelines and prior analysis outputs.

These functions read from disk (or hydrate objects that then do). They live
in ``io/`` so that ``runner/`` and ``analyses/`` can consume them without
pulling in orchestration.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from causalab.tasks.loader import Task

logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _build_quantization_config(quantization: str | None):
    """Build a BitsAndBytesConfig from a short string: 'int4' or 'int8'."""
    if quantization is None:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for quantization. Install it with: "
            "pip install bitsandbytes"
        ) from exc
    if quantization == "int4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(
        f"Unknown quantization {quantization!r}. Supported: 'int4', 'int8'."
    )


def load_pipeline(
    model_name: str,
    task: "Task",
    max_new_tokens: int,
    device: str | None = None,
    dtype: str | None = None,
    eager_attn: bool | None = None,
    quantization: str | None = None,
):
    """Load an ``LMPipeline`` from explicit parameters.

    Pass ``quantization='int4'`` or ``quantization='int8'`` to load a
    BitsAndBytes-quantized model. Requires ``bitsandbytes`` to be installed.
    Quantized models use ``device_map='auto'`` and ignore the ``dtype`` kwarg.
    """
    from causalab.neural.pipeline import LMPipeline, resolve_device

    quantization_config = _build_quantization_config(quantization)

    # Quantized models require device_map (handled inside LMPipeline._setup_model).
    # For unquantized multi-GPU setups, use device_map="auto" for sharding.
    use_device_map = quantization_config is not None or (
        (device is None or device == "auto") and torch.cuda.device_count() > 1
    )

    model_kwargs: dict[str, Any]
    if use_device_map:
        model_kwargs = {"device_map": "auto"}
        logger.info(
            "Loading model: %s (device_map=auto, %d GPUs, quantization=%s)",
            model_name,
            torch.cuda.device_count(),
            quantization or "none",
        )
    else:
        resolved_device = resolve_device(device)
        model_kwargs = {"device": resolved_device}
        logger.info("Loading model: %s (device=%s)", model_name, resolved_device)

    if dtype and quantization_config is None:
        model_kwargs["dtype"] = DTYPE_MAP.get(dtype, torch.bfloat16)
    if eager_attn is False:
        model_kwargs["eager_attn"] = False
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    pipeline = LMPipeline(model_name, max_new_tokens=max_new_tokens, **model_kwargs)

    if task.validate is not None:
        task.validate(pipeline)

    return pipeline


def load_lite_pipeline(
    model_name: str,
    max_new_tokens: int = 3,
):
    """Load an ``LMPipeline`` with tokenizer + config only (no model weights).

    For analyses that need ``pipeline.tokenizer`` and ``pipeline.model.config``
    (e.g. to build :class:`InterchangeTarget` from cached features) but never
    run forward passes. Returns immediately for any model size; calling
    ``pipeline.generate`` or running the model will fail.
    """
    from causalab.neural.pipeline import LMPipeline

    logger.info("Loading lite pipeline (tokenizer+config only): %s", model_name)
    return LMPipeline(model_name, max_new_tokens=max_new_tokens, load_weights=False)


# --------------------------------------------------------------------------- #
# Discovery helpers for prior analysis runs                                   #
# --------------------------------------------------------------------------- #


def find_subspace_dirs(root: str) -> list[str]:
    """Scan ``root/subspace/`` for completed subspace runs."""
    subspace_root = os.path.join(root, "subspace")
    if not os.path.isdir(subspace_root):
        return []
    return sorted(
        d
        for d in os.listdir(subspace_root)
        if os.path.isdir(os.path.join(subspace_root, d))
    )


def find_activation_manifold_dirs(root: str, subspace_sub: str) -> list[str]:
    """Scan ``root/activation_manifold/{subspace_sub}/`` for completed manifold runs.

    Returns relative paths from ``activation_manifold/{subspace_sub}/``.
    Handles both single-cell layout (``spline_s0.0/``) and grid-cell layout
    (``L{layer}_{pos}/spline_s0.0/``).
    """
    manifold_root = os.path.join(root, "activation_manifold", subspace_sub)
    if not os.path.isdir(manifold_root):
        return []

    found: list[str] = []
    for d in sorted(os.listdir(manifold_root)):
        d_path = os.path.join(manifold_root, d)
        if not os.path.isdir(d_path):
            continue
        # Direct method dir (single-cell layout)
        if os.path.isfile(os.path.join(d_path, "metadata.json")):
            found.append(d)
            continue
        # Grid-cell layout: L{layer}_{pos}/method_dir/
        # metadata.json may be directly in method_dir/ or one level deeper
        # (e.g. method_dir/target_variable/) when target_variable nesting is used.
        for sub in sorted(os.listdir(d_path)):
            sub_path = os.path.join(d_path, sub)
            if not os.path.isdir(sub_path):
                continue
            if os.path.isfile(os.path.join(sub_path, "metadata.json")):
                found.append(os.path.join(d, sub))
            else:
                # Check one level deeper (target_variable subdirectory)
                for tv_sub in sorted(os.listdir(sub_path)):
                    tv_path = os.path.join(sub_path, tv_sub)
                    if os.path.isdir(tv_path) and os.path.isfile(
                        os.path.join(tv_path, "metadata.json")
                    ):
                        found.append(os.path.join(d, sub))
                        break
    return found


def load_locate_result(root: str) -> dict:
    """Read best layer/position from ``locate/`` analysis output.

    Prefers ``results.json``; falls back to ``metadata.json`` for
    backwards compatibility with older runs.
    """
    locate_root = os.path.join(root, "locate")
    if not os.path.isdir(locate_root):
        return {}

    for method_dir in sorted(os.listdir(locate_root)):
        dir_path = os.path.join(locate_root, method_dir)
        results_path = os.path.join(dir_path, "results.json")
        if os.path.isfile(results_path):
            with open(results_path) as f:
                return json.load(f)
        meta_path = os.path.join(dir_path, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                return json.load(f)
    return {}


def load_subspace_metadata(
    root: str,
    subspace_sub: str,
    target_variable: str | None = None,
) -> dict:
    """Read metadata from a subspace run directory."""
    parts = [root, "subspace", subspace_sub]
    if target_variable:
        parts.append(target_variable)
    meta_path = os.path.join(*parts, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def load_activation_manifold_metadata(
    root: str,
    subspace_sub: str,
    manifold_sub: str,
    target_variable: str | None = None,
) -> dict:
    """Read metadata from an activation_manifold run directory."""
    parts = [root, "activation_manifold", subspace_sub, manifold_sub]
    if target_variable:
        parts.append(target_variable)
    meta_path = os.path.join(*parts, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)
