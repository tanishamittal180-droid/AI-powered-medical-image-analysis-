"""Pretrained chest X-ray classifier wrapper.

This is an educational binary classifier (NORMAL vs PNEUMONIA), not a diagnostic tool.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Public community model card and weights: https://huggingface.co/nickmuchi/vit-finetuned-chest-xray-pneumonia
MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, torch.device]:
    """Load once per app process; Transformers caches downloaded files locally."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device


def _clean_label(raw: str) -> str:
    normalized = str(raw).strip().upper().replace("_", " ").replace("-", " ")
    if "PNEUMONIA" in normalized:
        return "PNEUMONIA"
    if "NORMAL" in normalized:
        return "NORMAL"
    return normalized or "UNKNOWN"


def predict_xray(image: Image.Image) -> dict[str, Any]:
    """Return the model's class scores for a PIL image.

    Scores are softmax outputs and should not be interpreted as calibrated clinical probabilities.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("predict_xray expects a PIL.Image.Image object")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if rgb.width < 32 or rgb.height < 32:
        raise ValueError("Image is too small. Upload a larger chest X-ray image.")

    processor, model, device = _load_model()
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    id2label = getattr(model.config, "id2label", {}) or {}
    scores: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        raw_label = id2label.get(index, id2label.get(str(index), f"CLASS_{index}"))
        label = _clean_label(raw_label)
        # If a model has duplicate label aliases, preserve a unique key.
        if label in scores:
            label = f"{label}_{index}"
        scores[label] = float(probability)

    if not scores:
        raise RuntimeError("The model returned no class scores.")
    best_label = max(scores, key=scores.get)
    if best_label not in {"NORMAL", "PNEUMONIA"}:
        # Do not silently pretend an unknown model label is a supported diagnosis.
        raise RuntimeError(f"Unexpected model class label: {best_label}. Check the selected model card.")
    return {"label": best_label, "score": scores[best_label], "scores": scores, "model_id": MODEL_ID}
