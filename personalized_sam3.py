import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_TRANSFORMERS = False


@dataclass
class CaptionConfig:
    """Configuration for the caption model that describes a masked region."""

    # Default to BLIP base for lower memory footprint; override via CLI if needed.
    model_id: str = "Salesforce/blip-image-captioning-base"
    max_new_tokens: int = 40
    max_words: int = 10  # cap caption length (words) to keep it concise


@dataclass
class Sam3Config:
    """Configuration for loading the SAM3 (or compatible) checkpoint."""

    checkpoint_path: str = "weights/sam3_default.pt"
    checkpoint_url: Optional[str] = None  # Provide a URL to auto-download if the file is missing.


def _download_if_missing(target_path: str, url: Optional[str]) -> str:
    """Download a file when it does not exist locally."""
    if os.path.exists(target_path):
        return target_path
    if not url:
        raise RuntimeError(
            f"Checkpoint missing at {target_path}. Provide `sam3_checkpoint_url` to auto-download."
        )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.hub.download_url_to_file(url, target_path, progress=True)
    return target_path


def _load_mask(mask_path: str, threshold: float = 0.5) -> np.ndarray:
    """Load a mask file (PNG) and return a binary numpy array."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if mask.max() > 1:
        mask = mask / 255.0
    return (mask >= threshold).astype(np.uint8)


def _load_image_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _mask_to_points(mask: np.ndarray, num_points: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a binary mask to positive point prompts (x, y)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        xs = np.array([w // 2])
        ys = np.array([h // 2])
    coords = np.stack([xs, ys], axis=1)
    if len(coords) > num_points:
        indices = np.linspace(0, len(coords) - 1, num_points, dtype=int)
        coords = coords[indices]
    labels = np.ones(len(coords), dtype=np.int32)
    return coords.astype(np.float32), labels


class PersonalizedSAM3:
    """
    Pipeline that extracts a text description from an image + mask, generates
    positive points from the same mask, and optionally runs a SAM3 predictor.

    The SAM3 predictor is pluggable: pass a builder that accepts a checkpoint path
    and returns an object exposing `predict(image, text_prompt, point_coords, point_labels)`.
    """

    def __init__(
        self,
        caption_cfg: CaptionConfig = CaptionConfig(),
        sam3_cfg: Sam3Config = Sam3Config(),
        sam3_builder: Optional[Callable[[str], Any]] = None,
        device: Optional[str] = None,
        num_points: int = 3,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        self.caption_cfg = caption_cfg
        self._fallback_caption_model = None
        self._fallback_caption_processor = None

        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for captioning. Install it via `pip install transformers huggingface-hub`."
            )

        self.caption_model_name = caption_cfg.model_id.lower()
        # Load BLIP2 / BLIP (stable captioners)
        self.caption_processor = AutoProcessor.from_pretrained(caption_cfg.model_id)
        if "blip2" in self.caption_model_name:
            self.caption_model = (
                Blip2ForConditionalGeneration.from_pretrained(caption_cfg.model_id)
                .to(self.device)
                .eval()
            )
        else:
            self.caption_model = (
                BlipForConditionalGeneration.from_pretrained(caption_cfg.model_id)
                .to(self.device)
                .eval()
            )

        self.sam3_predictor = None
        if sam3_builder is not None:
            checkpoint_path = _download_if_missing(
                sam3_cfg.checkpoint_path, sam3_cfg.checkpoint_url
            )
            self.sam3_predictor = sam3_builder(checkpoint_path)

    def _caption_masked_region(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Describe the masked region using the caption model."""
        mask = mask.astype(bool)
        if mask.any():
            ys, xs = np.nonzero(mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            # No padding; keep the tightest box around the mask
            crop = image[y_min : y_max + 1, x_min : x_max + 1]
            crop_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
            crop_fg = crop.copy()
            crop_fg[~crop_mask] = 0  # zero background to avoid environment leakage
            roi = crop_fg
        else:
            roi = image

        pil_image = Image.fromarray(roi)
        inputs = self.caption_processor(
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.caption_model.generate(
                **inputs,
                max_new_tokens=self.caption_cfg.max_new_tokens,
                num_beams=3,
            )
        caption = self.caption_processor.decode(
            generated_ids[0], skip_special_tokens=True
        )
        caption = caption.strip()
        # Keep a concise object name: truncate to a few words and strip punctuation.
        max_w = max(1, self.caption_cfg.max_words)
        tokens = [w.strip(" ,.;:") for w in caption.split()]
        short = " ".join(tokens[:max_w]).strip()
        return short if short else caption

    def run(
        self, image_path: str, mask_path: str, run_sam3: bool = True
    ) -> Dict[str, Any]:
        """
        Load image + mask, caption the masked object, extract point prompts,
        and (optionally) run SAM3.
        """
        image = _load_image_rgb(image_path)
        try:
            mask = _load_mask(mask_path)
        except Exception as e:
            # If the mask is missing or unreadable, fall back to full-image captioning
            print(f"[caption] Mask load failed ({e}); using full image for caption.")
            mask = np.ones(image.shape[:2], dtype=np.uint8)


        # Caption with primary model; if it fails, fall back to BLIP2 on CPU (not folder names).
        try:
            text_prompt = self._caption_masked_region(image, mask)
        except Exception as e:
            print(f"[caption] Primary model failed ({e}); falling back to BLIP2 on CPU.")
            if self._fallback_caption_model is None:
                fallback_id = "Salesforce/blip2-opt-2.7b-coco"
                self.caption_model_name = fallback_id.lower()
                self._fallback_caption_processor = AutoProcessor.from_pretrained(fallback_id)
                self._fallback_caption_model = (
                    Blip2ForConditionalGeneration.from_pretrained(fallback_id)
                    .to("cpu")
                    .eval()
                )
            mask_cpu = mask.astype(bool)
            if mask_cpu.any():
                ys, xs = np.nonzero(mask_cpu)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                crop = image[y_min : y_max + 1, x_min : x_max + 1]
                crop_mask = mask_cpu[y_min : y_max + 1, x_min : x_max + 1]
                crop_fg = crop.copy()
                crop_fg[~crop_mask] = 0
                roi = crop_fg
            else:
                roi = image
            pil_image = Image.fromarray(roi)
            inputs = self._fallback_caption_processor(images=pil_image, return_tensors="pt").to("cpu")
            with torch.no_grad():
                generated_ids = self._fallback_caption_model.generate(
                    **inputs,
                    max_new_tokens=self.caption_cfg.max_new_tokens,
                    num_beams=3,
                )
            text_prompt = self._fallback_caption_processor.decode(
                generated_ids[0], skip_special_tokens=True
            ).strip()

        point_coords, point_labels = _mask_to_points(mask, num_points=self.num_points)

        result: Dict[str, Any] = {
            "text_prompt": text_prompt,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        if run_sam3 and self.sam3_predictor is not None:
            sam3_output = self.sam3_predictor.predict(
                image=image,
                text_prompt=text_prompt,
                point_coords=point_coords,
                point_labels=point_labels,
            )
            result["sam3_output"] = sam3_output

        return result


__all__ = ["CaptionConfig", "Sam3Config", "PersonalizedSAM3"]


def _build_arg_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Caption a masked region and emit positive points (SAM3 optional)."
    )
    parser.add_argument("--image", required=True, help="Path to the RGB image.")
    parser.add_argument("--mask", required=True, help="Path to the binary mask (png).")
    parser.add_argument(
        "--num-points", type=int, default=3, help="Number of positive points to sample from the mask."
    )
    parser.add_argument(
        "--sam3-ckpt",
        type=str,
        default="weights/sam3_default.pt",
        help="Local SAM3 checkpoint path (only used if you wire a builder).",
    )
    parser.add_argument(
        "--sam3-url",
        type=str,
        default=None,
        help="Optional URL to auto-download the SAM3 checkpoint if missing.",
    )
    parser.add_argument(
        "--run-sam3",
        action="store_true",
        help="Attempt to run SAM3; requires you to edit the file and provide a real builder.",
    )
    return parser


def _cli() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    def _placeholder_builder(_ckpt: str) -> Any:
        raise NotImplementedError("Provide a SAM3 builder that returns a predictor with `.predict`.")

    runner = PersonalizedSAM3(
        sam3_builder=None if not args.run_sam3 else _placeholder_builder,
        sam3_cfg=Sam3Config(checkpoint_path=args.sam3_ckpt, checkpoint_url=args.sam3_url),
        num_points=args.num_points,
    )
    result = runner.run(args.image, args.mask, run_sam3=args.run_sam3)
    print("Caption:", result["text_prompt"])
    print("Points:", result["point_coords"])
    print("Labels:", result["point_labels"])
    if args.run_sam3 and "sam3_output" in result:
        print("SAM3 output keys:", list(result["sam3_output"].keys()))


if __name__ == "__main__":
    _cli()
