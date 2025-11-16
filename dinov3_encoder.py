import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class DinoV3FeatureExtractor:
    """
    Light wrapper around a ViT model (loaded via timm) that mimics DinoV3-style
    dense feature extraction for similarity computation only.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch14_dinov2.lvd142m",
        image_size: int = 518,
        output_size: Optional[int] = 64,
        device: Optional[torch.device] = None,
        precision: torch.dtype = torch.float32,
        pretrained: bool = True,
    ) -> None:
        try:
            import timm  # pylint: disable=import-error
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "timm is required for DinoV3FeatureExtractor. "
                "Install it with `pip install timm`."
            ) from exc

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.image_size = image_size
        self.output_size = output_size
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        ).to(self.device, dtype=self.precision)
        self.model.eval()

        # Standard ImageNet normalization (DINOv3 follows the same stats).
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.precision)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.precision)
        self.registered_mean = mean.view(1, 3, 1, 1)
        self.registered_std = std.view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Args:
            image: numpy array in H x W x 3 (RGB).
            override_image_size: optional different resize for this call.
        Returns:
            torch.Tensor of shape (C, H_feat, W_feat) on the extractor's device.
        """
        tensor = torch.from_numpy(image).to(self.device)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        tensor = tensor.permute(2, 0, 1)[None, ...] / 255.0
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        tensor = (tensor - self.registered_mean) / self.registered_std

        feats = self.model.forward_features(tensor)
        if isinstance(feats, dict):
            tokens = feats.get("x_norm_patchtokens") or feats.get("x_norm")
            if tokens is None:
                tokens = feats.get("x")
        else:
            tokens = feats

        if tokens is None:
            raise RuntimeError("Unable to locate patch tokens from DinoV3 backbone output.")

        # Remove CLS token if present.
        if tokens.ndim == 3 and tokens.shape[1] > 1:
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens

        bsz, num_tokens, dim = patch_tokens.shape
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"Patch tokens ({num_tokens}) cannot form a square grid. "
                "Ensure the input resolution matches the ViT patch size."
            )
        feat_map = patch_tokens.transpose(1, 2).reshape(bsz, dim, grid_size, grid_size)
        feat_map = F.normalize(feat_map, dim=1, eps=1e-6)

        if self.output_size is not None and feat_map.shape[-1] != self.output_size:
            feat_map = F.interpolate(
                feat_map,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )

        return feat_map.squeeze(0)
