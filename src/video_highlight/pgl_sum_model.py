"""Mr.HiSum-compatible PGL-SUM network and checkpoint loader.

The architecture follows the academic, non-commercial implementation released
by the Mr.HiSum authors (https://github.com/MRHiSum/MR.HiSum, commit
``aeb667f71239887034c2974062a341d312762867``), which in turn adapts PGL-SUM.
This local version keeps the released state-dict names intact and only fixes the
absolute-position helper for one-frame local chunks.  That edge case does not
occur in the long Mr.HiSum training videos, but is common in this competition's
short clips.

Mr.HiSum's code is licensed for academic, non-commercial use.  Keep this notice
with redistributed source.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MRHISUM_PGL_INPUT_DIM = 1024
MRHISUM_PGL_NUM_SEGMENTS = 4
MRHISUM_PGL_GLOBAL_HEADS = 8


class SelfAttention(nn.Module):
    """State-dict-compatible attention cell used by the released checkpoint."""

    def __init__(
        self,
        input_size: int = MRHISUM_PGL_INPUT_DIM,
        output_size: int = MRHISUM_PGL_INPUT_DIM,
        freq: int = 10000,
        heads: int = 1,
        pos_enc: Optional[str] = None,
    ) -> None:
        super().__init__()
        if pos_enc is not None:
            pos_enc = str(pos_enc).lower()
            if pos_enc not in {"absolute", "relative"}:
                raise ValueError(f"unsupported positional encoding: {pos_enc}")
        if output_size % heads:
            raise ValueError("output_size must be divisible by heads")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.heads = int(heads)
        self.pos_enc = pos_enc
        self.freq = int(freq)
        self.Wk = nn.ModuleList(
            nn.Linear(input_size, output_size // heads, bias=False)
            for _ in range(heads)
        )
        self.Wq = nn.ModuleList(
            nn.Linear(input_size, output_size // heads, bias=False)
            for _ in range(heads)
        )
        self.Wv = nn.ModuleList(
            nn.Linear(input_size, output_size // heads, bias=False)
            for _ in range(heads)
        )
        self.out = nn.Linear(output_size, input_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, length: int) -> torch.Tensor:  # noqa: N802
        """Return the released PGL-SUM absolute position matrix.

        The upstream list-to-tensor construction produces a float index when
        ``length // 2 == 0``.  Explicit integer tensors retain upstream output
        for normal sequences while allowing one-frame local chunks.
        """

        length = max(0, int(length))
        device = self.out.weight.device
        result = torch.zeros(length, length, device=device)
        if length <= 1:
            return result
        positions = torch.arange(length, dtype=torch.long, device=device)
        dimensions = torch.arange(length // 2, dtype=torch.long, device=device)
        positions = positions.reshape(length, 1).repeat_interleave(
            dimensions.numel(), dim=1
        )
        dimensions = dimensions.repeat(length, 1)
        exponent = (2 * dimensions) / float(self.input_size)
        result[positions, 2 * dimensions] = torch.sin(
            positions / (float(self.freq) ** exponent)
        )
        result[positions, 2 * dimensions + 1] = torch.cos(
            positions / (float(self.freq) ** exponent)
        )
        return result

    def getRelativePosition(self, length: int) -> torch.Tensor:  # noqa: N802
        # The released Mr.HiSum checkpoint uses absolute encoding.  Keeping the
        # relative implementation makes the module state-compatible with the
        # upstream class without adding a second inference route.
        length = max(0, int(length))
        device = self.out.weight.device
        if length <= 1:
            return torch.zeros(length, length, device=device)
        i = torch.arange(length, dtype=torch.long, device=device).reshape(length, 1)
        i = i.repeat_interleave(length, dim=1)
        j = torch.arange(length, dtype=torch.long, device=device).repeat(length, 1)
        relative = j - i + length - 1
        result = torch.zeros(length, length, device=device)
        idx = torch.arange(length // 2, dtype=torch.long, device=device)
        denom = float(2 * length)
        result[:, 2 * idx] = torch.sin(
            relative[:, 2 * idx]
            / (float(self.freq) ** ((i[:, 2 * idx] + j[:, 2 * idx]) / denom))
        )
        result[:, 2 * idx + 1] = torch.cos(
            relative[:, 2 * idx + 1]
            / (
                float(self.freq)
                ** ((i[:, 2 * idx + 1] + j[:, 2 * idx + 1]) / denom)
            )
        )
        return result

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"attention input must be [B,T,D], got {tuple(x.shape)}")
        outputs = []
        last_weights: Optional[torch.Tensor] = None
        for head in range(self.heads):
            keys = self.Wk[head](x)
            queries = self.Wq[head](x)
            values = self.Wv[head](x)
            energies = torch.matmul(queries, keys.transpose(1, 2))
            if self.pos_enc == "absolute":
                energies = energies + self.getAbsolutePosition(energies.shape[2])
            elif self.pos_enc == "relative":
                energies = energies + self.getRelativePosition(energies.shape[2])
            if mask is not None:
                pair_mask = torch.matmul(
                    mask.unsqueeze(-1).float(),
                    mask.unsqueeze(-1).transpose(2, 1).float(),
                ).bool()
                energies = energies.masked_fill(~pair_mask, -1e9)
            weights = self.softmax(energies)
            dropped_weights = self.drop(weights)
            outputs.append(
                torch.matmul(values.transpose(1, 2), dropped_weights).transpose(1, 2)
            )
            last_weights = weights
        if not outputs or last_weights is None:
            raise RuntimeError("attention module contains no heads")
        return self.out(torch.cat(outputs, dim=2)), last_weights.clone()


class MultiAttention(nn.Module):
    def __init__(
        self,
        input_size: int = MRHISUM_PGL_INPUT_DIM,
        output_size: int = MRHISUM_PGL_INPUT_DIM,
        freq: int = 10000,
        pos_enc: Optional[str] = None,
        num_segments: Optional[int] = None,
        heads: int = 1,
        fusion: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(
            input_size=input_size,
            output_size=output_size,
            freq=freq,
            pos_enc=pos_enc,
            heads=heads,
        )
        self.num_segments = num_segments
        if num_segments is not None:
            if num_segments < 2 or output_size % num_segments:
                raise ValueError("num_segments must divide output_size and be at least two")
            self.local_attention = nn.ModuleList(
                SelfAttention(
                    input_size=input_size,
                    output_size=output_size // num_segments,
                    freq=freq,
                    pos_enc=pos_enc,
                    heads=4,
                )
                for _ in range(num_segments)
            )
        self.fusion = str(fusion).lower() if fusion is not None else None
        if self.fusion not in {None, "add", "mult", "avg", "max"}:
            raise ValueError(f"unsupported fusion: {fusion}")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weighted_value, attention_weights = self.attention(x, mask)
        if self.num_segments is None or self.fusion is None:
            return weighted_value, attention_weights
        segment_size = int(math.ceil(x.shape[1] / self.num_segments))
        for segment in range(self.num_segments):
            left = segment * segment_size
            right = min(x.shape[1], (segment + 1) * segment_size)
            local_x = x[:, left:right]
            local_mask = mask[:, left:right] if mask is not None else None
            local_value, _local_weights = self.local_attention[segment](
                local_x, local_mask
            )
            # Preserve the released implementation's normalization semantics.
            # Its checkpoint was trained with this exact batch-axis slice.
            weighted_value[left:right] = F.normalize(
                weighted_value[left:right].clone(), p=2, dim=2
            )
            local_value = F.normalize(local_value, p=2, dim=2)
            if self.fusion == "add":
                weighted_value[:, left:right] += local_value
            elif self.fusion == "mult":
                weighted_value[:, left:right] *= local_value
            elif self.fusion == "avg":
                weighted_value[:, left:right] += local_value
                weighted_value[:, left:right] /= 2
            else:
                weighted_value[:, left:right] = torch.max(
                    weighted_value[left:right].clone(), local_value
                )
        return weighted_value, attention_weights


class PGL_SUM(nn.Module):  # noqa: N801 - upstream checkpoint class spelling
    def __init__(
        self,
        input_size: int = MRHISUM_PGL_INPUT_DIM,
        output_size: int = MRHISUM_PGL_INPUT_DIM,
        freq: int = 10000,
        pos_enc: Optional[str] = "absolute",
        num_segments: Optional[int] = MRHISUM_PGL_NUM_SEGMENTS,
        heads: int = MRHISUM_PGL_GLOBAL_HEADS,
        fusion: Optional[str] = "add",
    ) -> None:
        super().__init__()
        self.attention = MultiAttention(
            input_size=input_size,
            output_size=output_size,
            freq=freq,
            pos_enc=pos_enc,
            num_segments=num_segments,
            heads=heads,
            fusion=fusion,
        )
        self.linear_1 = nn.Linear(input_size, input_size)
        self.linear_2 = nn.Linear(input_size, 1)
        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(input_size, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, frame_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame_features.ndim != 3 or frame_features.shape[-1] != MRHISUM_PGL_INPUT_DIM:
            raise ValueError(
                "PGL-SUM features must have shape [B,T,1024], got "
                f"{tuple(frame_features.shape)}"
            )
        residual = frame_features
        weighted_value, attention_weights = self.attention(frame_features, mask=mask)
        value = self.norm_y(self.drop(weighted_value + residual))
        value = self.norm_linear(self.drop(self.relu(self.linear_1(value))))
        value = self.sigmoid(self.linear_2(value))
        return value.view(frame_features.shape[0], -1), attention_weights


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _safe_torch_load(path: str | Path) -> Dict[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("PGL-SUM checkpoint does not contain a state dictionary")
    return payload


class MrHiSumPGLScorer:
    """Load one released Mr.HiSum fold and score a complete feature sequence."""

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Mr.HiSum PGL-SUM checkpoint not found: {checkpoint}")
        requested = str(device or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_sha256 = file_sha256(checkpoint)
        self.model = PGL_SUM()
        state = _safe_torch_load(checkpoint)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "checkpoint architecture mismatch: "
                f"missing={list(missing)} unexpected={list(unexpected)}"
            )
        self.model.to(self.device).eval()

    @property
    def fingerprint(self) -> Dict[str, Any]:
        return {
            "model": "mrhisum/PGL_SUM",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "input_dim": MRHISUM_PGL_INPUT_DIM,
            "num_segments": MRHISUM_PGL_NUM_SEGMENTS,
            "global_heads": MRHISUM_PGL_GLOBAL_HEADS,
            "device": str(self.device),
        }

    def score(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != MRHISUM_PGL_INPUT_DIM:
            raise ValueError(f"features must be [T,1024], got {array.shape}")
        if array.shape[0] < 1 or not np.all(np.isfinite(array)):
            raise ValueError("features must contain at least one finite timestep")
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            scores, _weights = self.model(tensor, mask=None)
        output = scores.squeeze(0).detach().float().cpu().numpy()
        if output.shape != (array.shape[0],) or not np.all(np.isfinite(output)):
            raise RuntimeError(
                f"PGL-SUM returned invalid score shape/value: {output.shape}"
            )
        return np.clip(output.astype(np.float64), 0.0, 1.0)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "MRHISUM_PGL_INPUT_DIM",
    "MrHiSumPGLScorer",
    "PGL_SUM",
    "file_sha256",
]
