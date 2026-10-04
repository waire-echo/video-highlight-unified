"""REZE-inspired binary semantic scoring for temporal highlight windows.

This module is deliberately an opt-in experiment.  It does not replace the
structured ``window_score`` scorer and it has no dependency on the spatial or
YOLO pipeline.  A frozen local Qwen model is asked one generic highlight
question per non-overlapping clip.  Instead of generating an answer, the
implementation reads the next-token logits for the *single-token* answers
``Yes`` and ``No`` and stores their conditional probability.

The class composes :class:`~video_highlight.window_scoring.QwenWindowScorer`
so model weights, the processor, E-drive runtime paths, original-video range
requests and decoder compatibility copies are shared with the existing
frontend.  Invalid tokenisation, non-finite logits and individual inference
errors are represented as missing values; they are never converted into a
fabricated binary score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

from .window_scoring import QwenWindowScorer
from .windowing import WindowSpec, probe_video, validate_window


REZE_WINDOW_SEC = 3.0
REZE_SAMPLE_FPS = 2.0
REZE_TAIL_MIN_SEC = 0.3
REZE_DEGENERATE_RANGE = 0.05


def _safe_record_float(value: Any) -> float:
    """Best-effort numeric value used only in invalid diagnostic records."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def build_generic_highlight_prompt() -> str:
    from .prompts import call_builder
    return call_builder("reze/yesno-v1", "build_generic_highlight_prompt", )


GENERIC_HIGHLIGHT_PROMPT = build_generic_highlight_prompt()


def generate_reze_windows(
    duration_sec: float,
    *,
    window_sec: float = REZE_WINDOW_SEC,
    sample_fps: float = REZE_SAMPLE_FPS,
    tail_min_sec: float = REZE_TAIL_MIN_SEC,
    start_window_id: int = 0,
) -> List[WindowSpec]:
    """Create non-overlapping REZE windows and discard only a tiny tail.

    Full windows are exactly ``window_sec`` seconds.  A remaining tail is kept
    as its own shorter window when it is at least ``tail_min_sec``; otherwise
    it is discarded.  This differs intentionally from the overlapping
    coarse/fine windows used by the structured scorer.
    """

    values = {
        "duration_sec": duration_sec,
        "window_sec": window_sec,
        "sample_fps": sample_fps,
        "tail_min_sec": tail_min_sec,
    }
    parsed: Dict[str, float] = {}
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        parsed[name] = number
    duration = parsed["duration_sec"]
    width = parsed["window_sec"]
    fps = parsed["sample_fps"]
    tail_min = parsed["tail_min_sec"]
    if duration < 0:
        raise ValueError("duration_sec must be non-negative")
    if width <= 0 or fps <= 0:
        raise ValueError("window_sec and sample_fps must be positive")
    if tail_min < 0 or tail_min > width:
        raise ValueError("tail_min_sec must be in [0, window_sec]")
    if duration <= 1e-9:
        return []

    windows: List[WindowSpec] = []
    start = 0.0
    next_id = int(start_window_id)
    while duration - start >= width - 1e-9:
        end = min(duration, start + width)
        windows.append(
            WindowSpec(
                next_id,
                round(start, 6),
                round(end, 6),
                fps,
                "reze",
                None,
            )
        )
        next_id += 1
        start = end

    remainder = duration - start
    if remainder >= tail_min - 1e-9 and remainder > 1e-9:
        windows.append(
            WindowSpec(
                next_id,
                round(start, 6),
                round(duration, 6),
                fps,
                "reze",
                None,
            )
        )
    return windows


def _normalise_token_ids(value: Any) -> List[int]:
    """Convert common tokenizer return shapes into one flat ID sequence."""

    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list):
        raise ValueError("tokenizer did not return an ID sequence")
    result: List[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("token ID cannot be boolean")
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError("token ID must be an integer") from exc
    return result


def _encode_without_special_tokens(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    elif callable(tokenizer):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    else:
        raise ValueError("processor has no usable tokenizer")
    return _normalise_token_ids(encoded)


@dataclass(frozen=True)
class YesNoTokenResolution:
    """Runtime proof that the two candidate answers are distinct single tokens."""

    yes_text: str = "Yes"
    no_text: str = "No"
    yes_token_ids: Tuple[int, ...] = ()
    no_token_ids: Tuple[int, ...] = ()
    valid: bool = False
    errors: Tuple[str, ...] = ()

    @property
    def yes_token_id(self) -> Optional[int]:
        return self.yes_token_ids[0] if self.valid else None

    @property
    def no_token_id(self) -> Optional[int]:
        return self.no_token_ids[0] if self.valid else None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["yes_token_ids"] = list(self.yes_token_ids)
        result["no_token_ids"] = list(self.no_token_ids)
        result["errors"] = list(self.errors)
        result["yes_token_id"] = self.yes_token_id
        result["no_token_id"] = self.no_token_id
        return result


def resolve_yes_no_tokens(
    processor_or_tokenizer: Any,
    *,
    yes_text: str = "Yes",
    no_text: str = "No",
) -> YesNoTokenResolution:
    """Resolve ``Yes``/``No`` at runtime and reject multi-token candidates."""

    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    errors: List[str] = []
    try:
        yes_ids = tuple(_encode_without_special_tokens(tokenizer, yes_text))
    except Exception as exc:
        yes_ids = ()
        errors.append(f"Yes tokenisation failed: {type(exc).__name__}: {exc}")
    try:
        no_ids = tuple(_encode_without_special_tokens(tokenizer, no_text))
    except Exception as exc:
        no_ids = ()
        errors.append(f"No tokenisation failed: {type(exc).__name__}: {exc}")
    if yes_ids and len(yes_ids) != 1:
        errors.append(f"Yes must be one token, got {len(yes_ids)}")
    if no_ids and len(no_ids) != 1:
        errors.append(f"No must be one token, got {len(no_ids)}")
    if len(yes_ids) == 1 and len(no_ids) == 1 and yes_ids[0] == no_ids[0]:
        errors.append("Yes and No resolved to the same token ID")
    valid = not errors and len(yes_ids) == 1 and len(no_ids) == 1
    if not valid and not errors:
        errors.append("Yes/No token IDs are unavailable")
    return YesNoTokenResolution(
        yes_text=str(yes_text),
        no_text=str(no_text),
        yes_token_ids=yes_ids,
        no_token_ids=no_ids,
        valid=valid,
        errors=tuple(errors),
    )


def stable_binary_probability(yes_logit: Any, no_logit: Any) -> Optional[float]:
    """Return ``exp(yes)/(exp(yes)+exp(no))`` without overflow.

    Only the two candidate logits participate in the denominator, matching the
    REZE-style conditional confidence.  Invalid or non-finite values remain
    missing instead of being coerced to zero.
    """

    try:
        yes = float(yes_logit)
        no = float(no_logit)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(yes) or not math.isfinite(no):
        return None
    maximum = max(yes, no)
    yes_exp = math.exp(yes - maximum)
    no_exp = math.exp(no - maximum)
    denominator = yes_exp + no_exp
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    result = yes_exp / denominator
    return result if math.isfinite(result) else None


@dataclass
class RezeWindowScore:
    """One auditable Yes/No probability record."""

    window_id: int
    start_sec: float
    end_sec: float
    sample_fps: float
    p_yes: Optional[float] = None
    yes_token_text: str = "Yes"
    no_token_text: str = "No"
    yes_token_ids: Tuple[int, ...] = ()
    no_token_ids: Tuple[int, ...] = ()
    yes_token_id: Optional[int] = None
    no_token_id: Optional[int] = None
    yes_logit: Optional[float] = None
    no_logit: Optional[float] = None
    valid: bool = False
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    stage: str = "reze"
    prompt: str = GENERIC_HIGHLIGHT_PROMPT

    @property
    def score(self) -> Optional[float]:
        return self.p_yes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": int(self.window_id),
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "sample_fps": float(self.sample_fps),
            "stage": self.stage,
            "semantic_mode": "reze_yesno",
            "p_yes": self.p_yes,
            "yes_token_text": self.yes_token_text,
            "no_token_text": self.no_token_text,
            "yes_token_ids": list(self.yes_token_ids),
            "no_token_ids": list(self.no_token_ids),
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "yes_logit": self.yes_logit,
            "no_logit": self.no_logit,
            "valid": bool(self.valid),
            "errors": list(self.errors),
            "attempts": int(self.attempts),
            "prompt": self.prompt,
        }


def diagnose_reze_scores(
    records: Sequence[Any],
    *,
    range_epsilon: float = REZE_DEGENERATE_RANGE,
    token_resolution: Optional[YesNoTokenResolution] = None,
) -> Dict[str, Any]:
    """Report whether the semantic curve has enough variation to be usable."""

    valid_scores: List[float] = []
    for record in records:
        if isinstance(record, Mapping):
            value = record.get("p_yes")
            valid = bool(record.get("valid", value is not None))
        else:
            value = getattr(record, "p_yes", None)
            valid = bool(getattr(record, "valid", value is not None))
        if not valid:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and 0.0 <= score <= 1.0:
            valid_scores.append(score)

    total = len(records)
    valid_count = len(valid_scores)
    score_min = min(valid_scores) if valid_scores else None
    score_max = max(valid_scores) if valid_scores else None
    score_range = (score_max - score_min) if valid_scores else None
    all_equal = bool(valid_scores) and (score_range or 0.0) <= 1e-12
    epsilon = max(0.0, float(range_epsilon))
    degenerate = not valid_scores or (score_range is not None and score_range <= epsilon)
    if not valid_scores:
        reason = "no_valid_probabilities"
    elif all_equal:
        reason = "all_probabilities_equal"
    elif score_range is not None and score_range <= epsilon:
        reason = "probability_range_too_small"
    else:
        reason = None
    result: Dict[str, Any] = {
        "semantic_mode": "reze_yesno",
        "total_windows": total,
        "valid_windows": valid_count,
        "missing_windows": total - valid_count,
        "p_yes_min": score_min,
        "p_yes_max": score_max,
        "p_yes_range": score_range,
        "degenerate_range_threshold": epsilon,
        "all_equal": all_equal,
        "degenerate": degenerate,
        "semantic_usable": not degenerate,
        "reason": reason,
    }
    if token_resolution is not None:
        result["token_resolution"] = token_resolution.to_dict()
    return result


@dataclass
class RezeScoringResult:
    records: List[RezeWindowScore] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0

    @property
    def window_records(self) -> List[RezeWindowScore]:
        return self.records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_mode": "reze_yesno",
            "duration_sec": float(self.duration_sec),
            "window_records": [record.to_dict() for record in self.records],
            "score_diagnostics": dict(self.diagnostics),
        }


InputBuilder = Callable[
    [QwenWindowScorer, str, float, float, float, str],
    Mapping[str, Any],
]


class RezeWindowScorer:
    """Read conditional Yes/No next-token confidence from a shared Qwen model."""

    def __init__(
        self,
        qwen_scorer: QwenWindowScorer,
        *,
        prompt: str = GENERIC_HIGHLIGHT_PROMPT,
        yes_text: str = "Yes",
        no_text: str = "No",
        input_builder: Optional[InputBuilder] = None,
        range_epsilon: float = REZE_DEGENERATE_RANGE,
    ):
        self.qwen_scorer = qwen_scorer
        self.prompt = str(prompt or GENERIC_HIGHLIGHT_PROMPT)
        self.yes_text = str(yes_text)
        self.no_text = str(no_text)
        self.input_builder = input_builder
        self.range_epsilon = max(0.0, float(range_epsilon))
        self._token_resolution: Optional[YesNoTokenResolution] = None
        self.fatal_disabled_reason: Optional[str] = None

    @property
    def model(self) -> Any:
        return getattr(self.qwen_scorer, "model", None)

    @property
    def processor(self) -> Any:
        return getattr(self.qwen_scorer, "processor", None)

    @property
    def token_resolution(self) -> YesNoTokenResolution:
        if self._token_resolution is None:
            if self.processor is None:
                self._token_resolution = YesNoTokenResolution(
                    yes_text=self.yes_text,
                    no_text=self.no_text,
                    valid=False,
                    errors=("Qwen processor is unavailable",),
                )
            else:
                self._token_resolution = resolve_yes_no_tokens(
                    self.processor,
                    yes_text=self.yes_text,
                    no_text=self.no_text,
                )
        return self._token_resolution

    def _prepare_model_inputs(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float,
        prompt: str,
    ) -> Mapping[str, Any]:
        if self.input_builder is not None:
            return self.input_builder(
                self.qwen_scorer,
                video_path,
                start_sec,
                end_sec,
                sample_fps,
                prompt,
            )
        if self.processor is None:
            raise RuntimeError("Qwen processor is unavailable")

        # Reuse the established original-MP4 range request.  The fallback copy
        # is produced only when the shared decoder reports a compatibility
        # error, matching the structured scorer's behaviour.
        content = self.qwen_scorer._video_request(
            video_path, start_sec, end_sec, sample_fps
        )
        messages = [
            {
                "role": "user",
                "content": [content, {"type": "text", "text": prompt}],
            }
        ]
        from qwen_vl_utils import process_vision_info

        try:
            image_inputs, video_inputs, video_kwargs = (
                self.qwen_scorer._process_vision_inputs(process_vision_info, messages)
            )
        except (OSError, RuntimeError, ValueError, EOFError, IndexError) as exc:
            compatible = self.qwen_scorer._transcode_video_for_rgb(video_path)
            warnings.warn(
                f"video decoder failed ({exc}); using cached compatibility copy: {compatible}",
                RuntimeWarning,
            )
            messages[0]["content"][0] = dict(content, video=compatible)
            image_inputs, video_inputs, video_kwargs = (
                self.qwen_scorer._process_vision_inputs(process_vision_info, messages)
            )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_kwargs = self.qwen_scorer._normalize_video_kwargs(video_kwargs)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _last_token_logits(logits: Any) -> Any:
        ndim = getattr(logits, "ndim", None)
        if ndim == 3:
            return logits[0, -1]
        if ndim == 2:
            return logits[0]
        if ndim == 1:
            return logits
        raise ValueError("model logits must have shape [B,T,V], [B,V], or [V]")

    @staticmethod
    def _scalar_at(vector: Any, token_id: int) -> float:
        shape = getattr(vector, "shape", None)
        if shape and int(shape[-1]) <= int(token_id):
            raise ValueError(f"token ID {token_id} exceeds logits vocabulary")
        value = vector[int(token_id)]
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def _forward_probability(
        self,
        inputs: Mapping[str, Any],
        tokens: YesNoTokenResolution,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.model is None:
            raise RuntimeError("Qwen model is unavailable")
        if not tokens.valid or tokens.yes_token_id is None or tokens.no_token_id is None:
            raise ValueError("Yes/No tokens are not valid single tokens")

        model_inputs: Any = inputs
        model_device = getattr(self.model, "device", None)
        if model_device is not None and hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(model_device)

        try:
            import torch

            context = torch.no_grad()
        except ImportError:  # pragma: no cover - production dependency
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            outputs = self.model(
                **model_inputs,
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("model output does not contain logits")
        vector = self._last_token_logits(logits)
        yes_logit = self._scalar_at(vector, tokens.yes_token_id)
        no_logit = self._scalar_at(vector, tokens.no_token_id)
        probability = stable_binary_probability(yes_logit, no_logit)
        return probability, yes_logit, no_logit

    def _missing_record(
        self,
        spec: WindowSpec,
        errors: Iterable[str],
        *,
        attempts: int,
    ) -> RezeWindowScore:
        tokens = self.token_resolution
        return RezeWindowScore(
            spec.window_id,
            spec.start_sec,
            spec.end_sec,
            spec.sample_fps,
            p_yes=None,
            yes_token_text=tokens.yes_text,
            no_token_text=tokens.no_text,
            yes_token_ids=tokens.yes_token_ids,
            no_token_ids=tokens.no_token_ids,
            yes_token_id=tokens.yes_token_id,
            no_token_id=tokens.no_token_id,
            valid=False,
            errors=[str(error) for error in errors],
            attempts=attempts,
            stage=spec.stage,
            prompt=self.prompt,
        )

    def score_window(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float = REZE_SAMPLE_FPS,
        *,
        window_id: int = 0,
        stage: str = "reze",
    ) -> RezeWindowScore:
        """Score one window once; any failure is returned as a missing record."""

        try:
            spec = validate_window(
                WindowSpec(
                    int(window_id),
                    float(start_sec),
                    float(end_sec),
                    float(sample_fps),
                    str(stage or "reze"),
                )
            )
        except Exception as exc:
            # Invalid external inputs have no trustworthy interval.  Preserve
            # what can be represented without letting the batch raise.
            return RezeWindowScore(
                int(window_id),
                _safe_record_float(start_sec),
                _safe_record_float(end_sec),
                _safe_record_float(sample_fps),
                valid=False,
                errors=[f"invalid window: {type(exc).__name__}: {exc}"],
                attempts=0,
                stage=str(stage or "reze"),
                prompt=self.prompt,
            )

        tokens = self.token_resolution
        if not tokens.valid:
            return self._missing_record(spec, tokens.errors, attempts=0)
        if self.fatal_disabled_reason:
            return self._missing_record(spec, [self.fatal_disabled_reason], attempts=0)
        base_load_error = getattr(self.qwen_scorer, "model_load_error", None)
        if base_load_error:
            return self._missing_record(spec, [str(base_load_error)], attempts=0)

        inputs: Optional[Mapping[str, Any]] = None
        try:
            inputs = self._prepare_model_inputs(
                video_path,
                spec.start_sec,
                spec.end_sec,
                spec.sample_fps,
                self.prompt,
            )
            probability, yes_logit, no_logit = self._forward_probability(inputs, tokens)
            if probability is None:
                return RezeWindowScore(
                    spec.window_id,
                    spec.start_sec,
                    spec.end_sec,
                    spec.sample_fps,
                    p_yes=None,
                    yes_token_text=tokens.yes_text,
                    no_token_text=tokens.no_text,
                    yes_token_ids=tokens.yes_token_ids,
                    no_token_ids=tokens.no_token_ids,
                    yes_token_id=tokens.yes_token_id,
                    no_token_id=tokens.no_token_id,
                    yes_logit=yes_logit,
                    no_logit=no_logit,
                    valid=False,
                    errors=["Yes/No logits produced a non-finite probability"],
                    attempts=1,
                    stage=spec.stage,
                    prompt=self.prompt,
                )
            return RezeWindowScore(
                spec.window_id,
                spec.start_sec,
                spec.end_sec,
                spec.sample_fps,
                p_yes=probability,
                yes_token_text=tokens.yes_text,
                no_token_text=tokens.no_text,
                yes_token_ids=tokens.yes_token_ids,
                no_token_ids=tokens.no_token_ids,
                yes_token_id=tokens.yes_token_id,
                no_token_id=tokens.no_token_id,
                yes_logit=yes_logit,
                no_logit=no_logit,
                valid=True,
                attempts=1,
                stage=spec.stage,
                prompt=self.prompt,
            )
        except Exception as exc:
            error = f"REZE inference failed: {type(exc).__name__}: {exc}"
            is_resource_error = getattr(
                self.qwen_scorer, "_is_resource_error", lambda _exc: False
            )(exc)
            if is_resource_error:
                self.fatal_disabled_reason = error
                clear_cache = getattr(self.qwen_scorer, "_clear_cuda_cache", None)
                if callable(clear_cache):
                    clear_cache()
            return self._missing_record(spec, [error], attempts=1)
        finally:
            # Batch tensors may retain decoded video frames.  The shared model
            # stays loaded; only per-window inputs are released.
            if inputs is not None:
                del inputs

    def score_windows(
        self,
        video_path: str,
        windows: Iterable[WindowSpec],
    ) -> RezeScoringResult:
        records: List[RezeWindowScore] = []
        duration = 0.0
        for external in windows:
            try:
                spec = validate_window(external)
            except Exception as exc:
                records.append(
                    RezeWindowScore(
                        int(getattr(external, "window_id", len(records))),
                        _safe_record_float(getattr(external, "start_sec", 0.0)),
                        _safe_record_float(getattr(external, "end_sec", 0.0)),
                        _safe_record_float(getattr(external, "sample_fps", 0.0)),
                        valid=False,
                        errors=[f"invalid window: {type(exc).__name__}: {exc}"],
                        attempts=0,
                    )
                )
                continue
            duration = max(duration, spec.end_sec)
            records.append(
                self.score_window(
                    video_path,
                    spec.start_sec,
                    spec.end_sec,
                    spec.sample_fps,
                    window_id=spec.window_id,
                    stage=spec.stage,
                )
            )
        diagnostics = diagnose_reze_scores(
            records,
            range_epsilon=self.range_epsilon,
            token_resolution=self.token_resolution,
        )
        return RezeScoringResult(records, diagnostics, duration)

    def score_video(
        self,
        video_path: str,
        *,
        duration_sec: Optional[float] = None,
    ) -> RezeScoringResult:
        """Probe and score one video using fixed 3-second/2-FPS windows."""

        if duration_sec is None:
            duration, _fps, _frames, _width, _height = probe_video(video_path)
        else:
            duration = float(duration_sec)
        windows = generate_reze_windows(duration)
        result = self.score_windows(video_path, windows)
        result.duration_sec = max(0.0, duration)
        result.diagnostics["configured_window_sec"] = REZE_WINDOW_SEC
        result.diagnostics["configured_sample_fps"] = REZE_SAMPLE_FPS
        result.diagnostics["configured_tail_min_sec"] = REZE_TAIL_MIN_SEC
        if not windows:
            result.diagnostics["reason"] = result.diagnostics.get("reason") or "no_windows"
        return result


__all__ = [
    "GENERIC_HIGHLIGHT_PROMPT",
    "REZE_DEGENERATE_RANGE",
    "REZE_SAMPLE_FPS",
    "REZE_TAIL_MIN_SEC",
    "REZE_WINDOW_SEC",
    "RezeScoringResult",
    "RezeWindowScore",
    "RezeWindowScorer",
    "YesNoTokenResolution",
    "build_generic_highlight_prompt",
    "diagnose_reze_scores",
    "generate_reze_windows",
    "resolve_yes_no_tokens",
    "stable_binary_probability",
]
