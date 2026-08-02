"""Deterministic PyTorch full-forward, cached decode, and parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slm_lab.models.qwen3_reference import (
    DEFAULT_CONTRACT_PATH,
    load_reference_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKEN_FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/t10/token-fixtures-v1.json"


class ReferenceExecutionError(RuntimeError):
    """A model response cannot support deterministic reference execution."""


@dataclass(frozen=True)
class NumericalTolerance:
    """Frozen criteria for same-model full-forward/cache parity."""

    atol: float
    rtol: float
    protected_relative_max: float
    cosine_min: float
    top5_overlap_min: float
    require_top1: bool = True
    relative_floor: float = 1.0

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("absolute and relative tolerances must be non-negative")
        if self.protected_relative_max < 0 or self.relative_floor <= 0:
            raise ValueError("protected-relative parameters must be positive")
        if not 0 <= self.cosine_min <= 1:
            raise ValueError("cosine_min must be between zero and one")
        if not 0 <= self.top5_overlap_min <= 1:
            raise ValueError("top5_overlap_min must be between zero and one")


# Same pinned model, dtype, device, and eager attention implementation. These
# thresholds admit BF16 accumulation-order noise, not backend or dtype changes.
DEFAULT_TOLERANCE = NumericalTolerance(
    atol=0.25,
    rtol=0.02,
    protected_relative_max=0.10,
    cosine_min=0.999,
    top5_overlap_min=0.8,
)


@dataclass(frozen=True)
class LogitMetrics:
    max_absolute_error: float
    mean_absolute_error: float
    max_protected_relative_error: float
    cosine_similarity: float
    top1_reference: int
    top1_candidate: int
    top1_agreement: bool
    top5_overlap: float
    reference_top1_top2_margin: float
    allclose: bool
    passed: bool


@dataclass(frozen=True)
class StepEvidence:
    step: int
    prefix_length: int
    selected_token_id: int
    full_logits_sha256: str
    cached_logits_sha256: str
    metrics: LogitMetrics


@dataclass(frozen=True)
class GenerationEvidence:
    """Compact, reproducible result; full logits and cache tensors stay external."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool
    eos_token_id: int | None
    max_new_tokens: int
    tolerance: NumericalTolerance
    steps: tuple[StepEvidence, ...]
    evidence_sha256: str

    @property
    def passed(self) -> bool:
        return all(step.metrics.passed for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ReferenceExecutionError(
            "PyTorch is required for numerical reference execution"
        ) from exc
    return torch


def _extract_output(output: Any) -> tuple[Any, Any]:
    logits = (
        output.get("logits")
        if isinstance(output, dict)
        else getattr(output, "logits", None)
    )
    past_key_values = (
        output.get("past_key_values")
        if isinstance(output, dict)
        else getattr(output, "past_key_values", None)
    )
    if logits is None:
        raise ReferenceExecutionError("model output has no logits")
    if getattr(logits, "ndim", None) != 3:
        raise ReferenceExecutionError(
            "model logits must have shape [batch, sequence, vocabulary]"
        )
    return logits, past_key_values


def _validate_input_ids(input_ids: Any) -> None:
    if getattr(input_ids, "ndim", None) != 2:
        raise ReferenceExecutionError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ReferenceExecutionError(
            "T11 reference generation requires one non-empty prompt"
        )
    if input_ids.dtype.is_floating_point:
        raise ReferenceExecutionError("input_ids must have an integer dtype")


def _last_logits(output: Any) -> tuple[Any, Any]:
    logits, past_key_values = _extract_output(output)
    return logits[:, -1, :], past_key_values


def _tensor_sha256(tensor: Any) -> str:
    canonical = (
        tensor.detach().to(dtype=_require_torch().float32, device="cpu").contiguous()
    )
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def _greedy_token(logits: Any) -> Any:
    # torch.argmax returns the first maximum, which freezes the T10
    # lowest-token-ID tie break because vocabulary is ordered by token ID.
    return logits.argmax(dim=-1, keepdim=True)


def compare_logits(
    reference: Any,
    candidate: Any,
    tolerance: NumericalTolerance = DEFAULT_TOLERANCE,
) -> LogitMetrics:
    """Measure and enforce full-forward versus cached next-token parity."""

    torch = _require_torch()
    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[0] != 1
        or reference.shape[1] < 1
    ):
        raise ReferenceExecutionError(
            "next-token logits must have matching non-empty [1, vocabulary] shapes"
        )
    ref = reference.detach().to(dtype=torch.float64, device="cpu")
    cand = candidate.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
        raise ReferenceExecutionError("logits contain NaN or infinite values")

    difference = (ref - cand).abs()
    denominator = ref.abs().clamp_min(tolerance.relative_floor)
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=-1).min().item()
    reference_top5 = torch.topk(ref, k=min(5, ref.shape[-1]), dim=-1).indices
    candidate_top5 = torch.topk(cand, k=min(5, cand.shape[-1]), dim=-1).indices
    overlap_values = []
    for ref_row, candidate_row in zip(reference_top5, candidate_top5, strict=True):
        overlap_values.append(
            len(set(ref_row.tolist()) & set(candidate_row.tolist()))
            / len(ref_row.tolist())
        )
    top1_reference = int(ref.argmax(dim=-1)[0].item())
    top1_candidate = int(cand.argmax(dim=-1)[0].item())
    top_values = torch.topk(ref, k=min(2, ref.shape[-1]), dim=-1).values
    margin = (
        float("inf")
        if top_values.shape[-1] == 1
        else float((top_values[:, 0] - top_values[:, 1]).min().item())
    )
    allclose = bool(
        torch.allclose(
            ref,
            cand,
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            equal_nan=False,
        )
    )
    protected_relative = float((difference / denominator).max().item())
    top1_agreement = top1_reference == top1_candidate
    top5_overlap = min(overlap_values)
    passed = (
        allclose
        and protected_relative <= tolerance.protected_relative_max
        and cosine >= tolerance.cosine_min
        and top5_overlap >= tolerance.top5_overlap_min
        and (top1_agreement or not tolerance.require_top1)
    )
    return LogitMetrics(
        max_absolute_error=float(difference.max().item()),
        mean_absolute_error=float(difference.mean().item()),
        max_protected_relative_error=protected_relative,
        cosine_similarity=float(cosine),
        top1_reference=top1_reference,
        top1_candidate=top1_candidate,
        top1_agreement=top1_agreement,
        top5_overlap=float(top5_overlap),
        reference_top1_top2_margin=margin,
        allclose=allclose,
        passed=passed,
    )


def _model_call(model: Any, **kwargs: Any) -> Any:
    torch = _require_torch()
    with torch.inference_mode():
        return model(**kwargs)


def generate_full_forward(
    model: Any,
    input_ids: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[int, ...]:
    """Greedy reference generation by recomputing the complete prefix."""

    torch = _require_torch()
    _validate_input_ids(input_ids)
    if max_new_tokens < 0:
        raise ReferenceExecutionError("max_new_tokens must be non-negative")
    prefix = input_ids.clone()
    generated: list[int] = []
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(prefix, dtype=torch.long)
        logits, _ = _last_logits(
            _model_call(
                model,
                input_ids=prefix,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        )
        token = _greedy_token(logits)
        token_id = int(token.item())
        generated.append(token_id)
        prefix = torch.cat((prefix, token.to(device=prefix.device)), dim=1)
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return tuple(generated)


def generate_cached(
    model: Any,
    input_ids: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[int, ...]:
    """Greedy reference generation using prefill once and one-token decode."""

    torch = _require_torch()
    _validate_input_ids(input_ids)
    if max_new_tokens < 0:
        raise ReferenceExecutionError("max_new_tokens must be non-negative")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    logits, past_key_values = _last_logits(
        _model_call(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    )
    if max_new_tokens and past_key_values is None:
        raise ReferenceExecutionError("cached prefill returned no past_key_values")

    generated: list[int] = []
    for step in range(max_new_tokens):
        token = _greedy_token(logits)
        token_id = int(token.item())
        generated.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if step + 1 < max_new_tokens:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            logits, past_key_values = _last_logits(
                _model_call(
                    model,
                    input_ids=token.to(device=input_ids.device),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            )
            if past_key_values is None:
                raise ReferenceExecutionError(
                    f"cached decode step {step + 1} returned no past_key_values"
                )
    return tuple(generated)


def _evidence_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compare_full_and_cached(
    model: Any,
    input_ids: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    tolerance: NumericalTolerance = DEFAULT_TOLERANCE,
) -> GenerationEvidence:
    """Run both paths in lockstep and retain per-step parity evidence."""

    torch = _require_torch()
    _validate_input_ids(input_ids)
    if max_new_tokens < 0:
        raise ReferenceExecutionError("max_new_tokens must be non-negative")
    full_prefix = input_ids.clone()
    cached_attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    cached_logits, past_key_values = _last_logits(
        _model_call(
            model,
            input_ids=input_ids,
            attention_mask=cached_attention_mask,
            use_cache=True,
            return_dict=True,
        )
    )
    if max_new_tokens and past_key_values is None:
        raise ReferenceExecutionError("cached prefill returned no past_key_values")

    steps: list[StepEvidence] = []
    generated: list[int] = []
    stopped_on_eos = False
    for step in range(max_new_tokens):
        full_attention_mask = torch.ones_like(full_prefix, dtype=torch.long)
        full_logits, _ = _last_logits(
            _model_call(
                model,
                input_ids=full_prefix,
                attention_mask=full_attention_mask,
                use_cache=False,
                return_dict=True,
            )
        )
        metrics = compare_logits(full_logits, cached_logits, tolerance)
        full_token = _greedy_token(full_logits)
        cached_token = _greedy_token(cached_logits)
        if not torch.equal(full_token, cached_token):
            metrics = LogitMetrics(**{**asdict(metrics), "passed": False})
        selected_token_id = int(full_token.item())
        steps.append(
            StepEvidence(
                step=step,
                prefix_length=int(full_prefix.shape[1]),
                selected_token_id=selected_token_id,
                full_logits_sha256=_tensor_sha256(full_logits),
                cached_logits_sha256=_tensor_sha256(cached_logits),
                metrics=metrics,
            )
        )
        generated.append(selected_token_id)
        full_prefix = torch.cat(
            (full_prefix, full_token.to(device=full_prefix.device)), dim=1
        )
        if eos_token_id is not None and selected_token_id == eos_token_id:
            stopped_on_eos = True
            break
        if step + 1 < max_new_tokens:
            cached_attention_mask = torch.cat(
                (
                    cached_attention_mask,
                    torch.ones(
                        (cached_attention_mask.shape[0], 1),
                        dtype=cached_attention_mask.dtype,
                        device=cached_attention_mask.device,
                    ),
                ),
                dim=1,
            )
            # Teacher-force the reference token into both branches. If cached
            # logits select a different token, the current step retains that
            # mismatch in its metrics while every later comparison remains on
            # the same prefix and diagnoses cache execution rather than
            # compounding autoregressive divergence.
            cached_logits, past_key_values = _last_logits(
                _model_call(
                    model,
                    input_ids=full_token.to(device=input_ids.device),
                    attention_mask=cached_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            )
            if past_key_values is None:
                raise ReferenceExecutionError(
                    f"cached decode step {step + 1} returned no past_key_values"
                )

    digest_payload = {
        "prompt_token_ids": [int(value) for value in input_ids[0].tolist()],
        "generated_token_ids": generated,
        "stopped_on_eos": stopped_on_eos,
        "eos_token_id": eos_token_id,
        "max_new_tokens": max_new_tokens,
        "tolerance": asdict(tolerance),
        "steps": [asdict(step) for step in steps],
    }
    return GenerationEvidence(
        prompt_token_ids=tuple(digest_payload["prompt_token_ids"]),
        generated_token_ids=tuple(generated),
        stopped_on_eos=stopped_on_eos,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        tolerance=tolerance,
        steps=tuple(steps),
        evidence_sha256=_evidence_digest(digest_payload),
    )


def load_fixture_token_ids(
    fixture_id: str,
    path: Path | str = DEFAULT_TOKEN_FIXTURE_PATH,
) -> tuple[int, ...]:
    """Load one authored T10 canary without re-tokenizing it."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceExecutionError(f"cannot load T10 token fixtures: {exc}") from exc
    fixtures: list[dict[str, Any]] = list(payload.get("raw_canaries", []))
    chat_canary = payload.get("chat_canary")
    if isinstance(chat_canary, dict):
        fixtures.append(chat_canary)
    selected = next(
        (fixture for fixture in fixtures if fixture.get("id") == fixture_id), None
    )
    if selected is None:
        raise ReferenceExecutionError(f"unknown T10 fixture ID {fixture_id!r}")
    token_ids = selected.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise ReferenceExecutionError(f"{fixture_id}: token_ids must be integers")
    return tuple(token_ids)


def _json_safe_evidence(
    evidence: GenerationEvidence, runtime: dict[str, Any], fixture_id: str
) -> dict[str, Any]:
    payload = evidence.as_dict()
    for step in payload["steps"]:
        margin = step["metrics"]["reference_top1_top2_margin"]
        if not math.isfinite(margin):
            step["metrics"]["reference_top1_top2_margin"] = None
    return {
        "schema_version": 1,
        "task_id": "T11",
        "fixture_id": fixture_id,
        "model_contract": str(DEFAULT_CONTRACT_PATH.relative_to(PROJECT_ROOT)),
        "runtime": runtime,
        "evidence": payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="raw_ascii")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow download of the immutable public Qwen revision",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = load_reference_model(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        local_files_only=not args.allow_download,
    )
    torch = _require_torch()
    token_ids = load_fixture_token_ids(args.fixture)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    evidence = compare_full_and_cached(
        reference.model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=reference.contract.eos_token_id,
    )
    print(
        json.dumps(
            _json_safe_evidence(evidence, reference.runtime.as_dict(), args.fixture),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if evidence.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
