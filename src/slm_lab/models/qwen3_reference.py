"""Pinned, deterministic Qwen3 PyTorch reference-model loading."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "configs/models/qwen3-0.6b.yaml"
EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SUPPORTED_DTYPES = {"float32", "bfloat16", "float16"}


class ReferenceConfigurationError(ValueError):
    """The reference model or execution request violates the frozen contract."""


class ReferenceDependencyError(RuntimeError):
    """An exact runtime dependency required for reference execution is absent."""


@dataclass(frozen=True)
class ModelContract:
    """Subset of the T00 model contract needed by T11."""

    model_id: str
    revision: str
    reference_dtype: str
    trust_remote_code: bool
    eos_token_id: int
    pad_token_id: int
    architecture: dict[str, Any]
    source_path: Path


@dataclass(frozen=True)
class ReferenceRuntime:
    """Exact execution identity recorded beside numerical evidence."""

    python_version: str
    torch_version: str
    transformers_version: str
    safetensors_version: str
    device: str
    dtype: str
    deterministic_algorithms: bool
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "safetensors_version": self.safetensors_version,
            "device": self.device,
            "dtype": self.dtype,
            "deterministic_algorithms": self.deterministic_algorithms,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReferenceModel:
    """Loaded model plus the immutable contract and runtime provenance."""

    model: Any
    contract: ModelContract
    runtime: ReferenceRuntime


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ReferenceDependencyError(
            f"{package} is required for PyTorch reference execution"
        ) from exc


def load_model_contract(path: Path | str = DEFAULT_CONTRACT_PATH) -> ModelContract:
    """Load and strictly validate the immutable Qwen3 model identity."""

    source_path = Path(path).resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        model = payload["model"]
        tokenizer = payload["tokenizer"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReferenceConfigurationError(
            f"invalid model contract {source_path}: {exc}"
        ) from exc

    model_id = model.get("id")
    revision = model.get("revision")
    trust_remote_code = model.get("trust_remote_code")
    reference_dtype = model.get("reference_dtype")
    eos_token_id = tokenizer.get("tokens", {}).get("eos_id")
    pad_token_id = tokenizer.get("tokens", {}).get("pad_id")
    architecture = model.get("architecture")

    if model_id != EXPECTED_MODEL_ID:
        raise ReferenceConfigurationError(
            f"expected model {EXPECTED_MODEL_ID!r}, found {model_id!r}"
        )
    if revision != EXPECTED_MODEL_REVISION:
        raise ReferenceConfigurationError(
            f"expected immutable revision {EXPECTED_MODEL_REVISION}, "
            f"found {revision!r}"
        )
    if trust_remote_code is not False:
        raise ReferenceConfigurationError("trust_remote_code must remain false")
    if reference_dtype not in SUPPORTED_DTYPES:
        raise ReferenceConfigurationError(
            f"unsupported reference dtype {reference_dtype!r}"
        )
    if not isinstance(eos_token_id, int) or not isinstance(pad_token_id, int):
        raise ReferenceConfigurationError("EOS and PAD token IDs must be integers")
    if not isinstance(architecture, dict):
        raise ReferenceConfigurationError("model architecture must be a mapping")

    return ModelContract(
        model_id=model_id,
        revision=revision,
        reference_dtype=reference_dtype,
        trust_remote_code=trust_remote_code,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        architecture=architecture,
        source_path=source_path,
    )


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ReferenceConfigurationError(
            f"dtype must be one of {sorted(mapping)}, found {name!r}"
        ) from exc


def configure_determinism(torch: Any, *, seed: int) -> None:
    """Configure deterministic inference without changing global default dtype."""

    if seed < 0:
        raise ReferenceConfigurationError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_reference_model(
    *,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    device: str = "cpu",
    dtype: str | None = None,
    seed: int = 0,
    local_files_only: bool = True,
    attn_implementation: str = "eager",
) -> ReferenceModel:
    """Load the pinned Qwen revision in eval mode with deterministic settings.

    Network access is opt-in. Passing ``local_files_only=False`` may download
    public weights, but never changes the revision or enables remote code.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ReferenceDependencyError(
            "install exact PyTorch and Transformers versions before loading "
            "the numerical reference"
        ) from exc

    contract = load_model_contract(contract_path)
    selected_dtype = dtype or contract.reference_dtype
    tensor_dtype = _torch_dtype(torch, selected_dtype)
    configure_determinism(torch, seed=seed)

    model = AutoModelForCausalLM.from_pretrained(
        contract.model_id,
        revision=contract.revision,
        trust_remote_code=contract.trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=tensor_dtype,
        attn_implementation=attn_implementation,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    runtime = ReferenceRuntime(
        python_version=platform.python_version(),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        safetensors_version=_package_version("safetensors"),
        device=str(device),
        dtype=selected_dtype,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        seed=seed,
    )
    return ReferenceModel(model=model, contract=contract, runtime=runtime)
