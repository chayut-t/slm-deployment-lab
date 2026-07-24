# ADR 0001: Pin Qwen3-0.6B and require exact toolchain evidence

Status: accepted
Date: 2026-07-24
Related task: `T00`

## Context

Every deployment lane must start from the same model and tokenizer bytes.
Floating Hugging Face branch names, unrecorded chat-template changes, and
partially captured compiler/runtime versions would make numerical comparisons
and failure reproductions unreliable.

The project plan selects `Qwen/Qwen3-0.6B` as the primary model, prioritizes
Qualcomm before Apple and NVIDIA, and reserves fallback models for a documented
public Qualcomm compiler or runtime boundary. T00 must freeze that source
identity without prematurely claiming that an untested combination of Python,
Qualcomm, Apple, or NVIDIA packages is compatible.

## Decision

### Source model and tokenizer

The primary checkpoint and tokenizer are both:

| Field | Pinned value |
|---|---|
| Repository | `Qwen/Qwen3-0.6B` |
| Model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Tokenizer revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| License | Apache-2.0 |
| Reference dtype | BF16 |
| Weight format | safetensors |
| `trust_remote_code` | `false` |

The revision is a full Git commit SHA, not the floating `main` ref. It was
resolved independently with:

```bash
git ls-remote https://huggingface.co/Qwen/Qwen3-0.6B refs/heads/main
```

The command returned the pinned revision on 2026-07-24. Downstream code must
pass the full revision when loading model or tokenizer data.

Small upstream metadata files were fetched at that revision and hashed:

| Content | SHA-256 |
|---|---|
| Raw `config.json` bytes | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Raw `tokenizer_config.json` bytes | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| Decoded chat-template UTF-8 bytes | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |

The chat-template digest is computed from the JSON-decoded `chat_template`
string encoded as UTF-8, with no added newline. The declarative source of truth
for all pinned values is
`configs/models/qwen3-0.6b.yaml`.

### Model and tokenizer behavior

The frozen architecture facts needed by downstream contracts are:

- 28 decoder layers, hidden size 1,024, and intermediate size 3,072.
- 16 query heads, 8 key/value heads, and head dimension 128.
- Vocabulary size 151,936.
- RoPE theta 1,000,000 with no configured scaling.
- Tied input/output embeddings.
- Upstream model-config maximum positions of 40,960.
- Upstream tokenizer `model_max_length` of 131,072.

Those two upstream length fields have different meanings and must not be used
interchangeably. The lab's core static graph contexts remain the smaller,
explicit set `128`, `512`, `1024`, and `4096`.

The upstream model config names token ID 151643 as BOS and 151645 as EOS. The
tokenizer does not automatically add a BOS token; it uses `<|im_end|>`/151645
as EOS and `<|endoftext|>`/151643 as PAD. Downstream fixtures must preserve
that distinction instead of assuming that tokenizer BOS behavior follows the
model-config field.

For bounded deterministic validation:

- Raw completion inputs are the canonical interface.
- When the chat template is intentionally used, callers must pass
  `enable_thinking=false`.
- The exact pinned chat template and its hash must be recorded in derived
  fixture or artifact manifests.
- T10 owns concrete prompts, token IDs, and generation settings.

Qwen's model card says Qwen3 support requires `transformers>=4.51.0`. That is a
compatibility floor, not the final environment pin. T01 will select and lock
the exact tested package set.

### Toolchain-version policy

T01 owns exact environment selection and the repository lockfile. All later
tasks must nevertheless follow this capture policy:

1. Reproducible package manifests use exact installed versions and lock or
   artifact hashes. Ranges, `latest`, and unexpanded branch names are not
   evidence.
2. Git-sourced dependencies use full commit SHAs.
3. Hosted services and vendor SDKs record the exact version/build reported by
   the service, compiler, runtime, or device at job time.
4. A version that is not exposed is recorded as `null` together with the
   command/source checked and an explanation. It is never guessed.
5. Every produced artifact or measurement links to the repository Git commit,
   task ID, source model/tokenizer revisions, and a host/target manifest.
6. Compatibility claims apply only to the exact recorded combination.

The model config reproduces the plan's flat artifact-manifest field names
exactly and defines flat platform-specific extensions for four evidence
groups:

- Artifact manifest: the complete field set frozen in the project plan,
  including `model_revision`, `runtime_version`, and `qairt_version`.
- Common environment/build evidence: Python, Transformers, PyTorch, ONNX,
  ONNX Runtime, exporter, opset, compiler, runtime, operating system, and
  repository Git commit.
- Qualcomm: `qai-hub`, AIMET, QAIRT/QNN compiler and runtime, target device,
  target OS/build, and hosted job identifiers after sanitization.
- Apple: MLX, MLX-LM when used, macOS, Xcode/Metal toolchain, Instruments, and
  the exact Mac hardware profile.
- NVIDIA: ONNX Runtime GPU package, CUDA, cuDNN, NVIDIA driver, GPU, and Linux
  distribution/kernel.

Generic `compiler`/`compiler_version` and `runtime`/`runtime_version` fields
remain required in build or execution evidence even when a platform-specific
field is also present. This makes cross-platform manifests queryable without
erasing vendor detail. Platform manifests may extend the stable artifact
contract but must not rename its fields.

### Scope

This decision preserves the project plan's priority and fallback rules:

1. Qualcomm public AI Hub and Device Cloud workflows.
2. Apple M4 with a custom MLX runtime.
3. NVIDIA CUDA through ONNX Runtime.

Qwen3-0.6B remains the only primary model. A fallback may be selected only
after preserving a reproducible Qwen compiler/runtime failure, and all fallback
results must be labeled as such.

## Alternatives considered

### Follow the Hugging Face `main` branch

Rejected. Model, tokenizer, or template data could change without a repository
change here, invalidating token fixtures and numerical baselines.

### Pin the model but allow a floating tokenizer

Rejected. Tokenization and template behavior are part of the graph input
contract and must be reproducible from the same immutable source snapshot.

### Freeze all package versions in T00

Rejected. The environment lanes have different compatibility constraints, and
T01 is explicitly responsible for selecting tested package versions and
creating `uv.lock`. T00 freezes the evidence policy and the upstream model
compatibility floor.

### Use Qwen3's default thinking mode for validation

Rejected for bounded deterministic validation. Raw completion inputs reduce
template coupling; explicit non-thinking chat formatting remains available
when a chat-shaped fixture is required.

### Choose a smaller fallback model now

Rejected. Public Qualcomm support changes over time. The plan requires a
current, evidence-based fallback only after the Qwen boundary is isolated.

## Consequences

- Downstream tasks have one immutable model/tokenizer identity and template
  hash.
- T01 must materialize the declared version fields in environment and host
  manifests and choose exact compatible package versions.
- T10 can generate stable prompt/token fixtures without ambiguity about
  thinking mode or special tokens.
- Updating the checkpoint, tokenizer, template behavior, or capture policy
  requires a new ADR that supersedes this one and regeneration of affected
  evidence.
- The pinned revision does not itself prove that Qualcomm, MLX, ONNX, or CUDA
  execution works. Those claims remain gated by their task-specific numerical
  and hardware verification.

## References

- [Project plan](../project/plan.md)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pinned Hugging Face revision](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
- [Pinned model config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
- [Pinned tokenizer config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
