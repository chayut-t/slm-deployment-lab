# T10: Token Prompt Evaluation Fixtures

Date: 2026-07-25
Task: `T10`
Visibility: `public`
Status: completed

## Outcome

T10 freezes the repository's authored prompts, exact Qwen3-0.6B token IDs,
generation policy, canaries, and small CC0 quality subset. The immutable
tokenizer revision reproduces every committed sequence, and all four static
workloads contain exactly 128, 512, 1,024, and 4,096 prompt tokens.

The fixture chain is traceable from authored source through a canonical source
hash, generated token bundle, canonical bundle hash, and declarative workload
manifest. Structural checks run without network access; the tokenizer extra
performs exact upstream regeneration without loading model weights.

## Changes

- Added the exact optional tokenizer environment:
  `transformers==4.51.3`, `tokenizers==0.21.4`, and `jinja2==3.1.6` through the
  committed lock.
- Added `slm-lab-fixtures` generation, offline checking, and pinned-tokenizer
  verification under `src/slm_lab/evaluation/fixtures.py`.
- Added four raw-completion canaries, one explicitly non-thinking chat canary,
  four exact context workloads, and four repository-authored CC0 quality cases.
- Named WikiText-2, HellaSwag, ARC Easy, and PIQA as T13 candidates without
  committing third-party dataset rows.
- Froze greedy/no-sampling decoding, argmax tie-breaking, seed policy,
  EOS/PAD behavior, and output limits for contexts and canaries.
- Added canonical metadata and token-array drift checks, including adversarial
  tests where an editor recomputes hashes after tampering.
- Added `docs/learning/token_prompt_fixtures.md` and expanded workload and
  evaluation READMEs.
- Updated the package-lock hash in host evidence and made the dependency-gate
  regression derive a genuinely blocked task instead of hard-coding one that
  later becomes ready.

## Verification

- `uv sync --extra dev --extra tokenizer --locked`
  - Exact locked development and tokenizer environment synchronized.
- `SLM_LAB_VERIFY_UPSTREAM=1 SLM_LAB_VERIFY_OFFLINE=1 pytest -q
  tests/repo/test_t10_fixtures.py`
  - 13 passed against the cached immutable public tokenizer.
- `pytest -q`
  - 58 passed, 2 expected upstream/network-gated skips.
- `ruff check src tests`
  - Passed.
- `slm-lab-fixtures check`
  - Offline structure, provenance, privacy, context, and hash validation passed.
- `slm-lab-fixtures verify --offline`
  - Exact tokenizer re-encoding passed.
- Isolated offline regeneration plus byte comparison
  - Both the token bundle and workload config reproduced byte-for-byte.
- Hash-constrained `uv build`
  - Source distribution and wheel built; the evaluation module and CLI entry
    point were included.
- `python3 scripts/ai/render_task_status.py --check`
  - Task graph valid and generated status current.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed for 163 public files including completion metadata.
- Independent review of `3a40e3e`
  - Found incomplete whole-manifest validation, a missing deterministic
    generation contract, and an eager-import warning.
- Independent re-review of `ff55784`
  - Passed with no remaining findings after all corrections.

## Decisions and evidence

- Raw completion remains the canonical numerical interface; only a dedicated
  chat canary uses the pinned template with `enable_thinking=false`.
- Qwen's non-thinking template still renders an empty `<think>…</think>` block.
  Those control tokens are preserved because removing them changes token IDs.
- Context prompts are made exact by truncating token IDs, decoding them, and
  requiring an identical re-encoding rather than estimating tokens from text.
- Source fixture canonical SHA-256:
  `c1c0b23904b47236121086b9167bdd467d12ada22bfc67af9b9561e25c9f0639`.
- Token bundle canonical SHA-256:
  `9f9268ae4a366faa4325271492ec52f035bbf3ba0973d2de61f63382e6302745`.
- Implementation/review-fix commit:
  `ff55784dd6d19d67617ff9761b8cd61e1b68cd56`.

## Risks and limitations

- T10 verifies tokenization and deterministic generation configuration, not
  model logits or generated answers. T11 owns the numerical oracle.
- The small CC0 quality subset is a regression surface, not a model-capability
  benchmark. T13 owns exact academic dataset revisions and selection.
- Offline structural validation cannot prove token IDs came from upstream
  bytes by itself. The pinned-tokenizer verification command provides that
  stronger check.
- No model weights, hardware profile, paid service, or private data was used.

## Follow-up

- Newly unblocked tasks: T11 and T13. T40 satisfies its T10 dependency but
  remains blocked on T20.
- Recommended next action: implement T11's deterministic PyTorch full-forward
  and cached reference using the committed token IDs and generation policy;
  T13 can proceed in parallel on benchmark methodology and academic subset
  pinning without changing T10 fixtures.
