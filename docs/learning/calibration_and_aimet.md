# Calibration data as a quantization contract

Task: `T40`

## Why this exists

Post-training quantization does not read weights alone. It reads *activations*,
and it reads them by running real inputs through the floating model while range
observers watch each tensor. The observed minimum and maximum of every observed
tensor become that tensor's scale and zero-point. Those encodings are then
baked into the deployed artifact.

That makes the calibration inputs part of the deployed model:

```text
frozen calibration inputs
  → observed activation ranges
  → per-tensor/per-block scales and zero-points
  → quantized graph
  → every quality, latency, and memory number measured afterward
```

Change the inputs and you change the encodings. Change the encodings and you
have a different model, even with identical weights and identical quantization
settings. A W8 quality delta measured next week is therefore not comparable to
one measured today unless the corpus revision, the preprocessing, and the
ordering are frozen exactly the way
[T10 froze the tokenizer revision](token_prompt_fixtures.md).

T40 freezes them. It quantizes nothing: AIMET is Linux + CUDA only, and the
primary development host is an Apple M4 Mac mini. What is frozen here is the
*input half* of every experiment in the Lane B matrix of
[`docs/project/plan.md`](../project/plan.md) section 7.2 (Q1–Q5), plus the
preflight that refuses to hand the corpus to a quantizer until the floating
baseline the delta will be measured against has been re-verified.

The committed entrypoint is
[`configs/quantization/calibration.yaml`](../../configs/quantization/calibration.yaml).
It is a generated file, and its generator is
[`src/slm_lab/quantization/calibration.py`](../../src/slm_lab/quantization/calibration.py).

## The corpus is a fixed point, not a document

The contract file is not hand-maintained. `validate_repository` rebuilds the
whole document from committed inputs and fails unless the file on disk is
**byte-identical** to a fresh regeneration. A hand-edited comment, a reordered
key, or an appended line fails the check (`test_formatting_drift_is_rejected`).

Four committed inputs are pinned by canonical-JSON hash inside the contract:

| Input | Role |
|---|---|
| [`tests/fixtures/t10/token-fixtures-v1.json`](../../tests/fixtures/t10/token-fixtures-v1.json) | the tier-1 calibration source |
| [`configs/workloads/t10-token-fixtures.json`](../../configs/workloads/t10-token-fixtures.json) | T10 workload manifest; pins the bundle hash |
| [`configs/models/qwen3-0.6b.yaml`](../../configs/models/qwen3-0.6b.yaml) | T00 model/tokenizer identity and vocabulary size |
| [`configs/models/qwen3-0.6b-onnx-export.json`](../../configs/models/qwen3-0.6b-onnx-export.json) | T20 export pins; fixes the shapes calibrated here |

Only the `calibration_corpus` block is hashed into the corpus revision. That is
deliberate: a licence note or a prose edit elsewhere in the file must not
change the identity of the calibration inputs, and a change to the samples,
preprocessing, or ordering must.

```text
calibration_corpus_canonical_json_sha256:
  d2b749e15dd5d98789305e06fbe164bfea7c4ce5cb87b2189e1a8374adbfe8f0

calibration_dataset_revision:
  t40-qwen3-0.6b-t10-derived-v1+sha256.d2b749e15dd5d987
```

`calibration_dataset_revision` is the string plan section 17.4 requires in
every artifact manifest. A T41+ manifest whose value differs was produced
against a different corpus, and its quality delta is not comparable to anything
produced against this one.

The freeze is load-bearing because the tests prove each knob moves the hash
*through the production validator*. `TestDriftGuards` in
[`tests/quantization/test_calibration.py`](../../tests/quantization/test_calibration.py)
writes a mutated contract into a scratch copy of the committed tree — reordered
samples, a reassigned context, a relaxed padding policy, a dropped sample, a
different tiling target, a tampered coverage figure — and asserts
`validate_repository` rejects each one with a specific message.

That distinction matters. A test that deep-copies the corpus dict, mutates the
copy, and asserts the hash changed would prove only that SHA-256 is not a
constant function; it would exercise no production code path. No such test
lives here.

## Two tiers, one of them committed

| Tier | ID | Status | Licence | In Git |
|---|---|---|---|---|
| 1 | `t10_derived` | required and committed | CC0-1.0 | yes |
| 2 | `external_diversity` | declared candidate, owned by T41 | mixed | no |

Tier 1 is 13 samples and 6,912 token positions derived deterministically from
the CC0-1.0 T10 fixtures. It regenerates offline, byte-for-byte, on any machine
with the standard library and PyYAML. Nothing heavier is needed, because the
corpus is built from token IDs and never touches a tokenizer or a model.

Tier 2 records three revision-pinned public corpora as candidates for T41. No
row of any of them is committed. See "Licensing" below for why that boundary is
where it is.

## Why this corpus represents the target workloads

This is the T40 study checkpoint in
[`docs/project/learning-checkpoints.md`](../project/learning-checkpoints.md):
"review the calibration corpus manifest and explain why it represents the
target workloads". The answer is per selection, and every selection is recorded
as a `rationale` string on the sample itself — a T10 fixture with no rationale
is a hard error in `_rationale_for`, so the corpus cannot silently widen.

| Sample | Group | Interface | Source tokens | Construction | Shape | Distinct IDs |
|---|---|---|---:|---|---:|---:|
| `cal-00-S128` | context_workloads | raw | 128 | verbatim | 128 | 80 |
| `cal-01-S512` | context_workloads | raw | 512 | verbatim | 512 | 80 |
| `cal-02-S1024` | context_workloads | raw | 1,024 | verbatim | 1,024 | 80 |
| `cal-03-S4096` | context_workloads | raw | 4,096 | verbatim | 4,096 | 80 |
| `cal-04-raw_ascii` | raw_canaries | raw | 18 | tiled ×8 | 128 | 18 |
| `cal-05-raw_structured` | raw_canaries | raw | 28 | tiled ×5 | 128 | 25 |
| `cal-06-raw_unicode` | raw_canaries | raw | 19 | tiled ×7 | 128 | 18 |
| `cal-07-raw_whitespace` | raw_canaries | raw | 24 | tiled ×6 | 128 | 16 |
| `cal-08-chat_non_thinking` | chat_canary | chat template | 34 | tiled ×4 | 128 | 25 |
| `cal-09-quality_arithmetic_sequence` | quality_subset | raw | 19 | tiled ×7 | 128 | 13 |
| `cal-10-quality_cache_definition` | quality_subset | raw | 20 | tiled ×7 | 128 | 20 |
| `cal-11-quality_json_contract` | quality_subset | raw | 27 | tiled ×5 | 128 | 21 |
| `cal-12-quality_timing_boundary` | quality_subset | raw | 22 | tiled ×6 | 128 | 22 |

The `Distinct IDs` column is `samples[].distinct_token_ids` in the contract,
computed from each emitted sequence at generation time. Read it before reading
any `rationale`: the rationales argue about token *classes*, and this column is
the measurement that stops those arguments being read as vocabulary coverage.

### 1. All four deployment context lengths, verbatim — one body of text

Range statistics at 128 tokens do not predict those at 4,096. Attention scores
and residual-stream magnitudes grow with sequence length, so a scale fitted
only on short prompts clips on long ones, and a scale fitted only on long
prompts throws away INT8 resolution on the short shape that decodes most often.

These are also not arbitrary lengths. 128, 512, 1,024, and 4,096 are exactly
the shapes T20 exported and T13 benchmarks, and each maps to one frozen T12
static cache capacity:

| Prompt length | Cache capacity |
|---:|---:|
| 128 | 160 |
| 512 | 576 |
| 1,024 | 1,152 |
| 4,096 | 4,224 |

Calibrating a shape that is never deployed would waste range budget; skipping a
shape that is deployed guarantees clipping on it.

**But this is one argument, not four.** T10 built all four context workloads by
repeating a single 602-character authored seed, so the four are strict token-ID
prefixes of one another: S128 ⊂ S512 ⊂ S1024 ⊂ S4096. All four contain the same
80 distinct token IDs, and their union contains no more. The S4096 prompt is
39 complete copies of that seed plus a truncated fortieth. The contract states
this in `calibration_corpus.coverage` (`context_workloads_are_nested_prefixes:
true`, with the chain listed), and `tests/quantization/test_calibration.py`
asserts the nesting independently, so the manifest cannot quietly present four
independent coverage arguments.

The four are still worth having — but as four *shapes*, not as four bodies of
text.

### 2. Four raw canaries for token-class presence

Activation ranges in the first layers are dominated by which embedding rows are
touched, and the BPE vocabulary is not uniform. The four canaries put four
token classes in front of the observer that the repeated seed does not
otherwise supply:

- **ASCII** letters and punctuation — the class every deployed prompt contains.
- **Whitespace** — newlines, indentation, and runs of spaces, which merge into
  whitespace tokens no prose sample in this corpus emits.
- **Unicode** — multi-byte UTF-8 byte-fallback tokens, a class the other three
  canaries never produce.
- **Structured** — JSON-like delimiters and quoted field names, a small set of
  punctuation tokens repeating at high frequency.

Note carefully what kind of claim this is, because it is easy to overread. It
is a *presence* argument over four named token classes, derived from how the
tokenizer works. It is **not** a claim that these canaries cover the embedding
table, sample its densest or highest-entropy regions, or bound any activation
tail. All four together contribute 61 distinct token IDs. No activation
histogram has been measured; nothing in T40 has been through a range observer.
T41 owns the first measurement that could confirm or refute the argument.

### 3. One chat canary because the control-token path is different

The Qwen chat template renders `<|im_start|>`, `<|im_end|>`, and — with
thinking disabled — an explicitly empty `<think>…</think>` block. Those token
IDs never appear on the raw-completion path. Their embeddings, and the
attention pattern that forms around them, are only observed if the chat canary
is calibrated. A deployment that serves chat traffic while calibrating only raw
completions has never shown the observer its own control tokens.

### 4. The CC0 quality subset because it is what the delta is measured on

The four committed quality prompts are the ones T41–T43 measure the W8/W4
quality delta on. Including them means the observers see the distribution the
delta is computed over. That is deliberate — and it is also a bias, declared in
the next section rather than hidden.

### Token budget

| Prompt shape | Tokens | Source group | Tokens |
|---|---:|---|---:|
| S128 | 1,280 | context_workloads | 5,760 |
| S512 | 512 | raw_canaries | 512 |
| S1024 | 1,024 | chat_canary | 128 |
| S4096 | 4,096 | quality_subset | 512 |
| **total** | **6,912** | **total** | **6,912** |

Ten of the thirteen samples sit at 128 tokens, but 59% of the token positions
come from the single S4096 sample, and 83% come from the four nested context
workloads together — that is, from one repeated seed. That is the shape of a
coverage argument, not of a statistical sample.

## What this corpus does not license

Be precise about this. The corpus is a good contract and a small dataset.

**The corpus touches 175 of 151,936 embedding rows — 0.115% of the
vocabulary.** That is `calibration_corpus.coverage.distinct_token_ids` and
`vocabulary_fraction`, measured from the emitted token IDs at generation time.
Per source group: 80 distinct IDs across all four context workloads, 61 across
the four raw canaries, 66 across the quality subset, 25 in the chat canary. A
per-tensor activation range fitted on this input is fitted on a very narrow
slice of the embedding table, and the corpus makes no claim about the rows it
never touches. Nothing here licenses a statement of the form "the observers saw
a representative sample of the vocabulary".

**83% of the token budget is one body of text.** `token_share_per_source_group`
records `context_workloads: 0.8333` — 5,760 of 6,912 positions — and those four
samples are nested prefixes of a single repeated 602-character seed. The nine
tiled samples supply the remaining 17%, and each of those is itself a short
sequence repeated to length. Read together with the previous paragraph: the
corpus is a shape-and-token-class contract, not a distributional sample.

**6,912 token positions is small for PTQ.** Published PTQ recipes commonly use
hundreds of thousands of tokens. A tail-sensitive observer (percentile or MSE
based) fitted on this much data can miss the true distribution tail. The
contract states the budget openly and says widening it is tier-2 work owned by
T41.

**Tiled samples are self-similar.** Nine of the thirteen samples repeat a short
source sequence to reach 128 tokens. Repetition is acceptable for fitting
activation *ranges*, which depend on which token classes appear rather than on
how natural the sequence reads. It is not acceptable anywhere else: a tiled
sample must never be reused as a quality-evaluation prompt or as a latency
workload, because its perplexity and its cache-reuse behaviour are artifacts of
the repetition.

**A tiled sample is not a verified tokenization.** T10's own context workloads
repeat the authored *text*, tokenize it, truncate the IDs, decode, and then
re-encode, failing generation unless the round trip reproduces every ID. T40's
tiling operates directly on token IDs and does no round-trip check, so the seam
between two repeats may be an ID adjacency the tokenizer would never emit from
any string. For range observation that is harmless. For anything that claims to
be a real prompt it is not. The same applies to the chat canary: its 34 tokens
tile into 128 as three complete template renderings plus a 26-token fragment,
which is not a valid chat conversation.

**Calibration and evaluation overlap.** Four of the thirteen tier-1 samples are
the same CC0 prompts the quality delta is measured on. The quantized model's
ranges were fitted on the evaluation inputs, so the measured delta is
optimistically biased. `licensing.evaluation_overlap` in the contract states
this and requires T41–T43 to report the overlap alongside any quality number
and never to present the delta as evidence of generalization.

**Prefill only.** Calibration exercises the prefill graph. The decode graph
consumes a materialized KV cache, so its activation ranges are a function of
the prefill ranges plus the decode step. Whether a decode-side observer pass is
needed is an open question owned by T41.

Tier 2 exists precisely because of the first and fourth limits. Admitting an
external corpus both widens coverage and removes the calibration/evaluation
overlap.

## The T12 padding conflict

This is a clean worked example of a frozen upstream contract constraining a
downstream design.

Short fixtures — 18 to 34 tokens — have to reach an exact frozen prompt length,
because the exported graphs are static. The obvious move is to pad. It was
rejected for two independent reasons.

The first is mechanical. The T12 prefill contract in
[`src/slm_lab/contracts/static_cache.py`](../../src/slm_lab/contracts/static_cache.py)
documents `attention_mask` as "One for every real prompt token; padding is not
permitted." A padded calibration sample emits a tensor the exported graph does
not accept. Calibrating on inputs the deployed graph would reject is not
calibration.

The second is statistical, and it is the more interesting one. Padding would
feed the pad embedding (ID `151643`) through the network for every padded
position. The range observers would then fit scales that account for
activations that never occur at inference. Scope that claim precisely: the 9
short fixtures carry 211 real tokens between them and would occupy 9 × 128 =
1,152 positions, so **within those nine samples** about 82% of observed
positions would be pad. Corpus-wide it is 941 of 6,912, about 13.6%, because
the four verbatim context workloads dominate the budget. Either way the nine
samples that exist to show the observer four token classes and the chat control
tokens would be showing it mostly pad, which defeats their entire purpose.

So the corpus tiles instead: repeat the source token-ID sequence end to end and
truncate to the target length. Every position is a real token, `attention_mask`
is all ones for every sample, and `position_ids` is a zero-based contiguous
range.

Be precise about *how* that is guaranteed: by construction, not by a check.
`build_prefill_tensors` writes `[1] * length` and `list(range(length))`
directly, so the assertions in `validate_prefill_tensors` and in
`test_no_sample_is_ever_padded` are construction post-conditions — they can
only fire if that function changes. What they do independently catch is a T12
contract violation: a token count that is not the graph's frozen prompt length,
or a dtype or shape `validate_tensor_mapping` rejects. Neither the module nor
the test suite proves "no padding" about a tensor mapping this module did not
build; nothing consumes such a mapping today.

The pad and EOS IDs are still recorded in the contract, marked
`pad_id_recorded_for_provenance_only` and
`eos_id_recorded_for_provenance_only`, so a reader can see that the decision
was made rather than overlooked.

## Baseline parity before quantization

A quantization quality delta is a subtraction. If the floating side of the
subtraction drifted — a re-export, a bit-rotted file on the external volume, a
manifest that no longer matches its artifact — the delta measures the drift,
not the quantization. So the floating baseline is re-verified before any weight
is quantized. That is the T40 acceptance criterion "baseline model parity is
checked before quantization", implemented as a fail-closed gate in
[`src/slm_lab/quantization/parity.py`](../../src/slm_lab/quantization/parity.py).

Parity has two halves and the module never lets a caller conflate them.

### Half one: artifact identity — ran here

Four checks, none of which needs anything heavier than `hashlib`:

| Check | What it compares |
|---|---|
| `attestation_manifest_agreement` | the T20 evidence attestation against the four committed `results/manifests/onnx/S*.json` manifests, for every graph and external-data digest, with no missing or orphan context |
| `model_revision_agreement` | model and tokenizer revision across the T00 contract, the T20 export config, the manifests, and `slm_lab.contracts.static_cache.MODEL_REVISION` |
| `t12_contract_conformance` | recorded graph boundaries against the frozen T12 prefill/decode contracts by tensor name, order, dtype, and static shape |
| `artifact_byte_identity` | the bytes on the external artifact root, re-hashed, against the committed digests and sizes |

All four passed on 2026-08-02. The byte check re-hashed 16 files —
`prefill.onnx`, `prefill.onnx.data`, `decode.onnx`, `decode.onnx.data` for each
of S128/S512/S1024/S4096 — totalling 9,586,211,364 bytes, every one
matching its recorded SHA-256 and `size_bytes`. The record is
[`results/quantization/t40-baseline-parity-2026-08-02.json`](../../results/quantization/t40-baseline-parity-2026-08-02.json).

Read that record's own provenance honestly: `repository.git_commit` is
`bda66e2` and `repository.git_tree_clean` is **`false`**. The gate ran on top
of the uncommitted T40 working tree, because the T40 work is what produced the
gate. So the run is pinned to the T20 inputs at `bda66e2` — the export config,
the manifests, and the artifact bytes it measured are all committed at that
commit and none of them is T40-owned — but it is not a run of a clean tree, and
the record says so rather than implying otherwise. The next time the gate runs
on a committed tree, that field will read `true` and the claim will be
correspondingly stronger.

The `repository` block carries no checkout path. An absolute checkout path is
meaningless on any machine that later reads the record, so `repository` holds
the commit and the clean flag and nothing else. The record as a whole is not
path-free, and should not be described as if it were: `artifact_root` records
absolute `requested`, `resolved`, and `artifact_directory` values. That is
deliberate. The external artifact root is a published location — the same
`/Volumes/T9/slm-deployment-lab` that
`configs/storage/external-ssd.example.yaml` already commits — and *which* root
was measured is part of what the record attests.

A check status is one of `passed`, `failed`, `unavailable`, or `skipped`, and
only `passed` is evidence. An unmounted artifact root yields `unavailable` and
names every file it could not check; `--no-verify-artifact-bytes` yields
`skipped`. Neither is ever silently upgraded to a pass.

### Half two: numerical logit parity — did not run

Logit-level agreement between the T11 deterministic PyTorch reference and the
T20 float16 ONNX export needs `torch` and `onnxruntime`. The primary macOS host
carries neither, and T40 installs no heavy dependency. The requirement is
therefore recorded as data, not performed:

```text
numerical_parity.status              not_run
numerical_parity.measurement         declared_requirement_not_executed
numerical_parity.owner_task          T21
numerical_parity.consumer_tasks      T41, T42, T43
```

The record carries the two commands that would close it, each labelled with
what it establishes and whether it exists yet: the T11 PyTorch oracle is
`implemented_at_this_commit` and establishes the oracle only, not ONNX
agreement; the ONNX Runtime parity run is `not_implemented_at_this_commit` and
is delivered by T21. It also records that no export-parity *tolerance* has been
frozen yet — `src/slm_lab/generation/reference.py::DEFAULT_TOLERANCE` bounds
full-forward versus cached-decode drift for one model, dtype, and device, which
is a different quantity. Until T21 freezes an export-parity tolerance, no
PyTorch-versus-ONNX pass or fail claim can be made at all.

### Reading the verdict

Three fields must be read together, and the committed record shows why:

```text
verdict                                          partial
verdict_scope                                    artifact_identity_only
released_for_calibration_on_artifact_identity    true
```

- `verdict` cannot read `verified` at this commit — not "does not happen to",
  but *cannot*: `check_baseline_parity` composes its two halves through the
  pure function `overall_verdict`, and no branch of it assigns `verified` to
  the overall verdict, because half two is a declared requirement and no input
  turns a declaration into a measurement. Note that the *identity* half does
  have a `verified` state; `artifact_identity.verdict` is where to read it.
  Two tests in
  [`tests/quantization/test_baseline_parity.py`](../../tests/quantization/test_baseline_parity.py)
  pin the behaviour, and it is worth seeing why it takes two. A unit test walks
  `overall_verdict` over every identity verdict including `verified`. That
  alone would not catch a `verified` branch reintroduced at the *call site*, so
  a second test drives the real `check_baseline_parity` to a fully `verified`
  identity half against three tiny stub files in `tmp_path`, and asserts the
  overall verdict is still `partial`. Before the function was extracted, the
  only test that reached this state at all was gated on the opt-in nine-
  gigabyte re-hash, so the guarantee was asserted by three documents and
  exercised by nothing that runs by default. Making `verified` reachable is T21
  work that must add a real numerical result, not flip a flag.
- `verdict_scope` says which half the verdict is about.
- `released_for_calibration_on_artifact_identity` is keyed on artifact identity
  alone, and its name says so. It is `true` here. That is the gate's actual
  contract: it releases the corpus once the floating artifact is provably the
  recorded one, and it does not pretend numerical parity was established. The
  field is deliberately *not* called `released_for_calibration`; a bare boolean
  with that name sitting beside `verdict: partial` is exactly how a partial
  result gets read as a complete one, and this is the field T41 keys on.
- The CLI exit code follows artifact identity, not the overall verdict. `parity
  verify` exits 0 with `verdict: partial`. **Exit code 0 is not full parity.**

`claim_boundary` in the record is authoritative in both directions. It lists
what a passing record establishes, and it lists what it never establishes:
PyTorch-versus-ONNX logit parity, ONNX Runtime execution or multi-step cache
correctness, any quantized quality delta, compiler acceptance or accelerator
placement, and latency or memory performance. See
[`results/quantization/README.md`](../../results/quantization/README.md).

## Licensing

Reproducibility does not grant a right to redistribute data. T10 already made
that distinction; T40 inherits it.

**Tier 1 is CC0-1.0.** It is a deterministic token-ID transformation of the T10
fixtures, which were authored for this repository and released CC0-1.0. No
third-party text enters the corpus, so the derivative stays CC0 and stays
compatible with the Apache-2.0 repository.

**Tier 2 carries obligations that would not stay compatible.** Three candidates
are pinned to exact dataset repository revisions, with licences read from the
Hugging Face dataset API on 2026-08-02:

| Dataset | Revision | Licence | Obligation |
|---|---|---|---|
| `Salesforce/wikitext` | `b08601e04326c79dfdd32d625aee71d232d685c3` | CC-BY-SA-3.0, GFDL | attribution + share-alike; GFDL adds copy-of-licence and invariant-section terms |
| `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | ODC-BY | attribution to the database producer and a licence notice on any redistributed database or derivative database |
| `wikimedia/wikipedia` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC-BY-SA-3.0, GFDL | same as wikitext |

Share-alike is the decisive one: a committed derivative of wikitext or
Wikipedia would have to be relicensed away from Apache-2.0. So the repository
records identifiers, revisions, licences, an assessment, and a risk — and no
rows. That is enough to make a future calibration run reproducible (`verify
--online` re-fetches every revision and licence and fails closed on drift)
without importing a licence obligation into the repository.

The tier-2 table is also duplicated on purpose. `FETCHED_EXTERNAL_METADATA` in
the test file is an independent second copy of the fetched revisions and
licences, so a drift in the module constants fails the test instead of quietly
agreeing with itself.

## The AIMET environment

[`environments/linux-aimet/`](../../environments/linux-aimet/README.md) is
specified, not executed. No AIMET code has run for this repository.

### Why it cannot run on the primary host

The evidence is the wheel platform tag. `aimet-onnx` 2.36.0 ships exactly one
file on PyPI:

```text
aimet_onnx-2.36.0-cp310-abi3-manylinux_2_34_x86_64.whl
```

`manylinux_2_34_x86_64` excludes macOS and excludes `arm64`, and requires glibc
2.34 or newer. The same holds for every release from 2.32.1 through 2.36.0.

The 2.36.0 GitHub release does publish two Windows wheels that PyPI does not
carry (`+cpu` `win_amd64` and `win_arm64`), so "AIMET is Linux-only" would be
wrong. The accurate statement is narrower and is about the compiled
distribution: `aimet-onnx` has no macOS wheel — the installation page's macOS
(Apple Silicon) row reads "Build from source" — and the only wheel in the
release carrying CUDA kernels is `aimet_onnx-2.36.0+cu126` for Linux x86-64.
The release's second `+cu126` asset, `aimet_torch-2.36.0+cu126-py310-none-any`,
is pure Python. Diffing all 380 entries against its `+cpu` twin on 2026-08-02
found no compiled object in either and exactly two differences: the four
`dist-info/*` files, and the shipped source file
`aimet_torch/common/_version.py`, which records `torch = '2.13.0+cu126'`
against `torch = '2.13.0+cpu'`. So it differs in its version metadata and in
one recorded `torch` build string — not in code. Note that
`win_arm64` is Windows-on-ARM — it does nothing for an `arm64` Mac. The
conclusion for this host is unchanged.

`aimet-torch` 2.36.0 does ship a pure-Python `py310-none-any` wheel, which
installs anywhere — and installing it on macOS produces nothing usable, because
it contains no compiled extension. Its `aimet_common` content is a deprecation
shim that re-exports `aimet_onnx.common` or `aimet_torch.common`. The
quantization kernels live in `aimet-onnx`. The size split across the 2.36.0
release assets confirms where the code is: `aimet_onnx-2.36.0+cu126` is
68,864,490 bytes against 778,078 bytes for `+cpu` of the same release, while
every `aimet-torch` variant is about 792 KB.

That single platform tag is also why AIMET is deliberately *not* a
`pyproject.toml` optional extra. An extra would make `uv sync --extra` fail
outright on the primary macOS host. The stack is pinned under
`environments/linux-aimet/` instead, which is where
[`environments/common-toolchain.json`](../../environments/common-toolchain.json)
already registers platform extensions and names T40 as this one's owner.

### Verified versus unverified

The environment README separates these rigorously, and the split is the point.

Verified against a live source on 2026-08-02: the AIMET 2.36.0 versions and
digests from the PyPI JSON API, the BSD-3-Clause licence and `>=3.10` Python
requirement, the `+cu126`/`+cpu` wheel digests and sizes from the GitHub
releases API, and a complete transitive resolution.
[`aimet-requirements.lock`](../../environments/linux-aimet/aimet-requirements.lock)
was produced by `uv` 0.11.32 with `--python-platform x86_64-manylinux_2_34
--python-version 3.11 --generate-hashes --no-build`, so it resolves from
published wheel metadata alone and reproduces from the arm64 macOS host.

What the resolution establishes: AIMET 2.36.0 co-resolves with T20's
`torch==2.7.1`, `onnx==1.18.0`, and `transformers==4.51.3` with no backtracking
conflict and no version relaxation, and pulls in
`nvidia-cuda-runtime-cu12==12.6.77` and `nvidia-cudnn-cu12==9.5.1.17` — CUDA
12.6, the same minor AIMET's validated `+cu126` wheels target.

What it does not establish: that the packages import, that AIMET's compiled
kernels load, that CUDA initializes, or that any Qwen3-0.6B graph quantizes.
Those are runtime facts, and the README lists fifteen of them in an
`unverified` table, each with the exact command that would produce the value on
a real Linux CUDA host. Do not copy any of those into a manifest until the
command has actually run there.

The same discipline applies to
[`aimet-host.template.json`](../../environments/linux-aimet/aimet-host.template.json):
it deliberately fails schema validation while its `<REPLACE: ...>` placeholders
remain, because there is no truthful version string for a GPU that does not
exist yet.

But the validator is *not* a completeness gate, and that is worth internalising
as a general lesson about schema validation. `host-v1.schema.json` types
`hardware.product`, `hardware.cpu.cores`, `hardware.gpu.count`, and
`platform_details.nvidia.gpu_name` loosely enough that a literal
`"<REPLACE: nproc, as an integer>"` satisfies them. Filling in only the fields
the validator complains about yields `host manifest valid` with four
placeholders still in the file — verified on 2026-08-02 by doing exactly that.
The environment README therefore prescribes two checks: the schema validator,
then a `grep -n '<REPLACE:'` that must print nothing. **A schema tells you a
document is well-formed, never that it is filled in.**

Two findings worth knowing before the first run, both from reading the shipped
wheel rather than executing it:

- The AIMET docs write the smoke test as `from aimet_common import libpymo`,
  but that module has been deprecated since AIMET 2.20 and raises `ImportError`
  when both `aimet_onnx` and `aimet_torch` are installed — which is exactly this
  environment. Use `from aimet_onnx.common import libpymo`.
- The docs' `libpymo.EncodingAnalyzerForPython(...)` form does not exist in
  2.36.0 either. The compiled Cython module `aimet_onnx/common/_libpymo`
  defines exactly `TfEncoding`, `BlockTensorQuantizer`, and `PtrToInt64`; the
  `*EncodingAnalyzer` classes are C++ symbols with no Python binding. The
  README smoke test drives `BlockTensorQuantizer`, the class `aimet_onnx`'s own
  `qc_quantize_op.py` uses, and records the exact `strings` command that lists
  the exported names so the derivation can be re-checked.

There is a method lesson in the second one. The shared object is compiled with
`CYTHON_COMPRESS_STRINGS`, so most of its Python-level string constants are
compressed: *grepping the binary for a symbol and finding nothing does not
prove the symbol is absent*. The reliable signal is the uncompressed
qualified-name strings Cython emits for its types and methods
(`_libpymo.BlockTensorQuantizer.updateStats`, and so on), which enumerate the
Python surface positively rather than by absence.

Both remain **derived, not executed** — the wheel is Linux x86-64 and was
downloaded and hashed, never installed. The README says so at the point of use,
which is the only place that label does any good.

### The T20 pin-divergence rule

The export host and the calibration host are separate environments and are
allowed to differ. They are not allowed to differ *silently*.

Holding T20's pins is currently free rather than lucky: `aimet-torch` 2.36.0
declares `torch` and `torchvision` with no upper bound in its base
requirements, and `aimet-onnx` declares `torch` and `onnx` unbounded, with
`onnxruntime>=1.19`, `onnxscript>=0.4.0`, and `onnx_ir>=0.1.16` the only floors
that constrain this stack. A
hard `torch` pin appears only under the optional `v1-deps` extra, which this
environment does not install — but it is a clear signal of what Qualcomm
actually tests against, and this environment sits well behind it. "Resolves"
and "is tested" are different claims.

The pin is also a small lesson in reading a version number too fast. On PyPI,
`aimet-torch` 2.36.0's `v1-deps` extra says `torch==2.12.*` and its shipped
`_version.py` records `torch = '2.12.1+cu126'`. In the GitHub release assets of
the *same* 2.36.0 — the `+cpu` and `+cu126` wheels this lab actually pins —
both read `torch==2.13.*` and `torch = '2.13.0+...'`. Same release number,
different build. A run manifest should therefore record which distribution was
installed, not only `2.36.0`.

The rule, in short:

1. Prefer holding T20's pins. A quantization result is comparable to the T20
   float16 baseline only if the graph it consumed is the T20 graph.
2. If a future AIMET release forces a different `torch`, `onnx`, or
   `transformers`, record the forced version, the release that forced it, and
   the resolver error — then re-run baseline parity on the new stack.
3. Any run on a diverged stack records both version sets in its artifact
   manifest and is labelled as such wherever its numbers appear.
4. Never re-export the reference graph from this environment to make the pins
   agree. A re-export changes the `graph_sha256` values every downstream
   comparison is keyed on.

The interpreter falls under the same rule. T20 originally exported on CPython
3.11.15 while this environment uses the repository-pinned 3.11.13; the T23
re-attestation moved the pin to 3.11.13, so no interpreter divergence remains.

## Run it yourself

Repository commands use the `uv run` convention. The parity commands need
`SLM_LAB_ARTIFACT_ROOT` (on the primary machine it resolves to the committed
default in
[`configs/storage/external-ssd.example.yaml`](../../configs/storage/external-ssd.example.yaml)).

| Command | Proves | Works offline on macOS |
|---|---|---|
| `uv run python -m slm_lab.quantization.calibration check` | the contract regenerates byte-identically from committed inputs, every recorded hash matches, and every sample satisfies the T12 prefill contract | yes |
| `uv run python -m slm_lab.quantization.calibration verify --online` | additionally, the three tier-2 dataset revisions and card licences still match upstream | no — needs the Hugging Face dataset API |
| `uv run python -m slm_lab.quantization.parity verify` | artifact identity, including a full re-hash of all 16 recorded T20 files | needs the external artifact volume |
| `uv run pytest tests/quantization` | the whole offline contract: freeze, drift guards, coverage, licensing, parity verdict logic | yes |
| `SLM_LAB_T40_VERIFY_ARTIFACT_BYTES=1 uv run pytest tests/quantization -q` | additionally, the opt-in ~9 GB re-hash of every recorded artifact | needs the external artifact volume |

Results observed on the primary Mac mini on 2026-08-02, at commit `bda66e2`
with the T40 working tree uncommitted:

- `calibration check` and `calibration verify --online` both exited 0.
- `parity verify` printed `baseline parity verdict: partial (scope:
  artifact_identity_only)` with all four identity checks `passed`, `numerical
  parity: not_run (owner T21)`, and `released for calibration (artifact
  identity only): True`. Re-hashing 9,586,211,364 bytes from the external SSD
  took about 12 seconds of wall time on that host — an I/O timing for planning
  purposes, not a benchmark result.
- `pytest tests/quantization` reported 93 passed, 1 skipped; the skip is the
  opt-in byte re-hash.
- With `SLM_LAB_T40_VERIFY_ARTIFACT_BYTES=1`, 94 passed in 12.8 s.

Recompute the corpus revision by hand, without the module, to see that the hash
is over exactly the `calibration_corpus` block:

```bash
python3 -c "import hashlib,json,yaml; \
  c=yaml.safe_load(open('configs/quantization/calibration.yaml'))['calibration_corpus']; \
  b=json.dumps(c,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'); \
  print(hashlib.sha256(b).hexdigest())"
```

It prints `d2b749e15dd5d98789305e06fbe164bfea7c4ce5cb87b2189e1a8374adbfe8f0`.

Regeneration must be a fixed point. `generate` rewrites the committed contract
in place, so run it only on a clean tree and expect an empty diff:

```bash
uv run python -m slm_lab.quantization.calibration generate
git diff --stat configs/quantization/calibration.yaml   # expect no change
```

Regenerating into a scratch copy of the four committed inputs on 2026-08-02
reproduced the committed file byte-for-byte. `calibration check` asserts the
same property without writing anything, which is why it is the command to reach
for first.

Inspect what each sample claims to expose:

```bash
uv run python -c "import yaml; \
  c=yaml.safe_load(open('configs/quantization/calibration.yaml')); \
  [print(s['id'], s['context_length'], s['construction']) \
   for s in c['calibration_corpus']['samples']]"
```

The AIMET half of T40 cannot be exercised from here at all. The commands below
need a Linux x86-64 CUDA host and have **not** been run for this repository. Do
not record their output as evidence until they have:

```bash
uv pip install --python "${AIMET_VENV}/bin/python" \
  --require-hashes -r environments/linux-aimet/aimet-requirements.lock
"${AIMET_VENV}/bin/python" -c 'import aimet_onnx; print(aimet_onnx.__version__)'
```

## Primary references

Recorded and reachability-checked on 2026-08-02 in
[`environments/linux-aimet/README.md`](../../environments/linux-aimet/README.md),
which also notes that `quic.github.io` and `qualcomm.github.io` both serve
`aimet-pages` and that PyPI metadata treats the latter as primary.

- [AIMET](https://github.com/quic/aimet)
- [AIMET documentation](https://qualcomm.github.io/aimet-pages/)
- [AIMET installation](https://qualcomm.github.io/aimet-pages/releases/latest/overview/install/index.html)
- [AIMET LPBQ](https://quic.github.io/aimet-pages/releases/latest/techniques/lpbq.html)
- [AIMET GitHub releases](https://github.com/qualcomm/aimet/releases)

## Questions to answer before marking your study complete

1. Why does changing the calibration corpus change the deployed model even when
   the weights and the quantization settings are identical?
2. Why are all four of 128, 512, 1,024, and 4,096 calibrated instead of just
   the longest, or just the shortest — and why is that one coverage argument
   rather than four?
3. What would the range observers have learned if short fixtures had been
   padded, and which two separate objections defeat padding?
4. Why is a tiled sample acceptable as a calibration input but not as a quality
   or latency workload — and what does T10's round-trip check do that T40's
   tiling does not?
5. The four quality prompts are calibrated on and evaluated on. What does that
   do to the measured delta, and what must T41–T43 report because of it?
6. A parity record reads `verdict: partial`,
   `released_for_calibration_on_artifact_identity: true`, and the CLI exited 0.
   What has been established, and what has not?
7. Why is AIMET pinned in `environments/linux-aimet/` instead of as an optional
   extra in `pyproject.toml`, and what single piece of metadata proves it?
8. Which tier-2 licence obligation would force the repository off Apache-2.0,
   and how does recording only identifiers and revisions avoid it?
9. The corpus touches 0.115% of the vocabulary and 83% of its token budget is
   one repeated seed. Which claims does that rule out, and where in the
   contract are those two figures recorded as measurements rather than argued?
