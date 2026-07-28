# Device Cloud Qwen3-0.6B baseline

Task: `T32`
Status: completed
Last checked: 2026-07-28

## Current outcome

A live Device Cloud capture is preserved in ignored private storage. The
session UI states that information about the allocated device is confidential
under the Device Cloud access agreement. A review of Qualcomm's public terms
and Device Cloud documentation found no express permission to publish the
complete live-device record.

On 2026-07-28, the learner selected a narrow publication boundary: publish the
aggregate latency measurements and generic reproducibility setup produced by
the learner-controlled benchmark, but withhold allocated-device evidence,
observed placement proof, exact installed software versions, session/account
identifiers, raw logs, manifests, and evidence digests. This learner decision
is not presented as a legal determination or a general license to republish
Device Cloud information.

The public catalog observation remains:

| Field | Public catalog value |
|---|---|
| Product | Snapdragon X Elite |
| Form factor | Compute Reference Design |
| Catalog code | `CRD8380X` |
| OS label | Windows |
| Access label | Unlock Free Minutes |

Catalog discovery is not live-device evidence. The learner explicitly
confirmed the T32 timing, claim-boundary, and evidence-split debrief on
2026-07-28.

## Benchmark setup

| Setting | Published configuration |
|---|---|
| Service | Qualcomm Device Cloud interactive session |
| Public target | Snapdragon X Elite Compute Reference Design (`CRD8380X`) |
| OS family | Windows 11 |
| Model | Qwen3-0.6B, GGUF `Q4_0` |
| Model artifact | Immutable revision `272676c9e0eb9f33a7719ba3d27482fbb445e801`, SHA-256 `33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb` |
| Runtime route | GenieX `llama_cpp` |
| Requested compute | NPU, configured as `HTP0`; observed-placement evidence remains private |
| Prompt | `Reply with five consecutive integers beginning at 41, separated by spaces.` |
| Prompt formatting | Model chat template, thinking disabled, generation prompt appended |
| Formatted input | 27 tokens |
| Generation configuration | Maximum 32 tokens, temperature 0, top-p 1, top-k 0, minimum-p 0, repetition penalty 1, seed 0 |
| Accepted output | 18 generated tokens, EOS termination, exact-answer validation passed |
| Measurement scope | One persistent device-side generation call with a runtime-completion fence |

## Published single-run latency

These values come from one validated acceptance capture. They are not a
distribution benchmark and have no variance, percentile, or repeatability
claim.

| Boundary | Latency (ms) | Definition |
|---|---:|---|
| Artifact open/map | 6.3797 | Open/map the GGUF before runtime initialization. |
| Model load | 1067.7551 | Create the GenieX/`llama.cpp` model after artifact availability. |
| Tokenization | 36.6186 | Convert the exact formatted prompt to input IDs. |
| Prefill | 69.9470 | Process the prompt IDs and populate the KV cache. |
| First decode | 0.5520 | Produce the first output token after prefill. |
| Remaining decode | 292.7550 | Produce the later output tokens in the persistent loop. |
| Generation total | 363.2540 | Prefill through the final output token. |
| Complete request | 5325.8185 | Host wall time from artifact open/map through generation completion, including instrumentation and orchestration overhead. |

No Device Cloud allocation, SSH transport, download, or interactive
turnaround time is relabelled as device execution. Complete request is an
observed wall clock and must not be reconstructed by adding the individually
instrumented stages.

## Prepared reproducible route

The Qualcomm AI Hub Models v0.58.0 card publishes a ready asset for
`Qwen3-0.6B` with runtime `geniex_llamacpp` and precision `q4_0`. The prepared
probe also supports the direct GenieX model-registry route, applies the model
chat template with thinking disabled, tokenizes the exact formatted prompt
with the installed `llama.cpp` vocabulary, and passes those token IDs to one
persistent generation call.

GenieX exposes two distinct runtime families:

- `llama_cpp` consumes GGUF and can use Hexagon NPU, Adreno GPU, or CPU;
- `qairt` consumes precompiled Qualcomm AI Engine Direct bundles and is
  NPU-only.

This T32 ready-made baseline uses the first route. It cannot substitute for
the later custom QNN/QAIRT graph evidence, and no Workbench single-graph
measurement may be relabelled as its end-to-end timing.

Raw evidence and the sanitized intermediate manifest stay below ignored
`.ai-local/profiles/T32/`. T31 owns `results/processed/qualcomm/`, so T32
writes no result there. Only the generic benchmark setup and aggregate latency
column above are published from the live record; the normalized record and
its evidence digests remain private.

## Public model provenance

The public Q4_0 file at immutable Hugging Face revision
`272676c9e0eb9f33a7719ba3d27482fbb445e801` has SHA-256
`33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb`.
This pins the model bytes independently of the rolling GenieX registry alias.

## Sources checked

- [Qwen3-0.6B model card and v0.58.0 fetch command](https://github.com/qualcomm/ai-hub-models/blob/v0.58.0/src/qai_hub_models/models/qwen3_0_6b/README.md)
- [Pinned `qai-hub-models-cli` 0.58.0 package](https://pypi.org/project/qai-hub-models-cli/0.58.0/)
- [GenieX CLI quickstart](https://geniex.aihub.qualcomm.com/en/run/cli/quickstart)
- [GenieX model/runtime and local GGUF guidance](https://geniex.aihub.qualcomm.com/en/models/supported)
- [GenieX source repository](https://github.com/qualcomm/GenieX)
- [Immutable public Qwen3-0.6B Q4_0 artifact](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/272676c9e0eb9f33a7719ba3d27482fbb445e801/Qwen3-0.6B-Q4_0.gguf)
- [Qualcomm Device Cloud](https://qdc.qualcomm.com/)
- [Qualcomm Terms of Use](https://www.qualcomm.com/site/terms-of-use)

## Learner checkpoint

- [x] Sign in, confirm free minutes, and start the X Elite session.
- [x] Review the publication boundary and choose to publish the generic setup
  and aggregate latency measurements while withholding the remaining
  live-device record.
- [x] Explain which timings belong to loading, tokenization, prefill, first
  decode, remaining decode, and the complete request.
- [x] Explain why this GenieX/llama.cpp result does not prove the custom QNN
  path.
- [x] Review the sanitized manifest and private/public evidence split before
  T32 is accepted.
