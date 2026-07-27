# Device Cloud Qwen3-0.6B baseline

Task: `T32`
Status: blocked at learner authentication and live-device execution
Last checked: 2026-07-27

## Current outcome

The reproducible capture and privacy validation path is ready, but no
Snapdragon X Elite generation result is claimed. The Device Cloud public
catalog was reachable without authentication and showed:

| Field | Public catalog observation |
|---|---|
| Product | Snapdragon X Elite |
| Form factor | Compute Reference Design |
| Catalog code | `CRD8380X` |
| OS label | Windows |
| Access state | Login or Sign Up required |
| Allocation label | Unlock Free Minutes |

This is catalog discovery, not an allocated-device environment manifest.
Account minutes, current session availability, exact allocated hardware,
runtime versions, generation output, and timings remain unobserved.

## Exact external boundary

The learner must sign in at
[Qualcomm Device Cloud](https://qdc.qualcomm.com/), confirm free-minute access,
and start the Snapdragon X Elite interactive session. No paid resource is
authorized. After login, the learner should use the
[session capture procedure](../../../scripts/qualcomm/device_cloud/README.md)
and participate in or review the run.

T32 must remain incomplete until the resulting evidence demonstrates:

- valid Qwen3-0.6B multi-token output on the allocated device;
- exact device and GenieX runtime identity;
- observed NPU/HTP placement evidence;
- separately sourced artifact load, model load, tokenization, prefill,
  first-decode, remaining-decode, generation-total, and request-total timings;
- zero paid-resource use; and
- a sanitized public manifest with raw session data kept private.

## Prepared reproducible route

The current Qualcomm AI Hub Models v0.58.0 card publishes a ready asset for
`Qwen3-0.6B` with runtime `geniex_llamacpp` and precision `q4_0`. The prepared
PowerShell workflow fetches that asset, registers its local GGUF directory
with GenieX, explicitly chooses the NPU compute target, and enters a persistent
interactive generation loop. Raw output is written only below ignored
`.ai-local/profiles/T32/`.

GenieX exposes two distinct runtime families:

- `llama_cpp` consumes GGUF and can use Hexagon NPU, Adreno GPU, or CPU;
- `qairt` consumes precompiled Qualcomm AI Engine Direct bundles and is
  NPU-only.

This T32 ready-made baseline uses the first route. It cannot substitute for
the later custom QNN/QAIRT graph evidence, and no Workbench single-graph
measurement may be relabeled as its end-to-end timing.

## Sources checked

- [Qwen3-0.6B model card and v0.58.0 fetch command](https://github.com/qualcomm/ai-hub-models/blob/v0.58.0/src/qai_hub_models/models/qwen3_0_6b/README.md)
- [Pinned `qai-hub-models-cli` 0.58.0 package](https://pypi.org/project/qai-hub-models-cli/0.58.0/)
- [GenieX CLI quickstart](https://geniex.aihub.qualcomm.com/en/run/cli/quickstart)
- [GenieX model/runtime and local GGUF guidance](https://geniex.aihub.qualcomm.com/en/models/supported)
- [Qualcomm Device Cloud](https://qdc.qualcomm.com/)

## Learner checkpoint

- [ ] Sign in, confirm free minutes, and start the X Elite session.
- [ ] Explain which timings belong to loading, tokenization, prefill, first
  decode, remaining decode, and the complete request.
- [ ] Explain why this GenieX/llama.cpp result does not prove the custom QNN
  path.
- [ ] Review the sanitized manifest and private/public evidence split before
  T32 is accepted.
