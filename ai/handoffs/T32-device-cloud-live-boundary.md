# T32 Device Cloud live-capture handoff

Date: 2026-07-28
Task: `T32`
Status: completed

## Outcome

A free Qualcomm Device Cloud session was captured with the T32 fixed prompt
and ready-made Qwen3-0.6B GenieX/`llama.cpp` route. The private validator
currently accepts device identity, runtime identity, placement, multi-token
generation, all eight timing boundaries, synchronization, model provenance,
and zero paid-resource use.

The Device Cloud UI labels information about the allocated device
confidential under its access agreement, and research found no express public
permission for the complete live-device record. The learner directed
publication of the generic reproducibility setup and aggregate latency
measurements. Allocated-device evidence, observed placement proof, exact
installed versions, session/account identifiers, logs, manifests, and
evidence digests remain private. This learner-selected boundary is not
presented as a legal determination.

## Reproducible public route

- Qualcomm AI Hub Models v0.58.0 publishes the
  `Qwen3-0.6B`/`geniex_llamacpp`/`Q4_0` route.
- The direct GenieX registry route is also supported by the capture tooling.
- The public Q4_0 file is pinned to immutable Hugging Face revision
  `272676c9e0eb9f33a7719ba3d27482fbb445e801` and SHA-256
  `33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb`.
- `measure_qwen_boundaries.ps1` applies the model chat template with thinking
  disabled, invokes the exact installed `llama.cpp` vocabulary tokenizer, and
  passes the resulting IDs to one persistent generation call.

The direct registry source must not be relabelled as a
`qai-hub-models==0.58.0` download. The rolling registry alias is not immutable;
the public revision and artifact hash pin the actual model bytes.

## Timing and claim boundaries

The record separates artifact open/map, model load, tokenization, prefill,
first decode, remaining decode, generation total, and complete request wall
time. The aggregate values are published in
`docs/results/qualcomm/device-cloud.md`. Complete request includes the
tokenizer subprocess and probe overhead from artifact open/map through
generation completion. Device allocation, SSH transport, downloads, and
interactive turnaround are excluded.

This single acceptance capture is not a distribution benchmark. The
ready-made GenieX/`llama.cpp` result does not prove the custom QNN/QAIRT
static-graph path.

## Private/public evidence split

- Private evidence root: `.ai-local/profiles/T32/qdc-2026-07-28/`.
- Raw transcripts, environment identity, exact output, detailed timing
  records, checksums, and sanitized manifest remain ignored.
- Public result: `docs/results/qualcomm/device-cloud.md`.
- Public live evidence is limited to the learner-approved generic setup and
  aggregate latency measurements.
- T31 owns `results/processed/qualcomm/`; T32 writes no result there.
- No paid resource was used.

## Completion

The final private evidence is frozen and checksum-verified. Engineering and
publication work are complete. Fresh post-publication review and re-review
approved the generic-setup, aggregate-latency, and privacy boundaries.
The learner explicitly confirmed the timing, GenieX-versus-custom-QNN, and
private/public evidence-split debrief on 2026-07-28.
