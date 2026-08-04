# T31 Qualcomm AI Hub Workbench records

One record lives here at this commit:

- `t31-workbench-run-plan-2026-08-04.json` — the complete three-target run
  plan T31 built, and the exact boundary it stopped at.

**No job was submitted, no service was contacted, and no device evidence
exists for any target.** There is no latency, no memory figure, no placement,
and no numerical comparison in this directory, because none was measured. The
human-readable account is `docs/results/qualcomm/workbench.md`.

## What the record is

A deterministic function of committed inputs and the policy constants in
`src/slm_lab/deployment/qualcomm/workbench.py`. It reads:

- `configs/targets/qualcomm-snapdragon-x-elite-crd.json`,
  `configs/targets/qualcomm-dragonwing-iq-9075-evk.json`, and
  `configs/targets/qualcomm-snapdragon-8-elite-qrd.json`;
- `results/manifests/qnn/packages/S*.json` — the T22 package records;
- `results/manifests/qnn/S*.json` — the T22 candidate manifests, bound by
  digest through the package records;
- `results/manifests/qnn/inspection/S*.json` — the T22 structural
  inspections, which supply the submission ordering key;
- `results/manifests/qnn/parity/S*-ort-cpu.json` — the reference an inference
  stage would be compared against.

Every one of those is bound by repository-relative path and SHA-256, so the
record cannot silently drift from its inputs.

## How to read it

| Block | What it holds |
|---|---|
| `summary` | Counts, and the single first submission with its reason |
| `ordering_policy` | The sort keys for targets, graphs, and the matrix |
| `targets` | Three selectors, each with its device-evidence strength |
| `graphs` | Eight candidate graphs with digests, sizes, shape residues, and the parity reference |
| `stage_contracts` | What `ready` and `pending_predecessor` mean, once, for all entries |
| `plan` | 24 entries in submission order, each with three stages |
| `first_failure_hypothesis` | What is expected to break first, and how to attribute it |
| `submission_boundary` | What is missing before anything may be submitted |
| `cost` | Zero jobs, zero device minutes, US$0.00 |
| `claim_boundary` | What the plan establishes and what it does not |
| `run_observation` | The dated observation of the machine that wrote it |

`targets[]` and `graphs[]` hold everything shared; a `plan[]` entry carries
only what is specific to one (target, graph) pair and joins back on
`target` and `(variant_id, graph_kind)`.

One encoding detail to read carefully: `targets[].device.os` is `""` for the
two catalog-only selectors. That empty string is how `ai_hub._device` encodes
an *absent* operating system, not a declared one — those selector files
deliberately carry no `os` key, because no committed evidence records an
operating system for either device. Only `Snapdragon X Elite CRD`, whose
`Windows 11` came back from an authenticated device query, declares one.

## The two readiness values

`compile` stages read `ready`. Every field the T30 compile stage requires is
fixed by committed inputs, and the request is accepted by the committed T30
validation chain. It is submittable the moment a client and permission exist.

`inference` and `profile` stages read `pending_predecessor`. Both need a
successful compile manifest and the digest of the compiled artifact it
produced, and `ai_hub._load_predecessor` and `ai_hub._compiled_artifact`
enforce that. Neither exists before a real compile job runs, so those fields
are `null` rather than filled with a placeholder. The inference stage
additionally has no input dataset: this repository contains no AI Hub-
compatible HDF5 dataset at this commit.

## Reproducing and checking

```bash
PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py
PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py --check
```

Both run offline, need no artifact root, and import no Qualcomm client.
`--check` re-derives the plan from the committed inputs and fails on the
first differing key path; it also refuses a record that claims a submitted
job or a contacted service.

The optional full check runs the real T30 compile preflight over all 24
compile requests, which needs the assembled T22 packages on the external
artifact root:

```bash
SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python3 \
  scripts/qualcomm/plan_workbench_run.py --preflight
```

It writes each request into `.ai-local/profiles/T31/` — never into the
repository — and refuses any request whose real preflight id differs from the
id the planner derived offline. The committed record was written by that mode,
so `run_observation.preflight` reads `mode: ran`, `requests_validated: 24`,
`all_request_ids_matched_the_plan: true`. Rebuilding without `--preflight`
leaves the plan identical and only changes that observation, which is why
`--check` excludes it.

## The boundary, stated plainly

`request_id` values in this record are the ids the T30 compile stage *would*
record. The first one, `t30-compile-83b8813c19a37ac036ad` for Snapdragon X
Elite CRD / S128 / prefill, is byte-identical to the id already committed in
`results/manifests/qnn/packages/S128.json`, because it is the same request.
That equality is evidence about this repository's own contract and about
nothing else. Qualcomm AI Hub has never seen any of these requests.
