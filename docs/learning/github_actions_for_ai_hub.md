# GitHub Actions for Qualcomm AI Hub

This guide explains the repository's manual Qualcomm workflow and the security
boundary around it. The workflow is an orchestrator: a GitHub-hosted Linux
runner validates local inputs and calls Qualcomm AI Hub, while the selected
Qualcomm device executes the model. A workflow run is not device latency, and
the queue plus service turnaround time must never be reported as such.

The workflow is intentionally not automatic. It can submit a remote job, use
service quota, and expose a credential if its trust boundary is weakened.
Adding the GitHub secret, approving the protected environment, preparing a
real request bundle, and making the first dispatch are learner-owned hands-on
steps.

## Read the workflow as a program

The file
[`qualcomm-benchmark.yml`](../../.github/workflows/qualcomm-benchmark.yml)
has these layers:

1. `workflow_dispatch` is the only trigger. There is no pull-request, push, or
   schedule trigger that can submit a Qualcomm job.
2. Dispatch inputs select a target, static context, precision, stage, and a
   prior Actions run containing the private request bundle.
3. The `authorize` job has no AI Hub secret. It rejects forks, non-default
   refs, unknown choices, and request bundles not produced by a successful
   same-repository default-branch run.
4. The `submit` job uses the protected `qualcomm-ai-hub` GitHub environment.
   Required reviewers can inspect the dispatch tuple and request-bundle run
   before releasing the job.
5. The runner installs the exact `qai-hub==0.53.0` client, downloads the
   private bundle, and maps the selected stage to one fixed local script.
6. Only then does a short step read `QAI_HUB_API_TOKEN` and write an ephemeral
   mode-`600` client file. The workflow never calls `qai-hub configure`,
   because that client command can print its generated file and token.
7. The local adapter captures SDK output, waits for the bounded job, and
   writes its sanitized manifest. The client file is removed even after a
   failure.
8. A successful run uploads only the sanitized stage manifest for 14 days.
   It does not upload the request bundle, model, dataset, compiled binary,
   inference tensor output, raw profile, token, client file, or SDK log.

GitHub calls the machine that performs the steps a **runner**. A **job** is a
sequence of steps on one runner. An **action** is a reusable step implementation
such as checkout or artifact download. This workflow pins official actions to
immutable commit SHAs rather than mutable tags.

## Why one dispatch runs one stage

T30 made compile, inference, and profile independently restartable:

```text
private request + content-addressed local inputs
  -> scripts/qualcomm/<stage>.py
  -> Qualcomm AI Hub
  -> private raw/result artifact
  -> sanitized public stage manifest
```

Inference and profile start from a downloaded compiled artifact plus the
sanitized compile manifest. They do not resume through an in-memory SDK object
or a public job ID. The Actions workflow preserves that design: `stage` is one
of `compile`, `inference`, or `profile`, and a later stage gets its own request
bundle and dispatch.

The workflow does not reimplement submission, polling, normalization, or
redaction in YAML. Its only execution entry points are:

- `scripts/qualcomm/compile.py`
- `scripts/qualcomm/inference.py`
- `scripts/qualcomm/profile.py`

Read [`scripts/qualcomm/README.md`](../../scripts/qualcomm/README.md) for the
schema-v2 request contract and
[`ai_hub.py`](../../src/slm_lab/deployment/qualcomm/ai_hub.py) for the
validated adapter.

## Inputs and what they prove

| Input | Allowed values | Meaning |
|---|---|---|
| `target` | X Elite, IQ-9075, or 8 Elite repository slugs | Selects a request; actual device identity still comes from the result manifest |
| `context` | 128, 512, 1,024, or 4,096 | Selects a fixed graph/cache shape, not the model's native maximum |
| `precision` | FP16, W8A16, W8A8, or W4A8 | Describes the selected artifact; it does not prove compiler support or achieved arithmetic |
| `stage` | compile, inference, or profile | Runs exactly one restartable adapter |
| `request_artifact_run_id` | Positive Actions run ID | Locates an immutable `qualcomm-request-bundle` artifact from a trusted run |

The workflow validates the input choices again in the shell even though the
GitHub UI presents choice controls. This protects API dispatches and keeps
values out of command construction. The stage is mapped with a fixed `case`
statement; arbitrary input cannot become an executable path.

## Private request-bundle contract

The named Actions artifact is `qualcomm-request-bundle`. After download, the
selected request must be at:

```text
artifacts/qualcomm-request-bundle/
  <target>/<context>/<precision>/<stage>-request.json
```

The bundle includes every local artifact named by that schema-v2 request.
Paths inside the JSON are relative to the checked-out repository and rooted
under `artifacts/qualcomm-request-bundle/`. T30 verifies the declared SHA-256
of each input before initializing the client. Output paths remain under
ignored `artifacts/` storage; the sanitized manifest goes under
`results/qualcomm-actions/`.

Create the bundle only in a reviewed, default-branch workflow that does not
accept executable content from a pull request. Its source workflow is outside
T72 because the producer depends on the artifact/export lane. Record the
source artifact and request hashes there. The Qualcomm workflow accepts only a
successful same-repository default-branch `push` or `workflow_dispatch` run.
Before environment approval, inspect that run and confirm it uploaded the
expected immutable artifact.

Do not put a request JSON in Git. It contains machine-local paths. Do not use a
fork or an unreviewed branch as the bundle producer. An artifact being
downloadable does not make its model, request, or claimed precision trusted.

## Learner setup: stop before adding the secret

First review the workflow security model and complete this checklist:

- [ ] Confirm the only trigger is `workflow_dispatch`.
- [ ] Confirm the authorization job requires the upstream default branch and
  rejects repositories where `fork` is true.
- [ ] Confirm the `submit` job names the `qualcomm-ai-hub` environment.
- [ ] Confirm the AI Hub secret appears only in the client-configuration step.
- [ ] Confirm checkout uses the triggering default-branch SHA with persisted
  Git credentials disabled.
- [ ] Confirm the selected request-bundle run is trusted and its inputs/hashes
  match the intended target, shape, precision, and stage.
- [ ] Check current Qualcomm quota, service terms, expected job count, and any
  cost before approving a run.

Only after that review:

1. In the upstream GitHub repository, create an environment named
   `qualcomm-ai-hub`.
2. Add required reviewers and prevent administrators from bypassing the
   protection if that matches the repository's governance.
3. Add the repository or environment secret `QAI_HUB_API_TOKEN`. Prefer an
   environment secret so it is released only after approval.
4. Never paste the value into an input, issue, pull request, workflow log,
   request bundle, artifact name, or shell command.
5. If the token is exposed, stop the run, revoke/rotate it in Qualcomm AI Hub,
   remove compromised artifacts/logs under the applicable retention policy,
   and record a sanitized incident.

Secrets are not passed to workflows running in a contributor's fork. The
default-branch and `fork == false` checks add another boundary: even a manual
dispatch of this file on an alternate ref does not reach the protected job.
Environment review is the last authorization boundary for quota-consuming
work.

## Dispatch and evidence checklist

On GitHub, open **Actions**, select **Qualcomm AI Hub benchmark**, and choose
**Run workflow**. Keep the branch selector on the repository's default branch.
Select the four workload inputs and paste only the trusted producer run's
numeric ID.

Before approving the protected job:

- Compare the selected tuple with the request JSON's public-safe projection.
- Confirm the stage is necessary; retrying a failed job submits a new job
  because T30 sets SDK retry to false.
- For inference/profile, confirm compiled-artifact and predecessor-manifest
  hashes match the prior compile evidence.
- Confirm the bundle has no executable hook and the producer run used reviewed
  default-branch code.

After success, download the manifest artifact and check:

- exact `qai-hub` client, requested QAIRT, target selector, and observed device;
- source, predecessor, request, and result hashes;
- graph latency, load time, peak memory, placement, and warning count when the
  selected stage exposes them;
- service turnaround is labeled separately from device execution;
- unobserved runtime or hardware fields remain null rather than inferred.

Raw profiles and result tensors remain private. Sanitize and normalize them
through the T30 path before publishing small evidence. Never publish service
job IDs/URLs, account/project identity, quota identifiers, filesystem paths,
or raw service responses.

## Failure interpretation

An authorization job that is skipped usually means the dispatch did not use
the upstream default branch. A provenance failure means the run ID was
missing, failed, came from another repository/branch, or used an unsupported
event. A missing-request failure means the bundle layout does not contain the
selected tuple.

Adapter failures intentionally suppress external details. Inspect raw service
information only in the private runner log/artifact boundary, then create a
minimal sanitized failure record under `docs/failures/` if it is
reproducible. Do not weaken output capture merely to expose a diagnostic.

## What T72 verifies—and does not

Offline tests verify YAML structure, enumerated inputs, default-branch/fork and
environment gates, minimal permissions, fixed script delegation, pinned
actions/client, secret placement, cleanup, and sanitized-only upload.

T72 does **not** add or inspect a real secret, create the protected GitHub
environment, produce a real request bundle, dispatch Actions, submit a
Qualcomm job, or validate hardware output. Those hands-on actions require the
learner's authentication and approval. Until performed, the workflow is
structurally validated but externally unverified.
