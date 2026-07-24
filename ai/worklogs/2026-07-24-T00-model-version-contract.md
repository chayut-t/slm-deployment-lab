# T00: Model Version Contract

Date: 2026-07-24
Task: `T00`
Visibility: `public`
Status: completed

## Outcome

Pinned the Qwen3-0.6B model and tokenizer to one immutable upstream commit,
recorded reproducible metadata and chat-template hashes, and accepted a formal
ADR that freezes model behavior and the toolchain-version evidence policy.
Downstream tasks now have a machine-readable source contract and an offline
regression test, with an opt-in live provenance check.

## Changes

- Added `docs/decisions/0001-model-and-version-pins.md` with the source,
  tokenizer, thinking-mode, special-token, scope, fallback, and version-capture
  decisions.
- Added JSON-compatible YAML at `configs/models/qwen3-0.6b.yaml` with the full
  model/tokenizer revision, metadata hashes, architecture facts, project
  contexts, platform priority, and mandatory evidence fields.
- Kept the stable artifact-manifest field names exactly aligned with
  `docs/project/plan.md`; platform manifests extend rather than rename them.
- Added `tests/repo/test_model_contract.py` for immutable-pin, digest,
  special-token, plan-scope, artifact-schema, and toolchain-field regression
  checks.
- Added an opt-in network test that re-fetches the two small public metadata
  files at the pinned commit and recomputes all three recorded hashes.
- Completed and archived the T00 execution plan.
- Updated the coordinator handoff with T00 evidence and the four newly ready
  tasks.

## Verification

- Command:
  `git ls-remote https://huggingface.co/Qwen/Qwen3-0.6B refs/heads/main`
- Result: resolved
  `c1899de289a04d12100db370d81485cdf75e47ca` on 2026-07-24.
- Command:
  `SLM_LAB_VERIFY_UPSTREAM=1 python3 -m unittest tests.repo.test_model_contract.ModelContractTests.test_pinned_upstream_metadata`
- Result: passed; raw `config.json`, raw `tokenizer_config.json`, and decoded
  chat-template hashes reproduced from the immutable revision.
- Command: `python3 -m unittest discover -s tests -p 'test_*.py'`
- Result: 18 tests discovered; 17 passed and the opt-in network test was
  skipped as intended. The opt-in test passed separately above.
- Command: `python3 scripts/ai/render_task_status.py --check`
- Result: task graph valid and generated status current.
- Command: `python3 scripts/repo/check_hygiene.py --all`
- Result: passed for tracked and untracked public files.
- Command: `git diff --check`
- Result: passed.

## Decisions and evidence

- The model and tokenizer share commit
  `c1899de289a04d12100db370d81485cdf75e47ca`; no floating source ref appears
  in the load contract.
- T00 defines exact version-field and provenance requirements. T01 retains
  responsibility for selecting tested environment versions and creating
  `uv.lock`.
- Raw completion is the canonical deterministic validation interface. Chat
  fixtures must explicitly disable thinking and record the pinned template
  hash.
- An independent subagent reproduced the upstream revision and hashes, found
  and prompted correction of an artifact-field naming mismatch, and passed the
  corrected contract and tests on follow-up review.

## Risks and limitations

- Model weights were not downloaded or hashed in T00; later artifact manifests
  must record source artifact hashes when weights are materialized.
- Exact environment compatibility and public service versions are unproven
  until T01 and T02 complete.
- No model inference or hardware execution was required or claimed for T00.

## Follow-up

- Newly unblocked tasks: T01, T02, T03, and T10.
- Recommended next action: start those tasks from the commit that completes
  T00, using separate task branches/worktrees and respecting T01/T02 resource
  locks.
