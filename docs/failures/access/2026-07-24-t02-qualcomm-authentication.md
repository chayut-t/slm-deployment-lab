# T02 blocker: Qualcomm authentication unavailable

Date: 2026-07-24
Task: `T02`
Status: bounded external-access blocker

## Intended outcome

Use one minimal free model to prove an AI Hub Workbench compile, inference, and
profile lifecycle, then record current account-visible devices, framework
versions, quota, and Device Cloud X Elite minutes without publishing private
service data.

## Environment

- Repository branch: `codex/T02-platform-access`
- Host kernel: Darwin `24.6.0`, `arm64`
- System Python: `3.9.6`
- Public `qai-hub` release observed: `0.53.0`
- Observation time: 2026-07-24 18:27 UTC

The public client version is discovery evidence only. It was not installed and
is not represented as a tested environment version.

## Minimal reproduction

The shell check:

```bash
command -v qai-hub
python3 -c 'import importlib.util; print(importlib.util.find_spec("qai_hub"))'
```

reported no CLI path and `None`, respectively. No Qualcomm credential variable
or local client configuration was discoverable. Only presence/absence was
checked; no secret value was printed.

Opening Workbench in the available Chrome session redirected to Qualcomm
account sign-in. Opening Device Cloud showed an unauthenticated landing page
with a `Login` action. The retained handoff tabs intentionally contain no
entered credentials.

## Effect

Without an authenticated Workbench session or API token, the task cannot:

- query the account-visible device list or supported framework tags;
- observe user/job quota;
- upload a toy model;
- submit or observe compile, inference, or profile jobs;
- record job-reported QAIRT/runtime versions and target details.

Without Device Cloud login, the task cannot observe the account's minute
balance, complete catalog, live availability, or ability to start an X Elite
session.

Public documentation proves that these service features exist. It does not
prove that this account can use them, so no lifecycle or access acceptance
criterion is marked satisfied.

## Safe unblock procedure

1. Sign in to Qualcomm in the retained Chrome Workbench and Device Cloud tabs.
2. In Workbench, create or retrieve an API token under Account → Settings →
   API Token.
3. Create an ignored, isolated Python 3.11 environment. The host's system
   Python 3.9.6 is below `qai-hub` 0.53.0's Python 3.10 minimum:

   ```bash
   uv venv --python 3.11 .ai-local/envs/qai-hub-0.53.0
   uv pip install \
     --python .ai-local/envs/qai-hub-0.53.0/bin/python \
     "qai-hub==0.53.0"
   .ai-local/envs/qai-hub-0.53.0/bin/qai-hub --version
   ```

   The version command must report `0.53.0` before continuing. Do not install
   into the system Python or add this discovery-only environment to the
   project lockfile.
4. Configure the token locally with the isolated executable:

   ```bash
   read -r -s "T02_QAI_HUB_TOKEN?Qualcomm credential: "
   printf '\n'
   .ai-local/envs/qai-hub-0.53.0/bin/qai-hub configure \
     --api_token "$T02_QAI_HUB_TOKEN"
   unset T02_QAI_HUB_TOKEN
   ```

   This interactive form keeps the literal value out of shell history and
   unsets the task-specific variable immediately afterward. Do not paste the
   token into chat, commit it, or preserve it in a script.
5. Verify authentication with the same isolated executable, keeping raw output
   under `.ai-local/`:

   ```bash
   .ai-local/envs/qai-hub-0.53.0/bin/qai-hub list-devices
   .ai-local/envs/qai-hub-0.53.0/bin/qai-hub list-frameworks
   ```

6. Run one minimal supported compile → inference → profile flow, recording
   sanitized target/version/status/latency evidence and service turnaround
   separately.
7. Observe Device Cloud minutes and X Elite availability without starting a
   session; T32 owns the Qwen/GenieX execution path.

If the account itself is pending approval, record the request date and service
response privately and keep T02 `blocked`. Do not mark it completed until
the toy lifecycle succeeds.

## Privacy boundary

This record excludes the OAuth request URL/state, email address, account or
organization identity, API token, quota identifiers, job identifiers, and raw
service responses. Those values must remain under `.ai-local/` if later
captured.
