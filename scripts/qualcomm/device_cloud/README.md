# T32 Device Cloud capture

These commands prepare and sanitize the live Qwen3-0.6B baseline. They do not
authenticate, allocate minutes, or create a Device Cloud session.

## Learner-owned session boundary

1. Sign in at `https://qdc.qualcomm.com/`.
2. Confirm that the account has free minutes. Do not activate paid resources.
3. Start an interactive Compute session on the Snapdragon X Elite Compute
   Reference Design. Record the exact catalog code, OS, chipset, memory, and
   session time privately.
4. Use Remote Desktop or the browser terminal supplied by Device Cloud.
5. Copy this repository or just this directory to the device.

The public catalog observed on 2026-07-27 showed Snapdragon X Elite,
Compute Reference Design, catalog code `CRD8380X`, and Windows. Treat that only
as catalog discovery; the live capture must record the allocated device.

## Ready-made asset and generation loop

The Qualcomm AI Hub Models v0.58.0 model card publishes this exact fetch
contract:

```powershell
python -m pip install "qai-hub-models-cli==0.58.0"
qai-hub-models fetch Qwen3-0.6B `
  --runtime geniex_llamacpp `
  --precision q4_0
```

Install the current Windows ARM64 GenieX release using Qualcomm's official
installer, then pass the directory containing the downloaded `.gguf` file to:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts\qualcomm\device_cloud\run_qwen_baseline.ps1 `
  -BundlePath C:\PRIVATE\PATH\TO\QWEN3_0_6B
```

The script registers the local bundle, explicitly chooses `--compute npu`,
disables thinking, fixes seed 0, caps generation at 32 tokens, and records raw
output under ignored `.ai-local/profiles/T32/`. The raw transcript can contain
private paths and must never be committed.

At the interactive prompt, use exactly:

```text
Reply with five consecutive integers beginning at 41, separated by spaces.
```

Confirm the response contains at least two generated tokens and is a valid
answer before setting `valid_multi_token_output_confirmed` to `true`. Record
the UTF-8 SHA-256 of the exact prompt and exact output rather than copying
their text into the public manifest.

## Timing boundaries

Do not infer missing metrics. Capture each boundary explicitly:

| Field | Boundary |
|---|---|
| `artifact_load` | Open/map the GGUF artifact; excludes runtime/model initialization. |
| `model_load` | Initialize GenieX/llama.cpp after artifact availability. |
| `tokenization` | Convert the fixed prompt to input IDs; excludes model execution. |
| `prefill` | Process all prompt IDs and populate the KV cache. |
| `first_decode` | Produce and materialize the first output token after prefill. |
| `decode` | Produce the remaining output tokens in the persistent loop. |
| `generation_total` | Prefill through materialization of the final output token. |
| `request_total` | Artifact load through the final output token. |

Use one of these source labels for every timing:

- `geniex_runtime_report`
- `instrumented_host_clock`
- `derived_from_runtime_counters`

If the installed GenieX build does not expose one boundary, stop and record
the limitation in the task handoff. Do not derive prefill by subtracting an
unrelated Workbench graph latency or label service/session turnaround as
device execution.

## Sanitize the live record

Copy the template into ignored private storage and fill it only with observed
values:

```powershell
Copy-Item `
  scripts\qualcomm\device_cloud\private-capture.template.json `
  .ai-local\profiles\T32\capture.private.json
```

Back on the repository host:

```bash
PYTHONPATH=src python3 scripts/qualcomm/device_cloud/capture.py \
  --capture .ai-local/profiles/T32/capture.private.json \
  --manifest results/processed/qualcomm/device-cloud-qwen3-0.6b.json
```

The validator fails closed unless the record proves multi-token generation,
provides all timing boundaries with source labels, names observed NPU/HTP
placement evidence, identifies the exact runtime and device, uses no paid
resource, and contains no URL-, account-, session-, email-, or path-like
public text.

## Claim boundary

This baseline proves a persistent device-side GenieX/llama.cpp generation
loop on the recorded Snapdragon X Elite. It does not prove the custom static
QNN graph path, Workbench graph latency, QAIRT bundle execution, or an
eNPU/LPAI claim.
