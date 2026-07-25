# Public platform access observation — 2026-07-24

Task: `T02`
Observed: 2026-07-24 18:27 UTC
Status: blocked on Qualcomm authentication

> Historical observation: the Workbench authentication blocker was resolved
> on 2026-07-25. See the
> [completed toy lifecycle](2026-07-25-workbench-toy-lifecycle.md). This file
> preserves the earlier unauthenticated boundary as observed.

## Outcome

The public service surfaces for Qualcomm AI Hub Workbench, Qualcomm Device
Cloud, Google Colab, and Kaggle were reachable. That does **not** establish
account quota or hardware access:

| Platform | Public surface | Account-specific observation | Proven conclusion |
|---|---|---|---|
| AI Hub Workbench | Documentation and model catalog reachable | Browser redirected to Qualcomm sign-in; no local client or configured token was discoverable | Service capability is public; this account's Workbench access, devices, quota, and jobs are unproven |
| Qualcomm Device Cloud | Landing page and partial device catalog reachable | Page displayed `Login`; full catalog and minutes require authentication | Public X Elite/8 Elite catalog entries and free-minutes flow exist; this account's minutes and session access are unproven |
| Google Colab | Home page reachable | An authenticated account UI was visible; identity omitted | Notebook UI access is proven; no NVIDIA runtime was allocated and no GPU model/quota is claimed |
| Kaggle | Notebook catalog and official GPU policy reachable | Page displayed `Sign In` | Public free-GPU program exists; this account's notebook/GPU access and remaining quota are unproven |

The mandatory toy Workbench compile → inference → profile lifecycle was not
submitted. T02 therefore remains `blocked` and its downstream tasks remain
blocked. See the [bounded blocker](../../failures/access/2026-07-24-t02-qualcomm-authentication.md).

## AI Hub Workbench

### Publicly verified service contract

Qualcomm's current documentation says Workbench can compile source models,
perform inference for numerical validation, and profile physical hosted
devices for latency, load time, memory, and compute-unit utilization. It also
states that Workbench is currently free to use.

The public package and hosted-toolchain facts observed on 2026-07-24 were:

| Field | Public value | Evidence boundary |
|---|---|---|
| Latest published `qai-hub` client | `0.53.0`, published 2026-07-20 | PyPI metadata; not installed or authenticated locally |
| Hosted QAIRT versions | `2.45.0`, `2.46.0`, `2.47.0` | Workbench release notes dated 2026-06-22 |
| Latest hosted QAIRT listed | `2.47.0` | Release notes; the account-specific/default selection was not queried |
| Hosted ONNX Runtime | `1.26.0` | Same release notes |
| ONNX Runtime QNN EP | `2.2.0` | Same release notes |
| Quantize Job AIMET | `2.34` | Release notes dated 2026-07-06 |

The default QAIRT tag is deliberately recorded as unknown. Workbench exposes
`default` and `latest` tags through the authenticated `get_frameworks()` /
`qai-hub list-frameworks` interface; public release notes alone do not prove
which installed version `default` resolves to for this account at job time.

The public Qwen3-0.6B model page currently lists these relevant targets:

- `Dragonwing IQ-9075 EVK`
- `Snapdragon 8 Elite QRD`
- `Snapdragon X Elite CRD`

It additionally lists Snapdragon 8 Elite Gen 5 and Snapdragon X2 Elite
targets. Catalog support does not prove that this account can schedule a job,
that quota is available, or that a custom Qwen graph compiles.

### Toy lifecycle state

| Stage | State | Hardware/job evidence |
|---|---|---|
| Compile | not run | none |
| Inference | not run | none |
| Profile | not run | none |

No job ID, URL, target artifact, result dataset, graph latency, load time,
memory value, or NPU-placement claim exists. In particular, public model-page
support is not substituted for a task-owned toy job.

## Qualcomm Device Cloud

The unauthenticated public catalog showed:

- `Snapdragon X Elite`, Compute Reference Design `CRD8380X`, Windows,
  labelled `Unlock Free Minutes`;
- `Snapdragon 8 Elite`, Mobile Reference Design `QRD8750`, Android,
  labelled `Unlock Free Minutes`.

The catalog explicitly said that only a partial list is visible until login.
Consequently, IQ-9075 availability, live device status, this account's minute
balance, and session eligibility were not inferred.

Qualcomm's current FAQ states that Device Cloud is free, that users can request
free minutes, and that new users may receive limited-time pre-provisioned X
Elite minutes. It also states that mobile, compute, and IoT catalog entries use
real devices. These are service-policy facts, not evidence of minutes granted
to this account.

## Free NVIDIA options

### Preferred: Google Colab

An authenticated Colab home page was observed, proving account-level notebook
UI access. No notebook, runtime, or GPU allocation was created. Google's
official FAQ says the free tier can provide GPUs, but GPU type, usage limits,
idle timeout, and availability vary and are not guaranteed or published.

Before T60 uses Colab, its runtime preflight must save:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
```

An acceptable CUDA run must show `CUDAExecutionProvider` and explicit provider
placement; merely opening Colab is not CUDA evidence.

### Alternative: Kaggle

Kaggle's official documentation currently advertises free NVIDIA Tesla P100
GPU access with a weekly quota of 30 hours, sometimes higher depending on
demand and resources. The browser session was not authenticated, so the
account's eligibility and remaining quota were not observed.

The same `nvidia-smi` and ONNX Runtime provider checks are required after a
Kaggle session is actually allocated.

## Paid fallback — documented, not launched

Runpod is the bounded paid fallback if both free services are unavailable.
The read-only inventory command is:

```bash
runpodctl gpu list
```

After recording the current hourly price, CUDA compatibility, region, image
digest, and an explicit user approval, the task owner may adapt this official
command template:

```bash
runpodctl pod create \
  --name slm-lab-cuda \
  --gpu-id "<approved GPU ID>" \
  --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" \
  --gpu-count 1 \
  --container-disk-in-gb 50 \
  --terminate-after 2h
```

This command was **not executed**. Pod creation is a paid external side effect,
even with automatic termination, and requires separate authorization. The
project-wide cloud ceiling remains US$100.

## Sources

- [AI Hub Workbench overview](https://workbench.aihub.qualcomm.com/docs/)
- [Workbench getting started and compute example](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html)
- [Workbench framework/version selection](https://workbench.aihub.qualcomm.com/docs/hub/frameworks.html)
- [Workbench release notes](https://workbench.aihub.qualcomm.com/docs/hub/release_notes.html)
- [`qai-hub` package metadata](https://pypi.org/project/qai-hub/)
- [Qwen3-0.6B public target catalog](https://aihub.qualcomm.com/models/qwen3_0_6b)
- [Device Cloud FAQ](https://qdc.qualcomm.com/support/faq)
- [Device Cloud public catalog](https://qdc.qualcomm.com/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Runpod CLI pod reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
