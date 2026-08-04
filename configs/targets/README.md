# Target configurations

Store public Qualcomm device targets, Apple host profiles, and NVIDIA runtime
targets with exact version and capability fields.

Every value in a target selector must be traceable to committed evidence in
this repository. Nothing here is inferred from vendor documentation, carried
over from a different device, or guessed. Each file therefore carries an
`evidence` block naming its sources, a `notes` block recording what was
deliberately *not* copied and why, and a `claim_boundary` block separating
what the file establishes from what it does not.

Each Qualcomm selector is validated by `load_target_config` and
`normalize_target` in `src/slm_lab/deployment/qualcomm/packaging.py`, which
delegate to the committed T30 request validators in
`src/slm_lab/deployment/qualcomm/ai_hub.py`. A selector this repository accepts
is a selector the compile stage would accept. `config_id` always matches the
filename stem. `tests/deployment/qualcomm/test_target_configs.py` enforces both
properties for every `qualcomm-*.json` in this directory. Those validators
require `client.name == "qai-hub"`, so an Apple host profile or an NVIDIA
runtime target committed here is named outside the `qualcomm-` prefix and
validated by its own platform's checks.

## The three Qualcomm Workbench targets

T31 targets the three public Qwen3-0.6B devices listed on the Qualcomm AI Hub
model catalog:

| File | Device selector | Owning task |
|---|---|---|
| `qualcomm-snapdragon-x-elite-crd.json` | `Snapdragon X Elite CRD` | T22 |
| `qualcomm-dragonwing-iq-9075-evk.json` | `Dragonwing IQ-9075 EVK` | T31 |
| `qualcomm-snapdragon-8-elite-qrd.json` | `Snapdragon 8 Elite QRD` | T31 |

All three share the same client version, the same QAIRT runtime version, and
the same compile option string. Those fields are device-independent: the
client version is the installed `qai-hub` distribution, and the QAIRT version
is the resolved `default` from an authenticated `get_frameworks()` query, which
lists hosted frameworks service-wide rather than per device.

## The evidence asymmetry

The three selectors are **not** equally well supported, and their
`claim_boundary` blocks say so in different words. Do not normalize them.

**Snapdragon X Elite CRD** is backed by an *authenticated* run. The
2026-07-25 toy lifecycle authenticated to Workbench, resolved this device, and
completed a compile → inference → profile cycle on it. That selector can
therefore claim
`device_and_runtime_identity_match_committed_T02_evidence`, and it carries an
`os` of `Windows 11` because an authenticated device query returned it.

**Dragonwing IQ-9075 EVK** and **Snapdragon 8 Elite QRD** are backed only by
an *unauthenticated public catalog listing*. Their device names are the exact
strings recorded under `workbench.public_qwen_targets` in
`results/hosts/public-platform-access-2026-07-24.json`, read from the public
Qwen3-0.6B model page on 2026-07-24 without signing in. No authenticated
device query has ever been run against either name. Consequently:

- Neither claims authenticated device confirmation. Both explicitly list
  `an_authenticated_device_query_confirmed_this_selector_resolves` and
  `this_account_can_reach_or_schedule_this_device` under
  `does_not_establish`.
- Neither declares an `os`. `_device` in `ai_hub.py` requires only
  `device.name` and treats `os` as optional, and no committed evidence gives an
  operating system for either Workbench device.
- The Device Cloud catalog in the same 2026-07-24 record lists `Android` for a
  `Snapdragon 8 Elite` / `QRD8750` entry. That was deliberately not copied:
  Device Cloud is a different service with its own device namespace, that
  catalog was also read unauthenticated, and the record itself flags it
  `public_catalog_is_partial`. Copying it would assert a cross-service identity
  this repository has never checked.
- Neither declares attributes. The Hexagon v73, HTP backend, and SoC model 60
  values in the T02 lifecycle were resolved by the service for an X Elite
  compile job and are not transferable.

No AI Hub job has been submitted with any of the three configurations. T31
owns the first real submission. If a device turns out to be unreachable, T31
reports the exact blocker rather than substituting a result from another
target.
