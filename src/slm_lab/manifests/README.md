# Manifests

Versioned JSON Schemas and validation for durable evidence belong here.

T01 defines:

- `artifact-v1.schema.json`: the stable provenance envelope from the project
  plan and T00 version policy.
- `host-v1.schema.json`: sanitized portable host, tool-version, storage, and
  platform-extension facts for Apple/macOS, NVIDIA/Linux, and hosted Qualcomm.

Validate JSON or YAML:

```bash
slm-lab-validate-manifest artifact path/to/artifact.json
slm-lab-validate-manifest host path/to/host.json
```

Every artifact field remains required, including fields that do not apply to a
particular stage. Such fields use JSON `null` with the reason captured in the
surrounding task evidence; they must not be omitted or guessed. Platform,
profile, benchmark, and cost extensions may add fields without weakening this
base envelope.

Host manifests use the `platform` discriminator. Only the matching
`platform_details.apple`, `platform_details.nvidia`, or
`platform_details.qualcomm` object is accepted. Portable hardware facts may be
null when a hosted service does not expose them; Apple neural-engine and
unified-memory fields are never required from Linux or Qualcomm evidence.
Likewise, `artifact_storage.storage_kind` distinguishes a mounted local root
from a hosted service with no exposed filesystem.

Every field named as an exact version rejects floating labels, comparison
operators, compatible-release syntax, comma ranges, and wildcards. A checked
but unavailable tool uses `version: null`, a non-verified status, and a
non-empty reason.
