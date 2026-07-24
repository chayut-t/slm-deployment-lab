# Manifests

Versioned JSON Schemas and validation for durable evidence belong here.

T01 defines:

- `artifact-v1.schema.json`: the stable provenance envelope from the project
  plan and T00 version policy.
- `host-v1.schema.json`: sanitized host, tool-version, and storage facts.

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
