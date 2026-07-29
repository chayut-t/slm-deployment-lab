# Development setup

This guide takes a new checkout from an empty machine to a working common
development environment. The common environment is intentionally portable; the
MLX, CUDA, AIMET, and Qualcomm environments are separate because they have
different host and runtime constraints.

The commands below use a POSIX shell on macOS or Linux. On Windows, install Git
and `uv` inside WSL2, clone into the WSL2 Linux filesystem rather than a
Windows-mounted path, and run the common repository workflow there. The
versioned setup and hook scripts are shell scripts.

## 1. Install prerequisites

Install:

- Git.
- `curl` or another way to install `uv`.
- Enough local space for the source checkout and `.venv/`.
- Optional external storage for model weights, compiler outputs, and traces.

The repository pins CPython 3.11.13 and `uv` 0.11.32 in
[`environments/common-toolchain.json`](environments/common-toolchain.json).
Check for an existing installation before installing anything:

```bash
command -v git
git --version
command -v uv
uv --version
```

If `uv --version` already reports `0.11.32`, keep it and skip installation.
Otherwise install that exact version with the
[official standalone installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh
```

Restart the shell only if the installer updates `PATH`, then confirm:

```bash
uv --version
```

`uv --version` should report `0.11.32`.

Check Git identity without changing it:

```bash
git config --get user.name
git config --get user.email
```

If either value is absent, configure the correct identity before a task that
authorizes local commits. Do not invent an identity or commit merely as part of
setup.

## 2. Clone and create the locked environment

```bash
git clone https://github.com/chayut-t/slm-deployment-lab.git
cd slm-deployment-lab
```

Use `git@github.com:chayut-t/slm-deployment-lab.git` instead when your GitHub
SSH key is configured, or substitute your fork URL.

Check for the pinned Python before downloading it:

```bash
uv python find 3.11.13
```

If that command prints an existing Python path, do not reinstall it. If it
reports that no interpreter was found, run:

```bash
uv python install 3.11.13
```

Next, inspect the existing project environment before changing it:

```bash
test -x .venv/bin/python && .venv/bin/python --version
uv sync --extra dev --locked --dry-run
```

If the dry run reports that the environment is already satisfied, skip the
real sync. If `.venv/` is missing, uses the wrong Python, or the dry run lists
required additions or upgrades, run:

```bash
uv sync --extra dev --locked
```

If the dry run proposes removing packages you expected to keep, stop and
identify which optional extra or task-specific environment installed them
before syncing.

`uv` reads [`.python-version`](.python-version), creates the ignored `.venv/`,
and installs the project and exact locked development dependencies. Use
`uv run --locked --no-sync ...` for repository commands after validation; it
selects `.venv/` without requiring manual activation or changing the lockfile
or environment. Do not delete or recreate a healthy `.venv/`; both Python and
downloaded dependencies may already be usable.

`uv sync` is exact by default: it can remove packages belonging to optional
extras omitted from the command. Match the environment's intended extras when
checking or syncing it. For example, use
`uv sync --extra dev --extra tokenizer --locked --dry-run` for an environment
that intentionally includes the tokenizer extra. Platform-specific stacks use
their separate task environments described below rather than the common
`.venv/`.

Do not replace `--locked` with a lock update during setup. Dependency changes
must be intentional changes to `pyproject.toml` and `uv.lock`.

## 3. Configure local storage and secrets

Large files never belong in Git. Choose an absolute directory outside the
checkout for model weights, download caches, ONNX data, compiled binaries, and
full traces. The primary project machine uses
`/Volumes/T9/slm-deployment-lab`, but every developer may use a different
absolute path.

Preserve an existing `.env`; it may contain machine paths or secrets:

```bash
if test -e .env; then
    echo ".env already exists; preserving it"
else
    cp .env.example .env
fi
```

Edit `.env` only if its existing values are not correct for this host. The
committed `/Volumes/T9/...` examples are specific to the primary Mac and must
be replaced on Linux, WSL2, or another machine.

Load the variables into the current shell, then require an absolute artifact
path:

```bash
set -a
. ./.env
set +a

case "$SLM_LAB_ARTIFACT_ROOT" in
    /*) ;;
    *) echo "SLM_LAB_ARTIFACT_ROOT must be absolute" >&2; exit 1 ;;
esac
```

The repository does not automatically load `.env`. Repeat the load step in a
new shell, or configure a trusted local environment loader. Keep credentials
only in the ignored `.env`, an approved secret store, or a service-specific
credential store. Never add tokens to `.env.example`, committed configuration,
notebooks, logs, or worklogs.

For local, non-removable storage, create the configured directories with:

```bash
mkdir -p "$SLM_LAB_ARTIFACT_ROOT" "$HF_HOME"
```

For removable storage, first verify that the volume is mounted. Create only
subdirectories on the mounted volume; do not create the mount point itself
while the device is absent, because that can silently place large artifacts on
the internal disk. External storage remains optional for lightweight source
and test work. If `SLM_LAB_ARTIFACT_ROOT` does not exist, local bootstrap
continues with a warning and does not create `artifacts`.

## 4. Reconstruct ignored local state and enable hooks

After loading `.env`, inspect any existing hook and artifact state:

```bash
git config --get core.hooksPath || true
readlink artifacts || true
```

Then run the versioned installer from the repository root:

```bash
uv run --locked --no-sync scripts/setup/install_git_hooks.sh
```

This command does not sync or reinstall dependencies. The installer itself is
idempotent: it preserves an existing valid registry and local README, fills in
any missing local directories, and refuses to overwrite or silently retarget
an existing `artifacts` path. It also:

- configures `core.hooksPath` as `.githooks`;
- creates the ignored `.ai-local/` workspace;
- initializes the private shared task/session registry from the committed
  schema-v2 template; and
- creates the ignored `artifacts` symlink when the configured artifact root
  exists.

Verify the result:

```bash
git config --get core.hooksPath
uv run --locked --no-sync python scripts/ai/session_registry.py validate
uv run --locked --no-sync python scripts/ai/session_registry.py path
```

The first command should print `.githooks`. The registry path should resolve
under the primary checkout even when the command is run from a linked
worktree. The post-checkout hook repeats local bootstrap so new branches and
worktrees retain the same local coordination structure. Both versioned hooks
prefer `.venv/bin/python`, then the uv-managed Python 3.11.13 interpreter, and
use a system `python3` only as a final fallback.

## 5. Know the hidden local files

These paths are intentionally absent from a fresh clone and ignored by Git:

| Path | Created by | Purpose |
|---|---|---|
| `.venv/` | `uv sync` | Locked common Python environment |
| `.env` | Developer | Machine-specific paths and secrets |
| `.ai-local/README.md` | Setup installer | Privacy and publishing boundary |
| `.ai-local/inputs/` | Setup installer | Private task inputs and feedback |
| `.ai-local/plans/` | Setup installer | Draft execution plans |
| `.ai-local/tasks/thread-registry.yaml` | Setup installer | Real private agent/session ownership |
| `.ai-local/handoffs/` | Setup installer | Unsanitized local handoffs |
| `.ai-local/worklogs/` | Setup installer | Raw or private worklogs |
| `.ai-local/profiles/` | Setup installer | Unsanitized cloud/profile output |
| `.ai-local/scratch/` | Setup installer | Temporary experiments |
| `artifacts` | Setup installer | Symlink to `SLM_LAB_ARTIFACT_ROOT` |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | Development tools | Disposable caches |
| `site/` | MkDocs | Generated documentation site |

Task-specific environments and private model inputs may add directories such
as `.ai-local/envs/` and `.ai-local/models/`; keep them ignored. Optional
Claude Code settings (`CLAUDE.local.md`, `.claude/settings.local.json`, and
`.claude/worktrees/`) are also local-only.

Before committing, use `git status --short --ignored` to confirm that private
and large paths remain ignored. Never force-add anything under `.ai-local/` or
`artifacts`.

## 6. Run the health checks

Run the common checks from the repository root:

```bash
uv run --locked --no-sync ruff check .
uv run --locked --no-sync pytest
uv run --locked --no-sync mkdocs build
uv run --locked --no-sync python scripts/ai/render_task_status.py --check
uv run --locked --no-sync python scripts/repo/check_hygiene.py --all
git status --short --ignored
```

`--locked` prevents an implicit lockfile update and `--no-sync` prevents these
checks from changing the already-validated environment.

Expected outcomes:

- lint and tests pass, with platform-dependent tests explicitly skipped when
  their optional runtime is unavailable;
- MkDocs builds the ignored `site/` directory;
- generated task status is current;
- repository hygiene passes; and
- only intentional work is untracked or modified, while local state is shown
  with `!!`.

The current normal MkDocs build succeeds but emits nine known warnings for
repository-relative links that point outside `docs/`. Strict mode promotes
those pre-existing warnings to failures; it is not yet a clean health gate.

The pre-commit hook repeats staged task-status, privacy, secret, policy-adapter,
and large-file checks. It does not replace the full checks above.

## 7. Add only the environment needed for a task

The common environment deliberately excludes heavyweight or
platform-constrained stacks:

- Tokenizer fixture work: `uv sync --extra dev --extra tokenizer --locked`.
- Apple M4 / MLX-LM: follow
  [`environments/macos-m4/README.md`](environments/macos-m4/README.md).
- Linux / NVIDIA CUDA: follow
  [`environments/linux-cuda/README.md`](environments/linux-cuda/README.md).
- Linux / AIMET: follow
  [`environments/linux-aimet/README.md`](environments/linux-aimet/README.md).
- Qualcomm hosted and Device Cloud work: follow the relevant task definition,
  environment evidence, and [`scripts/qualcomm/README.md`](scripts/qualcomm/README.md).

Do not install these stacks into the common `.venv/` unless their owning task
explicitly changes the shared environment contract. Record exact host, driver,
compiler, runtime, model, and device revisions for measured results.

## 8. Start repository work safely

Before substantial work, read:

1. [`AGENTS.md`](AGENTS.md) for repository policy.
2. [`docs/project/plan.md`](docs/project/plan.md) for scope and platform
   priorities.
3. [`docs/project/learning-checkpoints.md`](docs/project/learning-checkpoints.md)
   for learner checkpoints.
4. [`ai/tasks/task_graph.yaml`](ai/tasks/task_graph.yaml) and the selected task
   definition under `ai/tasks/definitions/` for dependencies, ownership,
   outputs, and acceptance criteria. New definitions start from
   [`task.template.yaml`](ai/tasks/definitions/task.template.yaml).

Project tasks use explicit task branches and Git worktrees. Private session IDs
stay in `.ai-local/tasks/thread-registry.yaml`; public task state stays in the
task graph. See
[`docs/agentic/dual-agent-setup.md`](docs/agentic/dual-agent-setup.md) for the
claim, worktree, registry, review, and handoff workflow.

## Troubleshooting

### `uv` cannot find Python 3.11.13

Run `uv python install 3.11.13` from the checkout, then repeat
`uv sync --extra dev --locked`. The committed `.python-version` must remain
`3.11.13`.

### Bootstrap reports an artifact-path mismatch

Inspect `readlink artifacts` and the current value of
`SLM_LAB_ARTIFACT_ROOT`. The installer refuses to replace a real directory or
silently retarget an existing symlink. Preserve any contents, then make the
environment value and symlink target agree before rerunning setup.

### Bootstrap reports no artifact root

Create the directory configured in `.env`, load `.env` into the shell, and
rerun
`uv run --locked --no-sync scripts/setup/install_git_hooks.sh`. Lightweight
development can continue without the symlink.

### A repository command imports the wrong package version

Run it through the locked, already-synced project environment, not the system
Python. Confirm with:

```bash
uv run --locked --no-sync python --version
uv run --locked --no-sync python -c "import slm_lab; print(slm_lab.__file__)"
```

### Hooks or local state are missing after a clone

Rerun `uv run --locked --no-sync scripts/setup/install_git_hooks.sh`. The
installer is designed to be safe to run repeatedly.
