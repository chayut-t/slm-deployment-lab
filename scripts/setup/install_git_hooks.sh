#!/bin/sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

git config core.hooksPath .githooks
python3 scripts/setup/bootstrap_local_state.py

echo "Git hooks enabled from .githooks"
