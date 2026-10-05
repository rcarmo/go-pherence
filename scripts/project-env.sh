#!/usr/bin/env bash
# Source before any cache/temp-producing command. No global env or cleanup.
# Retained evidence/checkpoints are deliberately outside this disposable root.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project-tmp.sh"
# Explicit/inherited root is validated; resolve before overriding child TMPDIR.
PROJECT_TMP_ROOT="$(project_tmp_resolve go-pherence)" || { return 1 2>/dev/null || exit 1; }
export PROJECT_TMP_ROOT
project_tmp_init "$PROJECT_TMP_ROOT" || { return 1 2>/dev/null || exit 1; }
export PROJECT_CACHE_ROOT="$PROJECT_TMP_ROOT/cache"
export PROJECT_BUILD_ROOT="$PROJECT_TMP_ROOT/build"
export PROJECT_TEST_ROOT="$PROJECT_TMP_ROOT/tests"
export PROJECT_LOG_ROOT="$PROJECT_TMP_ROOT/logs"
export PROJECT_RUN_ROOT="$PROJECT_TMP_ROOT/runs"
export GO_PHERENCE_RUN_PURPOSE="${GO_PHERENCE_RUN_PURPOSE:-command}"
export GO_PHERENCE_RUN_ID="${GO_PHERENCE_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}"
for component in "$GO_PHERENCE_RUN_PURPOSE" "$GO_PHERENCE_RUN_ID"; do
 case "$component" in ''|*[!A-Za-z0-9._-]*|.*|-*) echo 'Invalid project run purpose/id' >&2; return 1 2>/dev/null || exit 1;; esac
done
export PROJECT_RUN_DIR="$PROJECT_RUN_ROOT/$GO_PHERENCE_RUN_PURPOSE/$GO_PHERENCE_RUN_ID"
# Check every existing parent; never follow a substituted run/cache symlink.
for path in "$PROJECT_TMP_ROOT" "$PROJECT_CACHE_ROOT" "$PROJECT_BUILD_ROOT" "$PROJECT_BUILD_ROOT/bin" "$PROJECT_TEST_ROOT" "$PROJECT_LOG_ROOT" "$PROJECT_RUN_ROOT" "$PROJECT_RUN_ROOT/$GO_PHERENCE_RUN_PURPOSE" "$PROJECT_RUN_DIR"; do
 if ! project_path_usable "$path"; then
  echo "Refusing unowned/non-directory/symlink project path: $path" >&2
  return 1 2>/dev/null || exit 1
 fi
 mkdir -p "$path" || { return 1 2>/dev/null || exit 1; }
done
# Fixed caches; caller HOME/toolchain/install locations are not changed.
export GOCACHE="$PROJECT_CACHE_ROOT/go-build"
export GOMODCACHE="$PROJECT_CACHE_ROOT/go-mod"
export GOPATH="$PROJECT_CACHE_ROOT/go-path"
export XDG_CACHE_HOME="$PROJECT_CACHE_ROOT/xdg"
export BUN_INSTALL_CACHE_DIR="$PROJECT_CACHE_ROOT/bun"
export npm_config_cache="$PROJECT_CACHE_ROOT/npm"
export PIP_CACHE_DIR="$PROJECT_CACHE_ROOT/pip"
export UV_CACHE_DIR="$PROJECT_CACHE_ROOT/uv"
export PYTHONPYCACHEPREFIX="$PROJECT_CACHE_ROOT/python-bytecode"
export CARGO_TARGET_DIR="$PROJECT_BUILD_ROOT/cargo"
export TMPDIR="$PROJECT_RUN_DIR/tmp" TMP="$PROJECT_RUN_DIR/tmp" TEMP="$PROJECT_RUN_DIR/tmp"
export GOTMPDIR="$PROJECT_RUN_DIR/go-tmp"
for path in "$GOCACHE" "$GOMODCACHE" "$GOPATH" "$XDG_CACHE_HOME" "$BUN_INSTALL_CACHE_DIR" "$npm_config_cache" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$CARGO_TARGET_DIR" "$TMPDIR" "$GOTMPDIR"; do
 if ! project_path_usable "$path"; then
  echo "Refusing project tool path: $path" >&2; return 1 2>/dev/null || exit 1
 fi
 mkdir -p "$path" || { return 1 2>/dev/null || exit 1; }
done
