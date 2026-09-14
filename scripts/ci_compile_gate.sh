#!/usr/bin/env bash
# Fail-closed compile gate for SuperNPUBench PTO ISA (one-level) kernels.
#
# Rationale: each kernel dir ships a compile.all listing `make TESTCASE=...`
# lines. Authors comment out configs that are known-broken ("蓝区过不了",
# "已弃用"). So the set of *active* (un-commented) make lines across the
# selected kernels is the compile pass-list — curated in place. This gate runs
# every active make line and fails (exit 1) if any of them fails to compile.
# That is the signal a merge gate needs: a PR must not break a config that
# compiled before.
#
# Usage:
#   COMPILER_DIR=/path/to/linx.../bin \
#   LINX_SYSROOT=/path/to/linx.../sysroot/usr \
#   scripts/ci_compile_gate.sh [--list FILE] [KERNEL_SUBPATH ...]
#
#   --list FILE   Read kernel subpaths (relative to the kernel root) from FILE,
#                 one per line; blank lines and # comments ignored.
#   KERNEL_SUBPATH e.g. "matmul", "reduction/reducemax_col". If none given and
#                 no --list, every dir containing a compile.all is used.
#
# Env:
#   COMPILER_DIR (required) — linx toolchain bin dir (clang++/lld).
#   KERNEL_ROOT  (optional) — defaults to benchmark/one-level-arch/test/kernel.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_ROOT="${KERNEL_ROOT:-$REPO_ROOT/benchmark/one-level-arch/test/kernel}"

if [ -z "${COMPILER_DIR:-}" ]; then
  echo "ERROR: COMPILER_DIR is not set. Export it to the linx toolchain bin dir." >&2
  exit 2
fi
if [ ! -x "$COMPILER_DIR/clang++" ]; then
  echo "ERROR: $COMPILER_DIR/clang++ not found or not executable." >&2
  exit 2
fi
export COMPILER_DIR
[ -n "${LINX_SYSROOT:-}" ] && export LINX_SYSROOT

# ---- resolve the kernel selection -----------------------------------------
LIST_FILE=""
SELECTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST_FILE="$2"; shift 2 ;;
    --list=*) LIST_FILE="${1#--list=}"; shift ;;
    *) SELECTED+=("$1"); shift ;;
  esac
done

if [ -n "$LIST_FILE" ]; then
  if [ ! -f "$LIST_FILE" ]; then
    echo "ERROR: --list file not found: $LIST_FILE" >&2
    exit 2
  fi
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs 2>/dev/null || true)"
    [ -z "$line" ] && continue
    SELECTED+=("$line")
  done < "$LIST_FILE"
fi

# Build the list of compile.all files to run.
COMPILE_ALLS=()
if [ ${#SELECTED[@]} -gt 0 ]; then
  for sub in "${SELECTED[@]}"; do
    d="$KERNEL_ROOT/$sub"
    if [ -f "$d/compile.all" ]; then
      COMPILE_ALLS+=("$d/compile.all")
    else
      echo "ERROR: no compile.all under selected kernel: $sub ($d)" >&2
      exit 2
    fi
  done
else
  while IFS= read -r f; do
    COMPILE_ALLS+=("$f")
  done < <(find "$KERNEL_ROOT" -name compile.all | sort)
fi

if [ ${#COMPILE_ALLS[@]} -eq 0 ]; then
  echo "ERROR: no compile.all files selected under $KERNEL_ROOT" >&2
  exit 2
fi

echo "=========================================="
echo "SuperNPUBench compile gate"
echo "  COMPILER_DIR: $COMPILER_DIR"
echo "  LINX_SYSROOT: ${LINX_SYSROOT:-<unset>}"
echo "  kernels:      ${#COMPILE_ALLS[@]} compile.all file(s)"
echo "=========================================="

PASS=0
FAIL=0
FAILED_CMDS=""

for ca in "${COMPILE_ALLS[@]}"; do
  dir="$(dirname "$ca")"
  rel="${dir#"$KERNEL_ROOT"/}"
  echo ""
  echo "------------------------------------------"
  echo ">>> kernel: $rel"
  echo "------------------------------------------"
  # Active (un-commented) make lines are the pass-list for this kernel.
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    echo "+ ($rel) $cmd"
    if ( cd "$dir" && eval "$cmd" ); then
      echo "PASS  $rel :: $cmd"
      PASS=$((PASS+1))
    else
      echo "FAIL  $rel :: $cmd"
      FAIL=$((FAIL+1))
      FAILED_CMDS="$FAILED_CMDS
  [$rel] $cmd"
    fi
  done < <(grep -E '^[[:space:]]*make([[:space:]]|$)' "$ca")
done

echo ""
echo "=========================================="
echo "  PASS: $PASS  FAIL: $FAIL  TOTAL: $((PASS+FAIL))"
echo "=========================================="
if [ $FAIL -gt 0 ]; then
  echo "Failed compile configs:$FAILED_CMDS"
  exit 1
fi
if [ $((PASS+FAIL)) -eq 0 ]; then
  echo "ERROR: no active make lines found — nothing was gated." >&2
  exit 2
fi
