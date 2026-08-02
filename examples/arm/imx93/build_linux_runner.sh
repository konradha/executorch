#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${ET_BUILD_DIR:-${ET_ROOT}/cmake-out-imx93-fastpath}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CROSS=false
STAGED_ROOT=""
TOOLCHAIN_MODE="${TOOLCHAIN_MODE:-auto}"

if command -v nproc >/dev/null 2>&1; then
  BUILD_JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  BUILD_JOBS="$(getconf _NPROCESSORS_ONLN)"
else
  BUILD_JOBS="$(sysctl -n hw.ncpu)"
fi

for arg in "$@"; do
  case "$arg" in
    --cross) CROSS=true ;;
    --debug) BUILD_TYPE=Debug ;;
    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

detect_musl_root() {
  local compiler_path
  compiler_path="$(command -v aarch64-linux-musl-gcc 2>/dev/null || true)"
  if [ -z "$compiler_path" ]; then
    return 1
  fi
  dirname "$(dirname "$(realpath "$compiler_path")")"
}

detect_python() {
  if command -v python >/dev/null 2>&1; then
    python - <<'PY'
import sys
print(sys.executable)
PY
    return 0
  fi
  command -v python3
}

check_python_module() {
  local module_name="$1"
  "$PYTHON_EXECUTABLE" - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$module_name") else 1)
PY
}

detect_conda_cross_root() {
  local prefix
  prefix="${CONDA_TOOLCHAIN_ROOT:-${CONDA_PREFIX:-}}"
  if [ -z "$prefix" ]; then
    return 1
  fi
  if [ ! -x "$prefix/bin/aarch64-conda-linux-gnu-gcc" ]; then
    return 1
  fi
  printf '%s\n' "$prefix"
}

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(detect_python)}"

if ! check_python_module torch || ! check_python_module torchgen || ! check_python_module yaml; then
  echo "Error: PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE is missing required build modules."
  echo "  Required: torch, torchgen, yaml"
  echo "  Use your export env Python for codegen, for example:"
  echo "    export PYTHON_EXECUTABLE=\"\$(micromamba run -n rsmi python -c 'import sys; print(sys.executable)')\""
  echo "  Or install the missing packages into the active environment."
  exit 1
fi

stage_source_tree() {
  local stage_base="${TMPDIR:-/tmp}/executorch-fastpath-src"
  local staged_root="${stage_base}/executorch"
  mkdir -p "$stage_base"
  rm -rf "$staged_root"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude '.git/worktrees/*/index.lock' \
      --exclude 'cmake-out*' \
      "$ET_ROOT/" "$staged_root/"
  else
    cp -R "$ET_ROOT" "$staged_root"
  fi
  ET_ROOT="$staged_root"
  BUILD_DIR="${ET_BUILD_DIR:-${ET_ROOT}/cmake-out-imx93-fastpath}"
  STAGED_ROOT="$staged_root"
}

if [ "$(basename "$ET_ROOT")" != "executorch" ]; then
  stage_source_tree
fi

CMAKE_ARGS=(
  -S "$ET_ROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"
  -DBUILD_TESTING=OFF
  -DEXECUTORCH_BUILD_ARM_ETHOSU_IMX=ON
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON
  -DEXECUTORCH_ENABLE_LOGGING=ON
  -DCMAKE_C_FLAGS_RELEASE=-UNDEBUG
  -DCMAKE_CXX_FLAGS_RELEASE=-UNDEBUG
)

if [ -n "${ETHOSU_IMX_DRIVER_SOURCE_DIR:-}" ]; then
  CMAKE_ARGS+=("-DETHOSU_IMX_DRIVER_SOURCE_DIR=${ETHOSU_IMX_DRIVER_SOURCE_DIR}")
fi
if [ -n "${ETHOSU_IMX_DRIVER_GIT_REPO:-}" ]; then
  CMAKE_ARGS+=("-DETHOSU_IMX_DRIVER_GIT_REPO=${ETHOSU_IMX_DRIVER_GIT_REPO}")
fi
if [ -n "${ETHOSU_IMX_DRIVER_GIT_TAG:-}" ]; then
  CMAKE_ARGS+=("-DETHOSU_IMX_DRIVER_GIT_TAG=${ETHOSU_IMX_DRIVER_GIT_TAG}")
fi

if $CROSS; then
  if [ "$TOOLCHAIN_MODE" = "auto" ]; then
    if CONDA_TOOLCHAIN_ROOT="$(detect_conda_cross_root)" && [ -n "$CONDA_TOOLCHAIN_ROOT" ]; then
      TOOLCHAIN_MODE="conda"
    elif [ -n "${MUSL_TOOLCHAIN_ROOT:-}" ] || detect_musl_root >/dev/null 2>&1; then
      TOOLCHAIN_MODE="musl"
    else
      echo "Error: no cross toolchain detected"
      echo "  Micromamba path: install gcc_linux-aarch64 gxx_linux-aarch64 binutils_linux-aarch64 sysroot_linux-aarch64"
      echo "  Alternate path: set MUSL_TOOLCHAIN_ROOT for the existing musl toolchain file"
      exit 1
    fi
  fi

  case "$TOOLCHAIN_MODE" in
    conda)
      CONDA_TOOLCHAIN_ROOT="${CONDA_TOOLCHAIN_ROOT:-$(detect_conda_cross_root)}"
      CMAKE_ARGS+=(
        -DCMAKE_TOOLCHAIN_FILE="$ET_ROOT/examples/arm/ethos-u-setup/aarch64-linux-conda-toolchain.cmake"
        -DCONDA_TOOLCHAIN_ROOT="$CONDA_TOOLCHAIN_ROOT"
      )
      if [ -n "${CONDA_LINUX_SYSROOT:-}" ]; then
        CMAKE_ARGS+=("-DCONDA_LINUX_SYSROOT=${CONDA_LINUX_SYSROOT}")
      elif [ -n "${CONDA_BUILD_SYSROOT:-}" ]; then
        CMAKE_ARGS+=("-DCONDA_LINUX_SYSROOT=${CONDA_BUILD_SYSROOT}")
      fi
      ;;
    musl)
      if [ -z "${MUSL_TOOLCHAIN_ROOT:-}" ]; then
        MUSL_TOOLCHAIN_ROOT="$(detect_musl_root)"
      fi
      CMAKE_ARGS+=(
        -DCMAKE_TOOLCHAIN_FILE="$ET_ROOT/examples/arm/ethos-u-setup/aarch64-linux-musl-toolchain.cmake"
        -DMUSL_TOOLCHAIN_ROOT="$MUSL_TOOLCHAIN_ROOT"
      )
      ;;
    *)
      echo "Error: unsupported TOOLCHAIN_MODE=$TOOLCHAIN_MODE"
      exit 1
      ;;
  esac

  unset CC CXX AR LD RANLIB STRIP SDKROOT CFLAGS CXXFLAGS CPPFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS
fi

echo "Configuring ExecuTorch for the i.MX93 Linux fast path"
echo "  Build dir: $BUILD_DIR"
echo "  Build type: $BUILD_TYPE"
echo "  Cross compile: $CROSS"
echo "  Python: $PYTHON_EXECUTABLE"
if [ -n "$STAGED_ROOT" ]; then
  echo "  Staged source: $STAGED_ROOT"
fi
if $CROSS; then
  echo "  Toolchain mode: $TOOLCHAIN_MODE"
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS" --target executor_runner

echo "Built: $BUILD_DIR/executor_runner"
