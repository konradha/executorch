#!/bin/bash
# Build ExecuTorch executor_runner for NXP i.MX93 Ethos-U65 Linux path.
# Run this natively on the i.MX93 (aarch64 Linux) or cross-compile with
# an aarch64 toolchain.
#
# Usage (native on device):
#   cd /path/to/executorch
#   ./examples/arm/imx93/build_linux_runner.sh
#
# Usage (cross-compile):
#   MUSL_TOOLCHAIN_ROOT=/path/to/aarch64-linux-musl-cross \
#   ./examples/arm/imx93/build_linux_runner.sh --cross
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${ET_BUILD_DIR:-${ET_ROOT}/cmake-out-imx}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CROSS=false

for arg in "$@"; do
  case "$arg" in
    --cross) CROSS=true ;;
    --debug) BUILD_TYPE=Debug ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

CMAKE_ARGS=(
  -S "$ET_ROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DEXECUTORCH_BUILD_ARM_ETHOSU_IMX=ON
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON
  -DEXECUTORCH_ENABLE_LOGGING=ON
  -DCMAKE_C_FLAGS_RELEASE="-UNDEBUG"
  -DCMAKE_CXX_FLAGS_RELEASE="-UNDEBUG"
)

if $CROSS; then
  if [ -z "${MUSL_TOOLCHAIN_ROOT:-}" ]; then
    echo "Error: set MUSL_TOOLCHAIN_ROOT for cross-compilation"
    exit 1
  fi
  CMAKE_ARGS+=(
    -DCMAKE_TOOLCHAIN_FILE="$ET_ROOT/examples/arm/ethos-u-setup/aarch64-linux-musl-toolchain.cmake"
  )
fi

echo "=== Configuring ExecuTorch for i.MX93 Ethos-U Linux ==="
echo "  Build dir: $BUILD_DIR"
echo "  Build type: $BUILD_TYPE"
echo "  Cross-compile: $CROSS"

cmake "${CMAKE_ARGS[@]}"

echo "=== Building ==="
cmake --build "$BUILD_DIR" -j"$(nproc)" --target executor_runner

echo "=== Done ==="
echo "Binary: $BUILD_DIR/executor_runner"
echo ""
echo "Usage: sudo $BUILD_DIR/executor_runner --model_path=model.pte"
