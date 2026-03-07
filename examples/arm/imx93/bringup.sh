#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# End-to-end bringup script for ExecuTorch on i.MX93 Cortex-M33 + Ethos-U65.
# Run from the executorch repo root on the dev host.
#
# Prerequisites:
#   - micromamba activate rsmi (or your conda env with executorch installed)
#   - arm-none-eabi-gcc toolchain on PATH
#   - cloudflared for SSH proxy
#   - Ethos-U SDK at examples/arm/arm-scratch/ethos-u/
#
# Usage:
#   source examples/arm/imx93/bringup.sh
#   # Then call individual functions:
#   imx93_build           # build firmware
#   imx93_deploy          # copy to device and start
#   imx93_poll            # read diag slots
#   imx93_ddr_load <pte>  # load large model via DDR mailbox
#   imx93_numerics <pte>  # host-side reference for comparison

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
BUILD_DIR="${REPO_ROOT}/arm_test"
MODEL_BUILD="${BUILD_DIR}/model/cmake-out"
DELEGATE_BUILD="${BUILD_DIR}/cmake-out"
ELF="${MODEL_BUILD}/arm_executor_runner"

SSH_CMD="ssh -o ProxyCommand=\"cloudflared access ssh --hostname dev.apparatlabs.com\" fritz@dev.apparatlabs.com"
SCP_CMD="scp -o ProxyCommand=\"cloudflared access ssh --hostname dev.apparatlabs.com\""

REMOTE_USER="fritz@dev.apparatlabs.com"

# ─── Build ───────────────────────────────────────────────────────────────────

imx93_build() {
    echo "=== Building delegate library ==="
    cmake --build "${DELEGATE_BUILD}" -j8 -- executorch_delegate_ethos_u

    echo "=== Copying delegate lib ==="
    cp "${DELEGATE_BUILD}/backends/arm/libexecutorch_delegate_ethos_u.a" \
       "${DELEGATE_BUILD}/lib/libexecutorch_delegate_ethos_u.a"

    echo "=== Building arm_executor_runner ==="
    cmake --build "${MODEL_BUILD}" -j8 -- arm_executor_runner

    echo "=== ELF built: ${ELF} ==="
    arm-none-eabi-size "${ELF}" || true

    echo ""
    echo "Key symbols (update ddr_loader addresses from these):"
    arm-none-eabi-nm "${ELF}" | grep -E 'ddr_pte_mailbox|executorch_diag_words|executorch_fault_words' | while read addr type name; do
        a55_addr=$(printf "0x%08x" $(( 0x$addr + 0x200000 )))
        echo "  $name = 0x$addr  (A55: $a55_addr)"
    done
}

# ─── Deploy ──────────────────────────────────────────────────────────────────

imx93_deploy() {
    local elf="${1:-${ELF}}"
    echo "=== Copying firmware to device ==="
    eval ${SCP_CMD} "${elf}" "${REMOTE_USER}:/tmp/arm_executor_runner.elf"

    echo "=== Starting firmware via remoteproc ==="
    eval ${SSH_CMD} "bash -s" <<'REMOTE_EOF'
set -e
STATE=/sys/class/remoteproc/remoteproc0/state
FW=/sys/class/remoteproc/remoteproc0/firmware
sudo sh -c "echo stop > $STATE" 2>/dev/null || true
sleep 1
sudo cp /tmp/arm_executor_runner.elf /lib/firmware/arm_executor_runner.elf
sudo sh -c "echo arm_executor_runner.elf > $FW"
sudo sh -c "echo start > $STATE"
echo "Firmware started"
sleep 2
echo "State: $(cat $STATE)"
REMOTE_EOF
}

# ─── Poll diagnostics ───────────────────────────────────────────────────────

imx93_poll() {
    local diag_base="${1:-}"
    if [ -z "$diag_base" ]; then
        echo "Usage: imx93_poll <diag_a55_addr>"
        echo "Get address from: imx93_build output or:"
        echo "  arm-none-eabi-nm ${ELF} | grep executorch_diag_words"
        echo "  Add 0x200000 to the M33 address for A55 view"
        return 1
    fi

    echo "=== Polling diag slots at ${diag_base} ==="
    eval ${SSH_CMD} "bash -s" <<REMOTE_EOF
BASE=${diag_base}
echo "Diag slots:"
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 24 25 26 27 28 29 30 31 32 33 34 35 38 40 41 42 43 48 49 50 51 52 53 54 55; do
    addr=\$(printf "0x%x" \$(( \$BASE + \$slot * 4 )))
    val=\$(sudo devmem \$addr 32 2>/dev/null || echo "ERROR")
    printf "  slot[%2d] @ %s = %s\n" \$slot \$addr \$val
done
REMOTE_EOF
}

# ─── DDR model loading ───────────────────────────────────────────────────────

imx93_ddr_load() {
    local pte="${1:-}"
    local mailbox_addr="${2:-}"
    local diag_addr="${3:-}"

    if [ -z "$pte" ] || [ -z "$mailbox_addr" ] || [ -z "$diag_addr" ]; then
        echo "Usage: imx93_ddr_load <model.pte> <mailbox_a55_addr> <diag_a55_addr>"
        echo ""
        echo "Addresses from ELF (M33 addr + 0x200000):"
        echo "  arm-none-eabi-nm ${ELF} | grep -E 'ddr_pte_mailbox|executorch_diag_words'"
        return 1
    fi

    echo "=== Copying PTE and ddr_loader to device ==="
    eval ${SCP_CMD} "${pte}" "${REMOTE_USER}:/tmp/model.pte"
    eval ${SCP_CMD} "${REPO_ROOT}/examples/arm/imx93/ddr_loader.c" "${REMOTE_USER}:/tmp/ddr_loader.c"

    echo "=== Building and running ddr_loader on device ==="
    eval ${SSH_CMD} "bash -s" <<REMOTE_EOF
set -e
cd /tmp
gcc -o ddr_loader ddr_loader.c -O2
echo "Running: sudo ./ddr_loader /tmp/model.pte ${mailbox_addr} ${diag_addr}"
sudo ./ddr_loader /tmp/model.pte ${mailbox_addr} ${diag_addr}
REMOTE_EOF
}

# ─── Host-side numerics reference ────────────────────────────────────────────

imx93_numerics() {
    local pte="${1:-${BUILD_DIR}/model.pte}"

    echo "=== Running host-side reference inference ==="
    python3 -c "
import torch, numpy as np
from executorch.runtime import Runtime, Program, Method

runtime = Runtime.get()
program = runtime.load_program('${pte}')
method = program.load_method('forward')

# Zero input (same as device default)
inputs = method.method_meta.input_tensors()
input_list = []
for inp in inputs:
    shape = [inp.sizes[i] for i in range(inp.dim_order_len)]
    t = torch.zeros(shape, dtype=torch.float32)
    input_list.append(t)

outputs = method.execute(input_list)
out = outputs[0]
if isinstance(out, torch.Tensor):
    out_np = out.numpy().flatten()
    print(f'Output shape: {out.shape}')
    print(f'Output dtype: {out.dtype}')
    print(f'First 10 values: {out_np[:10]}')
    argmax = np.argmax(out_np)
    print(f'Argmax: {argmax}  Value: {out_np[argmax]:.6f}')
    top5 = np.argsort(out_np)[-5:][::-1]
    print(f'Top-5 indices: {top5}')
    print(f'Top-5 values: {out_np[top5]}')
"
}

# ─── Read trace buffer ───────────────────────────────────────────────────────

imx93_trace() {
    echo "=== Reading trace buffer from DTCM ==="
    eval ${SSH_CMD} "bash -s" <<'REMOTE_EOF'
# trace_buf is at start of .data in DTCM; trace_idx precedes it
# Read first 4KB of DTCM .data section — trace is typically near the start
# Adjust base address if trace_buf symbol moves
python3 -c "
import mmap, os, struct
fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
# DTCM base from A55 side
m = mmap.mmap(fd, 0x20000, mmap.MAP_SHARED, mmap.PROT_READ, offset=0x20200000)
# scan for printable ASCII
data = m.read(0x20000)
m.close()
os.close(fd)
# find longest printable run
best_start = 0
best_len = 0
cur_start = 0
cur_len = 0
for i, b in enumerate(data):
    if 0x20 <= b <= 0x7e or b in (0x0a, 0x0d, 0x09):
        if cur_len == 0:
            cur_start = i
        cur_len += 1
    else:
        if cur_len > best_len:
            best_start = cur_start
            best_len = cur_len
        cur_len = 0
if cur_len > best_len:
    best_start = cur_start
    best_len = cur_len
if best_len > 10:
    print(f'Trace at offset 0x{best_start:x} ({best_len} bytes):')
    print(data[best_start:best_start+best_len].decode('ascii', errors='replace'))
else:
    print('No significant trace data found')
" 2>/dev/null || echo "Could not read trace"
REMOTE_EOF
}

# ─── Quick reference ─────────────────────────────────────────────────────────

imx93_help() {
    cat <<'EOF'
i.MX93 ExecuTorch Bringup Commands
===================================

  imx93_build                Build firmware ELF (delegate + runner)
  imx93_deploy [elf]         Deploy ELF to device via remoteproc
  imx93_poll <diag_addr>     Read diagnostic slots from DTCM
  imx93_ddr_load <pte> <mbox> <diag>   Load large model via DDR mailbox
  imx93_numerics [pte]       Run host-side reference inference
  imx93_trace                Read M33 trace buffer

Typical workflow:
  1. imx93_build
  2. imx93_deploy
  3. imx93_poll 0x20205XXX       (addr from build output)
  4. For large models: imx93_ddr_load model.pte 0x20205XXX 0x20205XXX

Memory map:
  M33 ITCM: 0x0FFE0000 (128KB)  A55: 0x201E0000
  M33 DTCM: 0x20000000 (128KB)  A55: 0x20200000
  DDR:      0xA8000000 (128MB)   shared
  NPU:      0x4A900000           Ethos-U65-256

Diag slot meanings:
  slot[0]   magic (0x45544447 = ETDG)
  slot[1]   current stage
  slot[2-11] first 10 output values (IEEE754 bits)
  slot[12]  argmax
  slot[13]  max value (IEEE754 bits)
  slot[38]  total output float count
EOF
}

echo "i.MX93 bringup functions loaded. Run 'imx93_help' for usage."
