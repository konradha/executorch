# i.MX93 Ethos-U65 ExecuTorch Bringup — Current Handoff

## Goal
Run one real ExecuTorch inference on i.MX93 Cortex-M33 with Ethos-U65-256, and verify correctness both:
- on the device itself
- from this host machine against the same model/input

Definition of done:
1. real NPU-backed inference runs without rebooting the SoC
2. device-side result is verified correct
3. host-side result matches the device result for the same model/input

## Bottom Line
ExecuTorch bringup is no longer blocked in generic `Method::execute()`.

What now works on device:
- startup
- C++ init array
- operator registration
- `Program::load()`
- `load_method()`
- input preparation
- `Method::execute()`
- CPU quantize / dequantize path
- delegate entry
- delegate input copy path
- delegate output metadata path
- delegate output copyback path
- full `run_model()` completion when the real NPU call is skipped

The only remaining blocker is the real Ethos-U launch path.

## Final Deliverables Required
Do not stop at "it ran once".

The work is only complete when all of these are true:
1. Real NPU invoke is enabled and completes.
2. `run_model()` reaches `kStageRunModelDone` with no reboot.
3. Device-side correctness is checked.
4. Host-side correctness is checked from this machine.

Preferred correctness path:
- Turn on `ET_BUNDLE_IO` and use the existing `verify_result()` path in [arm_executor_runner.cpp](/Users/fritz/code/edge/executorch/examples/arm/executor_runner/arm_executor_runner.cpp), which already calls:
  - `compute_method_output_error_stats(...)`
  - `verify_method_outputs(...)`
- If BundleIO is not available for the current model, dump the device output tensor and compare it against a host reference run for the same input.

## Current Narrowest Boundary
The remaining fault is inside the Ethos-U core driver launch sequence, after power/reset succeeds and before `ethosu_dev_run_command_stream()` returns cleanly.

Most precise proven boundaries:
1. `ethosu_request_power()` is safe.
2. Reboot happens after that point when real NPU launch is enabled.
3. With `ethosu_invoke_v3()` replaced by `ethosu_invoke_async()` and wait skipped, the board still reboots.
4. Therefore the failure is not in `ethosu_wait()` / IRQ completion. It is in command submission / NPU start itself.
5. A driver-level park immediately after `ethosu_request_power()` is stable.
6. Advancing past that park reboots the SoC.

The next local image already instruments `ethosu_dev_run_command_stream()` so the next test will answer:
- are the initial `QBASE/QSIZE/QCONFIG` writes safe?
- or is the fault already in that first register block?

## What Was Fixed Already

### 1. Startup / operator registration bug
`run_init_array()` in [target.cpp](/Users/fritz/code/edge/executorch/examples/arm/imx93/target.cpp) used to skip the last constructor by position.
That skipped `_GLOBAL__sub_I_register_ioqdq_ops.cpp`, so quantize/dequantize kernels never registered and `load_method()` failed with `OperatorMissing`.

This is fixed now.

### 2. False execute-path diagnosis
The old handoff blamed generic `execute()` and stack/logging broadly.
That was stale.

After iterative checkpoints, the real path is:
- `Method::execute()` works
- delegate software path works
- only the real Ethos-U hardware launch is left

### 3. Delegate path cleanup
The following were necessary to stop false crashes in the delegate software path:
- bypass `ethosu_reserve_driver()` and use the single pre-initialized global driver directly
- remove nested profiling around delegate input memcpy
- remove fragile tensor-dimension walking in input path
- replace output-side repeated `nbytes()` pattern that was provoking the prior reboot boundary
- add stable diag breadcrumbs across delegate and platform layers

## Current Source State

### [backends/arm/runtime/EthosUBackend_Cortex_M.cpp](/Users/fritz/code/edge/executorch/backends/arm/runtime/EthosUBackend_Cortex_M.cpp)
- Uses `executorch_get_ethosu_driver()` from target code instead of `ethosu_reserve_driver()`.
- `kDebugPlatformStageLimit = 0xFFFFFFFFu`
- `kDebugOutputCheckpoint = 0xFFFFFFFFu`
- `kDebugSkipNpuWait = true`
- Real `ethosu_invoke_v3()` is not called directly right now.
- Instead:
  - `ethosu_invoke_async(...)`
  - record diag
  - skip `ethosu_wait(...)`
- This is intentional to split command submission from completion/IRQ waiting.

Important diag slots from this file:
- slot `40`: delegate stage
- slot `41`: NPU invoke markers
- slot `42`: async invoke return code
- slot `43`: wait return code
- slot `48`: platform stage
- slot `49`: driver ptr
- slots `50..53`: base addresses and command size
- slots `54..55`: total output byte accounting
- slots `56..63`: latest output-loop breadcrumbs

### [examples/arm/imx93/target.cpp](/Users/fritz/code/edge/executorch/examples/arm/imx93/target.cpp)
- exports `executorch_get_ethosu_driver()`
- contains init-array fix
- contains diag/fault storage
- persistent diag buffer now lives in DTCM, not DDR

### [examples/arm/executor_runner/iMX93.ld](/Users/fritz/code/edge/executorch/examples/arm/executor_runner/iMX93.ld)
- stack is 64 KB
- heap is 4 KB
- persistent diag section is placed in DTCM

### [examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_driver.c](/Users/fritz/code/edge/executorch/examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_driver.c)
- local diag helper added
- `handle_command_stream()` stage markers added
- current park threshold is `stage >= 3`

Current meaning:
- stage `0`: entered `handle_command_stream`
- stage `1`: after flush/invalidate hook
- stage `2`: after `ethosu_request_power()`
- stage `3`: after `ethosu_dev_run_command_stream()`

### [examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_device_u55_u65.c](/Users/fritz/code/edge/executorch/examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_device_u55_u65.c)
- local diag helper added
- `ethosu_dev_run_command_stream()` stage markers added
- current park threshold is `stage >= 1`

Current meaning:
- stage `0`: entered `ethosu_dev_run_command_stream`
- stage `1`: after `QBASE/QSIZE/QCONFIG` writes
- stage `2`: after `BASEP[]` and `REGIONCFG`
- stage `3`: after final `CMD` write that transitions to running state

This image is built locally but had not been deployed at the time of this handoff because Cloudflare SSH became flaky.

## Most Important Proven Runtime Results

### A. Full software-only path succeeds
With NPU invoke stubbed out, the board stayed up and diag showed:
- slot `1 = 0xA1000023`
- slot `35 = 0`

Meaning:
- `run_model()` reached `kStageRunModelDone`
- `Method::execute()` returned `Error::Ok`

This is the key milestone. ExecuTorch itself is running.

### B. Real NPU invoke reintroduces reboot
When real invoke was re-enabled:
- SSH died
- host later recovered
- `remoteproc0` came back `offline`
- firmware reset to `rproc-imx-rproc-fw`

### C. `ethosu_wait()` is not the first failing boundary
With `ethosu_invoke_async()` and wait skipped, the SoC still rebooted.

Meaning:
- failure is in async launch path itself
- not in completion wait / IRQ handling

### D. `ethosu_request_power()` is safe
With a park at `handle_command_stream` stage `2`, the board stayed `running`.

Example diag from that parked image:
- slot `41 = 0xE7000001`
- slot `42 = 0x00000450`
- slot `44 = 0xC1000002`

Interpretation:
- entered async invoke
- `ethosu_invoke_async()` had not finished cleanly
- parked immediately after power/reset path

### E. Advancing beyond stage 2 reboots
Raising the `handle_command_stream` park threshold to `3` caused the reboot to return.

Meaning:
- the remaining fault is between:
  - after `ethosu_request_power()`
  - and before or during return from `ethosu_dev_run_command_stream()`

## Current ELF Symbols
From the current local ELF:

```text
main                      = 0x0ffe43d8
executorch_get_ethosu_driver = 0x0ffe75fc
executorch_fault_words    = 0x200052e4
executorch_diag_words     = 0x200053e4
executorch_persist_words  = 0x2000dd80
```

A55-visible aliases:
- diag base: `0x202053e4`
- fault base: `0x202052e4`
- persist base: `0x2020dd80`

## Deploy / Debug Commands

### Build
Use the user’s environment:

```bash
cd /Users/fritz/code/edge/executorch
source ~/.bashrc
micromamba activate rsmi
```

Then always rebuild sequentially:

```bash
cmake --build arm_test/cmake-out -j8 -- executorch_delegate_ethos_u
cp arm_test/cmake-out/backends/arm/libexecutorch_delegate_ethos_u.a arm_test/cmake-out/lib/libexecutorch_delegate_ethos_u.a
cmake --build arm_test/model/cmake-out -j8 -- arm_executor_runner
```

Do not rebuild delegate and runner in parallel. That caused stale linkage earlier.

### Model Artifact
Current model artifact on this host:

```bash
/Users/fritz/code/edge/executorch/arm_test/model.pte
```

That is the model that device-side and host-side correctness checks must both use.

### Copy firmware
```bash
scp -i ~/.ssh/id_rsa \
  -o ProxyCommand="cloudflared access ssh --hostname dev.apparatlabs.com" \
  /Users/fritz/code/edge/executorch/arm_test/model/cmake-out/arm_executor_runner \
  fritz@dev.apparatlabs.com:/tmp/arm_executor_runner.elf
```

### Deploy and poll
```bash
ssh -tt -i ~/.ssh/id_rsa \
  -o ProxyCommand="cloudflared access ssh --hostname dev.apparatlabs.com" \
  fritz@dev.apparatlabs.com '
set -e
STATE=/sys/class/remoteproc/remoteproc0/state
FW=/sys/class/remoteproc/remoteproc0/firmware
sudo sh -lc "echo stop > $STATE" || true
sudo cp /tmp/arm_executor_runner.elf /lib/firmware/arm_executor_runner.elf
sudo sh -lc "echo arm_executor_runner.elf > $FW"
sudo sh -lc "echo start > $STATE"
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 5
  printf "poll %s state " "$n"
  cat $STATE || true
done
for slot in 1 24 34 35 40 41 42 43 44 45 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63; do
  addr=$((0x202053e4 + slot * 4))
  printf "%s 0x%08x " "$slot" "$addr"
  sudo devmem "$addr" 32 || true
done
'
```

### Useful reconnect check
```bash
for n in $(seq 1 20); do
  if ssh -i ~/.ssh/id_rsa \
       -o ConnectTimeout=5 \
       -o ProxyCommand="cloudflared access ssh --hostname dev.apparatlabs.com" \
       fritz@dev.apparatlabs.com 'echo up' 2>/dev/null; then
    break
  fi
  sleep 10
done
```

## Exact Next Step
Deploy the already-prepared image with the new `ethosu_dev_run_command_stream()` stage split.

Expected outcomes:

### If it parks with:
- slot `45 = 0xC2000001`

Then:
- `QBASE/QSIZE/QCONFIG` writes are safe
- next test is raise the park threshold so it stops after `BASEP[]/REGIONCFG`

### If it reboots before that park:
Then:
- the fault is in the first queue-register programming block itself
- likely one of:
  - register mapping wrong for i.MX93 integration
  - access width/order issue
  - security / privilege / protection issue specific to those registers

## Recommended Fast Path To Inference
Do not revisit generic ExecuTorch plumbing. That work is done.

Fastest path now:
1. Deploy the `ethosu_dev_run_command_stream()` stage-1 park image.
2. If stage 1 parks, raise to stage 2.
3. If stage 2 parks, raise to stage 3.
4. The first stage that reboots is the exact hardware write boundary to fix.
5. Once all stages park cleanly:
   - disable parks in `ethosu_device_u55_u65.c`
   - keep `kDebugSkipNpuWait = true`
   - verify async launch returns without reboot
6. Then set `kDebugSkipNpuWait = false`
   - if that reboots, focus on IRQ / completion / `ethosu_wait()`
7. Once real invoke completes:
   - remove temporary parks
   - keep minimal diag only
   - run one full inference and capture output
8. Then verify correctness:
   - first on device
   - then on this host against the same model/input

## Correctness Verification

### Device-side correctness
Preferred:
1. Build with `ET_BUNDLE_IO=ON`.
2. Re-run the successful NPU-backed image.
3. Require the runner to print BundleIO success through the existing path in [arm_executor_runner.cpp](/Users/fritz/code/edge/executorch/examples/arm/executor_runner/arm_executor_runner.cpp):
   - `TEST: BundleIO index[0] Test_result: PASS`

If BundleIO is not currently wired for this model:
1. Dump the device output tensor after inference.
2. Record exact values/bytes.
3. Compare them to host reference output for the same input.

### Host-side correctness
The next agent must verify correctness from this host as well, not only "device did not reboot".

Minimum requirement:
1. Use `/Users/fritz/code/edge/executorch/arm_test/model.pte` on this host.
2. Use the same input as the device run.
3. Produce a host-side reference output.
4. Compare device output vs host output.

Acceptance:
- int8 / quantized buffers: exact match unless there is a proven layout-adjustment reason
- float outputs: compare within explicit tolerance and report the tolerance used

If needed, the host-side reference can be produced by:
- a small native ExecuTorch verifier around `Program::load()`, `load_method()`, and the same input/output accessors
- or by reusing existing ExecuTorch verification helpers if BundleIO is enabled for the model

Do not declare success without both:
- device-side proof
- host-side comparison

## What Not To Waste Time On
- generic `Method::execute()` debugging
- stack-size speculation
- input/output tensor metadata debugging
- `ethosu_reserve_driver()` semaphore logic
- removing more logs from ExecuTorch core

Those were already eliminated or bypassed.

## Reporting Format For The Next Iteration
After each iteration, report only:
1. Which park threshold / code path changed
2. Whether board stayed `running` or host rebooted
3. Key diag slots and values
4. The new narrowed boundary
5. The exact next test
6. If inference ran, the current correctness status on:
   - device
   - host

## Kickoff Prompt For The Next Agent

```text
You are continuing ExecuTorch bringup on i.MX93 Cortex-M33 + Ethos-U65 in /Users/fritz/code/edge/executorch.

Read this first:
- /Users/fritz/code/edge/executorch/examples/arm/imx93/HANDOFF.md

Current truth:
- ExecuTorch software path is working.
- With NPU invoke skipped, run_model reaches kStageRunModelDone and execute returns Ok.
- The only remaining blocker is the real Ethos-U launch path.
- The remaining fault is between successful ethosu_request_power() and return from ethosu_dev_run_command_stream().
- Final success requires correctness verification on both device and host.

Most important immediate task:
- Deploy the current local image and determine whether the new park in
  examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_device_u55_u65.c
  reaches stage 1 after QBASE/QSIZE/QCONFIG writes.

Environment:
- Use: source ~/.bashrc && micromamba activate rsmi
- Build sequentially, never delegate+runner in parallel:
  1. cmake --build arm_test/cmake-out -j8 -- executorch_delegate_ethos_u
  2. cp arm_test/cmake-out/backends/arm/libexecutorch_delegate_ethos_u.a arm_test/cmake-out/lib/libexecutorch_delegate_ethos_u.a
  3. cmake --build arm_test/model/cmake-out -j8 -- arm_executor_runner

SSH:
- ssh -i ~/.ssh/id_rsa -o ProxyCommand="cloudflared access ssh --hostname dev.apparatlabs.com" fritz@dev.apparatlabs.com

SCP:
- scp -i ~/.ssh/id_rsa -o ProxyCommand="cloudflared access ssh --hostname dev.apparatlabs.com" /Users/fritz/code/edge/executorch/arm_test/model/cmake-out/arm_executor_runner fritz@dev.apparatlabs.com:/tmp/arm_executor_runner.elf

Current diag base from latest ELF:
- executorch_diag_words = 0x200053e4
- A55-visible base = 0x202053e4

Current model artifact:
- /Users/fritz/code/edge/executorch/arm_test/model.pte

Primary files in play:
- backends/arm/runtime/EthosUBackend_Cortex_M.cpp
- examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_driver.c
- examples/arm/arm-scratch/ethos-u/core_software/core_driver/src/ethosu_device_u55_u65.c
- examples/arm/imx93/target.cpp
- examples/arm/executor_runner/iMX93.ld

Interpretation of current debug stages:
- slot 44 = handle_command_stream stage
  - 0 entered
  - 1 after flush hook
  - 2 after ethosu_request_power()
  - 3 after ethosu_dev_run_command_stream()
- slot 45 = ethosu_dev_run_command_stream stage
  - 0 entered
  - 1 after QBASE/QSIZE/QCONFIG
  - 2 after BASEP[] and REGIONCFG
  - 3 after CMD write

Do not go back to generic execute debugging.
Do not spend time on stack or tensor metadata unless a new result explicitly points there.
Advance the NPU register-programming boundary until one inference runs or the exact failing register write is identified.

When the NPU path finally runs:
1. verify correctness on device
   - prefer ET_BUNDLE_IO and require BundleIO PASS through verify_result()
2. verify correctness on this host
   - use /Users/fritz/code/edge/executorch/arm_test/model.pte
   - run the same input
   - compare device output to host reference output

Do not stop at "inference runs". Stop only when inference runs and correctness is checked from both device and host.
```
