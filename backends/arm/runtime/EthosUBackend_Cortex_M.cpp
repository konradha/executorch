/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include <ethosu_driver.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

extern "C" void halt_point();
extern "C" struct ethosu_driver* executorch_get_ethosu_driver(void);

using executorch::runtime::BackendExecutionContext;
using executorch::runtime::Error;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

extern "C" __attribute__((weak)) void
executorch_diag_mark(uint32_t slot, uintptr_t value);

namespace {

// Debug controls — set to 0xFFFFFFFF to disable parking/checkpoints.
// Lower values cause the delegate to halt at that stage for inspection.
constexpr uint32_t kDebugPlatformStageLimit = 0xFFFFFFFFu;
constexpr uint32_t kDebugOutputCheckpoint = 0xFFFFFFFFu;
constexpr bool kDebugSkipNpuWait = false;

inline void maybe_diag_mark(uint32_t slot, uintptr_t value) {
  if (executorch_diag_mark != nullptr) {
    executorch_diag_mark(slot, value);
  }
}

inline bool debug_platform_stage(uint32_t stage) {
  maybe_diag_mark(1, 0xD2000000u | stage);
  maybe_diag_mark(48, stage);
  if (stage >= kDebugPlatformStageLimit) {
    while (true) {
      __asm volatile("wfi");
    }
  }
  return false;
}

inline bool debug_output_checkpoint(int output_index, uint32_t phase) {
  const uint32_t checkpoint =
      static_cast<uint32_t>(output_index) * 8u + phase;
  maybe_diag_mark(56, 0xACD00000u | checkpoint);
  maybe_diag_mark(57, kDebugOutputCheckpoint);
  if (checkpoint == kDebugOutputCheckpoint) {
    while (true) {
      __asm volatile("wfi");
    }
  }
  return false;
}

} // namespace

struct PlatformState {};

PlatformState* platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> /*specs*/,
    executorch::runtime::MemoryAllocator* /*allocator*/) {
  return nullptr;
}

void platform_destroy(PlatformState* state) {
  delete state;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* /*execution_handle*/,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* ethosu_scratch) {
  if (debug_platform_stage(0)) {
    return Error::Ok;
  }
  // Bypass ethosu_reserve_driver() which needs OS semaphores.
  // Use the pre-initialized global driver from target.cpp instead.
  auto* driver = executorch_get_ethosu_driver();
  if (driver == nullptr) {
    return Error::InvalidState;
  }
  maybe_diag_mark(49, reinterpret_cast<uintptr_t>(driver));
  if (debug_platform_stage(1)) {
    return Error::Ok;
  }

  // Ethos-U low level driver expected order for Ethos U-55, we have
  // constant weight data, then scratch (which contains input and output)
  // scratch is written above in this function.
  uint64_t bases[ETHOSU_NUM_BASE_ADDRS] = {
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>((handles.weight_data))),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_scratch)),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_fast_scratch))};
  size_t bases_size[ETHOSU_NUM_BASE_ADDRS] = {
      handles.weight_data_size,
      handles.scratch_data_size,
      ethosu_fast_scratch_size};
  maybe_diag_mark(50, bases[0]);
  maybe_diag_mark(51, bases[1]);
  maybe_diag_mark(52, bases[2]);
  maybe_diag_mark(53, handles.cmd_data_size);
  if (debug_platform_stage(2)) {
    return Error::Ok;
  }
  maybe_diag_mark(41, 0xE7000001u);
  int result = ethosu_invoke_async(
      driver,
      static_cast<const void*>(handles.cmd_data),
      handles.cmd_data_size,
      bases,
      bases_size,
      ETHOSU_NUM_BASE_ADDRS, /* fixed array of pointers to binary interface*/
      nullptr);
  maybe_diag_mark(42, static_cast<uintptr_t>(result));
  maybe_diag_mark(41, 0xE7000002u);
  if (result == 0 && !kDebugSkipNpuWait) {
    maybe_diag_mark(41, 0xE7000003u);
    result = ethosu_wait(driver, true);
    maybe_diag_mark(43, static_cast<uintptr_t>(result));
    maybe_diag_mark(41, 0xE7000004u);
  }

  if (result != 0) {
    return Error::InvalidProgram;
  }

  size_t tensor_bytes_total = 0;
  size_t io_bytes_total = 0;
  // Write outputs from scratch into EValue pointers
  for (int i = 0; i < output_count; i++) {
    int io_count = 1;
    const char* output_addr = ethosu_scratch + handles.outputs->io[i].offset;
    maybe_diag_mark(58, static_cast<uintptr_t>(i));
    maybe_diag_mark(61, reinterpret_cast<uintptr_t>(output_addr));
    if (debug_output_checkpoint(i, 0)) {
      return Error::Ok;
    }
    // Process input EValue into scratch
    // Outputs are in the index immediately after inputs
    auto tensor_out = args[input_count + i]->toTensor();
    maybe_diag_mark(59, static_cast<uintptr_t>(tensor_out.scalar_type()));
    const size_t tensor_bytes = tensor_out.nbytes();
    maybe_diag_mark(60, tensor_bytes);
    if (debug_output_checkpoint(i, 1)) {
      return Error::Ok;
    }
    maybe_diag_mark(
        63, reinterpret_cast<uintptr_t>(tensor_out.mutable_data_ptr<char>()));
    if (debug_output_checkpoint(i, 2)) {
      return Error::Ok;
    }

    for (int dim = 0; dim < shapeDim; ++dim) {
      const int shape_value = handles.outputs->io[i].shape[dim];
      maybe_diag_mark(62, static_cast<uintptr_t>(dim));
      maybe_diag_mark(63, static_cast<uintptr_t>(shape_value));
      if (debug_output_checkpoint(i, 3u + static_cast<uint32_t>(dim))) {
        return Error::Ok;
      }
      io_count *= shape_value;
    }

    size_t io_bytes = static_cast<size_t>(io_count) *
        static_cast<size_t>(handles.outputs->io[i].elem_size);
    maybe_diag_mark(62, io_bytes);
    if (debug_output_checkpoint(i, 9)) {
      return Error::Ok;
    }

    if (tensor_bytes != io_bytes) {
      Error status = copy_with_layout_adjustment(
          handles.outputs->io[i], i, output_addr, tensor_out, tensor_bytes);
      if (status != Error::Ok) {
        return status;
      }
      io_bytes_total += tensor_bytes;
    } else {
      memcpy(
          tensor_out.mutable_data_ptr<char>(),
          static_cast<const char*>(output_addr),
          tensor_bytes);
      io_bytes_total += io_bytes;
    }
    if (debug_output_checkpoint(i, 10)) {
      return Error::Ok;
    }

    // At times the topological order of the outputs may change.
    // Lets instead ensure that the sum of output bytes match.
    tensor_bytes_total += tensor_bytes;
  }
  if (tensor_bytes_total != io_bytes_total) {
    return Error::InvalidProgram;
  }
  maybe_diag_mark(54, tensor_bytes_total);
  maybe_diag_mark(55, io_bytes_total);

  // Dump first 4 bytes of each output tensor into diag slots 40+i
  for (int i = 0; i < output_count && i < 4; i++) {
    auto t = args[input_count + i]->toTensor();
    uint32_t preview = 0;
    size_t n = t.nbytes() < 4 ? t.nbytes() : 4;
    memcpy(&preview, t.const_data_ptr<char>(), n);
    maybe_diag_mark(static_cast<uint32_t>(40 + i), preview);
  }

  debug_platform_stage(3);
  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
