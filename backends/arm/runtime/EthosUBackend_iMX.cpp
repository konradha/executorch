/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * NXP i.MX uses a Vela model and one I/O/scratch arena through /dev/ethosu0.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ethosu.hpp>
#include <uapi/ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

// Match EthosU::Interpreter::Invoke.
constexpr int64_t kDefaultTimeoutNs = 60'000'000'000LL;
// Cache-line alignment for the scratch base.
constexpr size_t kArenaDataAlignment = 64;
constexpr size_t kBytesPerMiB = size_t{1} << 20;
// Match the driver library's default arena size.
constexpr size_t kDefaultArenaBytes =
    size_t{DEFAULT_ARENA_SIZE_OF_MB} * kBytesPerMiB;

struct LinuxDriverOptions {
  std::string device_path = "/dev/ethosu0";
  int64_t timeout_ns = kDefaultTimeoutNs;
  bool enable_cycle_counter = true;
  std::array<uint32_t, ETHOSU_PMU_EVENT_MAX> pmu_events{};
};

struct PlatformState {
  LinuxDriverOptions options;
  std::shared_ptr<EthosU::Device> device;
  std::shared_ptr<EthosU::Network> network;
  std::shared_ptr<EthosU::Buffer> arena;
  std::shared_ptr<EthosU::Inference> inference;
  EthosU::MemoryLayout layout{};
  size_t arena_capacity{0};
  std::mutex execution_mutex;
};

namespace {

size_t checked_add(size_t left, size_t right, const char* message) {
  if (left > std::numeric_limits<size_t>::max() - right) {
    throw std::overflow_error(message);
  }
  return left + right;
}

size_t checked_multiply(size_t left, size_t right, const char* message) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::overflow_error(message);
  }
  return left * right;
}

size_t align_up(size_t value, size_t alignment) {
  return checked_add(value, alignment - 1, "Arena size overflow") &
      ~(alignment - 1);
}

uint32_t checked_u32(size_t value, const char* message) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(message);
  }
  return static_cast<uint32_t>(value);
}

bool range_fits(size_t offset, size_t size, size_t capacity) {
  return offset <= capacity && size <= capacity - offset;
}

template <typename T>
bool read_scalar_value(const CompileSpec& spec, T* out) {
  if (spec.value.buffer == nullptr || spec.value.nbytes != sizeof(T)) {
    return false;
  }
  std::memcpy(out, spec.value.buffer, sizeof(T));
  return true;
}

std::string read_string_value(const CompileSpec& spec) {
  if (spec.value.buffer == nullptr || spec.value.nbytes == 0) {
    return "";
  }
  const char* begin = static_cast<const char*>(spec.value.buffer);
  std::string result(begin, begin + spec.value.nbytes);
  while (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

LinuxDriverOptions parse_options(ArrayRef<CompileSpec> specs) {
  LinuxDriverOptions options;
  constexpr char kDeviceKey[] = "ethosu.device";
  constexpr char kTimeoutKey[] = "ethosu.timeout_ns";
  constexpr char kCycleCounterKey[] = "ethosu.enable_cycle_counter";
  constexpr char kPmuPrefix[] = "ethosu.pmu_event";

  for (const CompileSpec& spec : specs) {
    if (spec.key == nullptr) {
      continue;
    }
    if (strcmp(spec.key, kDeviceKey) == 0) {
      std::string path = read_string_value(spec);
      if (!path.empty()) {
        options.device_path = path;
      }
    } else if (strcmp(spec.key, kTimeoutKey) == 0) {
      int64_t timeout = 0;
      if (read_scalar_value(spec, &timeout) && timeout > 0) {
        options.timeout_ns = timeout;
      }
    } else if (strcmp(spec.key, kCycleCounterKey) == 0) {
      uint8_t enabled = 0;
      if (read_scalar_value(spec, &enabled)) {
        options.enable_cycle_counter = enabled != 0;
      }
    } else if (strncmp(spec.key, kPmuPrefix, strlen(kPmuPrefix)) == 0) {
      const char* idx_str = spec.key + strlen(kPmuPrefix);
      char* endptr = nullptr;
      long idx = std::strtol(idx_str, &endptr, 10);
      if (endptr != idx_str && *endptr == '\0' && idx >= 0 &&
          idx < static_cast<long>(ETHOSU_PMU_EVENT_MAX)) {
        uint32_t event = 0;
        if (read_scalar_value(spec, &event)) {
          options.pmu_events[static_cast<size_t>(idx)] = event;
        }
      }
    }
  }

  return options;
}

class DeviceCache {
 public:
  std::shared_ptr<EthosU::Device> get(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (path == path_) {
      if (auto device = device_.lock()) {
        return device;
      }
    }

    auto device = std::make_shared<EthosU::Device>(path.c_str());
    path_ = path;
    device_ = device;
    return device;
  }

 private:
  std::mutex mutex_;
  std::string path_;
  std::weak_ptr<EthosU::Device> device_;
};

DeviceCache& get_device_cache() {
  static DeviceCache cache;
  return cache;
}

[[maybe_unused]] const char* status_string(EthosU::InferenceStatus s) {
  switch (s) {
    case EthosU::InferenceStatus::OK:
      return "OK";
    case EthosU::InferenceStatus::ERROR:
      return "ERROR";
    case EthosU::InferenceStatus::RUNNING:
      return "RUNNING";
    case EthosU::InferenceStatus::REJECTED:
      return "REJECTED";
    case EthosU::InferenceStatus::ABORTED:
      return "ABORTED";
    case EthosU::InferenceStatus::ABORTING:
      return "ABORTING";
  }
  return "UNKNOWN";
}

} // namespace

PlatformState* platform_init(
    ArrayRef<CompileSpec> specs,
    MemoryAllocator* allocator) {
  (void)allocator;
  auto* state = new (std::nothrow) PlatformState();
  if (state == nullptr) {
    return nullptr;
  }
  state->options = parse_options(specs);
  return state;
}

void platform_destroy(PlatformState* state) {
  delete state;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* execution_handle,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* /*ethosu_scratch*/) {
  if (execution_handle == nullptr ||
      execution_handle->platform_state == nullptr) {
    ET_LOG(Error, "Ethos-U i.MX backend is not initialized.");
    return Error::InvalidState;
  }
  if (input_count < 0 || output_count < 0 || input_count > ETHOSU_FD_MAX ||
      output_count > ETHOSU_FD_MAX ||
      static_cast<size_t>(input_count) + static_cast<size_t>(output_count) >
          args.size()) {
    ET_LOG(Error, "Ethos-U i.MX received invalid tensor counts.");
    return Error::InvalidArgument;
  }
  if (handles.inputs == nullptr || handles.outputs == nullptr ||
      handles.inputs->count != input_count ||
      handles.outputs->count != output_count) {
    ET_LOG(Error, "Vela I/O metadata does not match the delegate arguments.");
    return Error::InvalidProgram;
  }
  if (handles.vela_model_data == nullptr || handles.vela_model_size == 0) {
    ET_LOG(
        Error,
        "The i.MX backend requires a vela_model block. Re-export the model "
        "with NXP model embedding enabled.");
    return Error::InvalidProgram;
  }
  for (size_t i = 0; i < static_cast<size_t>(input_count + output_count); ++i) {
    if (args[i] == nullptr) {
      ET_LOG(Error, "Ethos-U i.MX received a null tensor argument.");
      return Error::InvalidArgument;
    }
  }

  PlatformState& state = *execution_handle->platform_state;
  std::lock_guard<std::mutex> execution_lock(state.execution_mutex);

  try {
    if (!state.network) {
      auto device = get_device_cache().get(state.options.device_path);
      auto model_buffer =
          std::make_shared<EthosU::Buffer>(*device, handles.vela_model_size);
      model_buffer->resize(handles.vela_model_size);
      std::memcpy(
          model_buffer->data(),
          handles.vela_model_data,
          handles.vela_model_size);
      auto network = std::make_shared<EthosU::Network>(*device, model_buffer);

      const size_t network_input_count = network->getInputCount();
      const size_t network_output_count = network->getOutputCount();
      if (network_input_count != static_cast<size_t>(input_count) ||
          network_output_count != static_cast<size_t>(output_count) ||
          network_input_count > ETHOSU_FD_MAX ||
          network_output_count > ETHOSU_FD_MAX) {
        ET_LOG(
            Error,
            "The NXP network I/O counts do not match the delegate payload.");
        return Error::InvalidProgram;
      }

      // The NXP API names byte counts "dims".
      const auto& input_sizes = network->getIfmDims();
      const auto& output_sizes = network->getOfmDims();
      if (input_sizes.size() != network_input_count ||
          output_sizes.size() != network_output_count) {
        ET_LOG(Error, "The NXP network returned incomplete I/O metadata.");
        return Error::InvalidProgram;
      }

      EthosU::MemoryLayout layout{};
      layout.input_count = static_cast<uint32_t>(network_input_count);
      layout.output_count = static_cast<uint32_t>(network_output_count);
      size_t max_io_extent = 0;
      for (size_t i = 0; i < network_input_count; ++i) {
        const int32_t offset = network->getInputDataOffset(i);
        if (offset < 0) {
          ET_LOG(
              Error,
              "The i.MX backend does not support separate input regions.");
          return Error::InvalidProgram;
        }
        const size_t size = input_sizes[i];
        const size_t end = checked_add(
            static_cast<size_t>(offset), size, "Input range overflow");
        layout.input_offset[i] = static_cast<uint32_t>(offset);
        layout.input_size[i] = checked_u32(size, "Input size exceeds UAPI");
        max_io_extent = std::max(max_io_extent, end);
      }
      for (size_t i = 0; i < network_output_count; ++i) {
        const int32_t offset = network->getOutputDataOffset(i);
        if (offset < 0) {
          ET_LOG(
              Error,
              "The i.MX backend does not support separate output regions.");
          return Error::InvalidProgram;
        }
        const size_t size = output_sizes[i];
        const size_t end = checked_add(
            static_cast<size_t>(offset), size, "Output range overflow");
        layout.output_offset[i] = static_cast<uint32_t>(offset);
        layout.output_size[i] = checked_u32(size, "Output size exceeds UAPI");
        max_io_extent = std::max(max_io_extent, end);
      }

      // Scratch follows the highest I/O byte range.
      const size_t arena_offset = align_up(max_io_extent, kArenaDataAlignment);
      layout.arena_offset =
          checked_u32(arena_offset, "Arena offset exceeds UAPI");
      const size_t required_capacity = checked_add(
          arena_offset, handles.scratch_data_size, "Arena capacity overflow");
      const size_t arena_capacity = align_up(
          std::max(kDefaultArenaBytes, required_capacity), kBytesPerMiB);
      auto arena = std::make_shared<EthosU::Buffer>(*device, arena_capacity);
      arena->resize(arena_capacity);

      state.device = std::move(device);
      state.network = std::move(network);
      state.arena = std::move(arena);
      state.layout = layout;
      state.arena_capacity = arena_capacity;
      ET_LOG(
          Debug,
          "Ethos-U i.MX staged model=%zu, arena=%zu, inputs=%d, outputs=%d",
          handles.vela_model_size,
          arena_capacity,
          input_count,
          output_count);
    }

    if (!state.inference) {
      state.inference = std::make_shared<EthosU::Inference>(
          state.network,
          state.arena,
          state.options.pmu_events,
          state.options.enable_cycle_counter,
          state.layout);
    }

    char* const arena_data = state.arena->data();
    for (int i = 0; i < input_count; ++i) {
      const Tensor tensor = args[i]->toTensor();
      const size_t tensor_bytes = tensor.nbytes();
      const size_t driver_bytes = state.layout.input_size[i];
      const size_t offset = state.layout.input_offset[i];
      if (tensor_bytes != driver_bytes ||
          !range_fits(offset, tensor_bytes, state.arena_capacity)) {
        ET_LOG(
            Error,
            "Ethos-U input %d has %zu bytes; the driver requires %zu.",
            i,
            tensor_bytes,
            driver_bytes);
        return Error::InvalidArgument;
      }
      std::memcpy(
          arena_data + offset, tensor.const_data_ptr<char>(), tensor_bytes);
    }

    const bool timed_out = state.inference->invoke(state.options.timeout_ns);
    if (timed_out) {
      ET_LOG(
          Error,
          "Ethos-U inference timed out after %lld ns.",
          static_cast<long long>(state.options.timeout_ns));
      state.inference->cancel();
      state.inference.reset();
      return Error::InvalidState;
    }

    const auto status = state.inference->status();
    if (status != EthosU::InferenceStatus::OK) {
      ET_LOG(Error, "Ethos-U inference failed: %s", status_string(status));
      state.inference.reset();
      return Error::InvalidState;
    }

    if (state.options.enable_cycle_counter) {
      try {
        ET_LOG(
            Debug,
            "Ethos-U cycle counter: %llu",
            static_cast<unsigned long long>(
                state.inference->getCycleCounter()));
      } catch (const std::exception& error) {
        ET_LOG(
            Debug,
            "Could not read the Ethos-U cycle counter: %s",
            error.what());
      }
    }

    for (int i = 0; i < output_count; ++i) {
      Tensor tensor_out = args[input_count + i]->toTensor();
      const size_t tensor_bytes = tensor_out.nbytes();
      const size_t driver_bytes = state.layout.output_size[i];
      const size_t offset = state.layout.output_offset[i];
      if (tensor_bytes != driver_bytes ||
          !range_fits(offset, tensor_bytes, state.arena_capacity)) {
        ET_LOG(
            Error,
            "Ethos-U output %d has %zu bytes; the driver requires %zu.",
            i,
            tensor_bytes,
            driver_bytes);
        return Error::InvalidArgument;
      }

      const char* const output_data = arena_data + offset;
      int tensor_count = 1;
      int io_count = 1;
      calculate_dimensions(
          tensor_out, &handles.outputs->io[i], &tensor_count, &io_count);
      if (handles.outputs->io[i].elem_size <= 0 || io_count < 0) {
        ET_LOG(Error, "Ethos-U output %d has invalid Vela metadata.", i);
        return Error::InvalidProgram;
      }
      const size_t io_bytes = checked_multiply(
          static_cast<size_t>(io_count),
          static_cast<size_t>(handles.outputs->io[i].elem_size),
          "Output size overflow");

      if (tensor_bytes != io_bytes) {
        const Error adjustment = copy_with_layout_adjustment(
            handles.outputs->io[i], i, output_data, tensor_out, tensor_bytes);
        if (adjustment != Error::Ok) {
          return adjustment;
        }
      } else {
        std::memcpy(
            tensor_out.mutable_data_ptr<char>(), output_data, tensor_bytes);
      }
    }
    return Error::Ok;
  } catch (const std::exception& error) {
    state.inference.reset();
    ET_LOG(Error, "Ethos-U i.MX driver failed: %s", error.what());
    return Error::InvalidState;
  }
}

} // namespace arm
} // namespace backends
} // namespace executorch
