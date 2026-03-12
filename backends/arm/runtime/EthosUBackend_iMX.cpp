/*
 * SPDX-License-Identifier: MIT
 *
 * Arm backend for NXP i.MX Ethos-U Linux driver stack.
 * Uses /dev/ethosu0 via the NXP ethos-u-driver-stack-imx userspace library.
 *
 * Unlike the bare-metal path (which passes raw command streams and base
 * addresses), NXP's Linux driver expects a full Vela-compiled model in a
 * DMA buffer and uses a single shared arena for I/O and scratch.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <ethosu.hpp>
#include <uapi/ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

constexpr int64_t kDefaultTimeoutNs = 60000000000LL;

struct LinuxDriverOptions {
  std::string device_path = "/dev/ethosu0";
  int64_t timeout_ns = kDefaultTimeoutNs;
  bool enable_cycle_counter = true;
  std::array<uint32_t, ETHOSU_PMU_EVENT_MAX> pmu_events{};
};

struct PlatformState {
  LinuxDriverOptions options;
  std::shared_ptr<EthosU::Network> network;
  std::shared_ptr<EthosU::Buffer> arena;
};

namespace {

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

  for (const CompileSpec& spec : specs) {
    if (spec.key == nullptr) {
      continue;
    }
    if (strcmp(spec.key, "ethosu.device") == 0) {
      std::string path = read_string_value(spec);
      if (!path.empty()) {
        options.device_path = path;
      }
    } else if (strcmp(spec.key, "ethosu.timeout_ns") == 0) {
      int64_t timeout = 0;
      if (read_scalar_value(spec, &timeout) && timeout > 0) {
        options.timeout_ns = timeout;
      }
    } else if (strcmp(spec.key, "ethosu.enable_cycle_counter") == 0) {
      uint8_t enabled = 0;
      if (read_scalar_value(spec, &enabled)) {
        options.enable_cycle_counter = enabled != 0;
      }
    } else if (strncmp(spec.key, "ethosu.pmu_event", 16) == 0) {
      const char* idx_str = spec.key + 16;
      char* endptr = nullptr;
      long idx = std::strtol(idx_str, &endptr, 10);
      if (endptr != idx_str && idx >= 0 &&
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
  EthosU::Device& get(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!device_ || path != path_) {
      device_ = std::make_unique<EthosU::Device>(path.c_str());
      path_ = path;
    }
    return *device_;
  }

 private:
  std::mutex mutex_;
  std::string path_;
  std::unique_ptr<EthosU::Device> device_;
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
  PlatformState* state = execution_handle->platform_state;
  if (state == nullptr) {
    ET_LOG(Error, "Ethos-U i.MX backend missing platform state");
    return Error::InvalidState;
  }

  if (handles.vela_model_data == nullptr || handles.vela_model_size == 0) {
    ET_LOG(
        Error,
        "i.MX backend requires vela_model block in delegate payload. "
        "Re-export with updated arm_vela.py.");
    return Error::InvalidProgram;
  }

  try {
    if (!state->network) {
      EthosU::Device& device =
          get_device_cache().get(state->options.device_path);

      auto model_buf = std::make_shared<EthosU::Buffer>(
          device, handles.vela_model_size);
      model_buf->resize(handles.vela_model_size);
      std::memcpy(
          model_buf->data(), handles.vela_model_data, handles.vela_model_size);
      state->network = std::make_shared<EthosU::Network>(device, model_buf);

      size_t arena_capacity = handles.scratch_data_size;
      size_t n_ifm = state->network->getInputCount();
      size_t n_ofm = state->network->getOutputCount();
      for (size_t i = 0; i < n_ifm; i++) {
        size_t extent = static_cast<size_t>(
            state->network->getInputDataOffset(i)) +
            state->network->getIfmDims()[i];
        if (extent > arena_capacity) {
          arena_capacity = extent;
        }
      }
      for (size_t i = 0; i < n_ofm; i++) {
        size_t extent = static_cast<size_t>(
            state->network->getOutputDataOffset(i)) +
            state->network->getOfmDims()[i];
        if (extent > arena_capacity) {
          arena_capacity = extent;
        }
      }
      if (arena_capacity < (1u << 20)) {
        arena_capacity = 1u << 20;
      }
      constexpr size_t kMinArenaMB = 16u << 20;
      if (arena_capacity < kMinArenaMB) {
        arena_capacity = kMinArenaMB;
      }
      arena_capacity = (arena_capacity + ((1u << 20) - 1)) & ~((1u << 20) - 1);
      ET_LOG(
          Info,
          "Ethos-U arena: scratch=%zu, computed=%zu, ifm=%zu, ofm=%zu",
          static_cast<size_t>(handles.scratch_data_size),
          arena_capacity,
          n_ifm,
          n_ofm);
      state->arena =
          std::make_shared<EthosU::Buffer>(device, arena_capacity);
      state->arena->resize(arena_capacity);
    }

    for (int i = 0; i < input_count; i++) {
      auto tensor = args[i]->toTensor();
      int32_t offset = state->network->getInputDataOffset(i);
      size_t ifm_size = state->network->getIfmDims()[i];
      ET_LOG(
          Info,
          "  IFM[%d]: offset=%d, driver_size=%zu, tensor_bytes=%zu",
          i, offset, ifm_size, static_cast<size_t>(tensor.nbytes()));
      std::memcpy(
          state->arena->data() + offset,
          tensor.const_data_ptr<char>(),
          tensor.nbytes());
    }

    EthosU::MemoryLayout layout = {};
    layout.input_count = static_cast<uint32_t>(input_count);
    for (int i = 0; i < input_count; i++) {
      layout.input_offset[i] = state->network->getInputDataOffset(i);
      layout.input_size[i] = state->network->getIfmDims()[i];
    }
    layout.output_count = static_cast<uint32_t>(output_count);
    for (int i = 0; i < output_count; i++) {
      layout.output_offset[i] = state->network->getOutputDataOffset(i);
      layout.output_size[i] = state->network->getOfmDims()[i];
      ET_LOG(
          Info,
          "  OFM[%d]: offset=%u, driver_size=%u",
          i,
          static_cast<unsigned>(layout.output_offset[i]),
          static_cast<unsigned>(layout.output_size[i]));
    }

    auto inference = std::make_shared<EthosU::Inference>(
        state->network,
        state->arena,
        state->options.pmu_events,
        state->options.enable_cycle_counter,
        layout);
    ET_LOG(Info, "Ethos-U i.MX inference object created");

    ET_LOG(Info, "Ethos-U i.MX invoking driver");
    bool timed_out = inference->invoke(state->options.timeout_ns);
    ET_LOG(Info, "Ethos-U i.MX invoke completed");
    if (timed_out) {
      ET_LOG(
          Error,
          "Ethos-U inference timed out after %lld ns",
          static_cast<long long>(state->options.timeout_ns));
      return Error::InvalidState;
    }

    ET_LOG(Info, "Ethos-U i.MX querying inference status");
    auto status = inference->status();
    ET_LOG(Info, "Ethos-U i.MX status query completed");
    if (status != EthosU::InferenceStatus::OK) {
      ET_LOG(
          Error,
          "Ethos-U inference failed: %s",
          status_string(status));
      return Error::InvalidState;
    }

    if (state->options.enable_cycle_counter) {
      try {
        ET_LOG(Info, "Ethos-U i.MX querying cycle counter");
        ET_LOG(
            Info,
            "Ethos-U cycle counter: %llu",
            static_cast<unsigned long long>(inference->getCycleCounter()));
      } catch (...) {
      }
    }

    if (handles.outputs == nullptr) {
      ET_LOG(Error, "Ethos-U backend missing output metadata");
      return Error::InvalidProgram;
    }

    ET_LOG(Info, "Ethos-U i.MX copying outputs");
    for (int i = 0; i < output_count; i++) {
      auto tensor_out = args[input_count + i]->toTensor();
      int32_t offset = state->network->getOutputDataOffset(i);
      const char* output_addr = state->arena->data() + offset;

      int tensor_count = 1, io_count = 1;
      calculate_dimensions(
          tensor_out, &handles.outputs->io[i], &tensor_count, &io_count);
      size_t io_bytes = static_cast<size_t>(io_count) *
          static_cast<size_t>(handles.outputs->io[i].elem_size);
      const size_t tensor_bytes = tensor_out.nbytes();

      if (tensor_bytes != io_bytes) {
        Error adj = copy_with_layout_adjustment(
            handles.outputs->io[i], i, output_addr, tensor_out, tensor_bytes);
        if (adj != Error::Ok) {
          return adj;
        }
      } else {
        std::memcpy(
            tensor_out.mutable_data_ptr<char>(), output_addr, tensor_bytes);
      }
    }

  } catch (const std::exception& e) {
    ET_LOG(Error, "Ethos-U i.MX driver failed: %s", e.what());
    return Error::InvalidState;
  }

  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
