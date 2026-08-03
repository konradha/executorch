/*
 * Copyright 2023, 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Warning: Do not change this without changing arm_vela.py::vela_compile
 *          as that function emits this format and the two need to align.
 */

#include <executorch/backends/arm/runtime/VelaBinStream.h>

#include <cstring>

#include <executorch/runtime/core/error.h>

namespace executorch {
namespace backends {
namespace arm {

// Each header and payload starts on a 16-byte boundary. arm_vela.py emits the
// same padding, and the firmware command stream requires this alignment.
constexpr size_t kVelaBlockAlignment = 16;

bool block_name_is(const VelaBinBlock& block, const char* expected) {
  const size_t length = std::strlen(expected);
  return length <= kVelaBlockNameLength &&
      std::memcmp(block.name, expected, length) == 0 &&
      (length == kVelaBlockNameLength || block.name[length] == '\0');
}

bool next_block(
    const char** cursor,
    const char* end,
    const VelaBinBlock** block) {
  const size_t remaining = static_cast<size_t>(end - *cursor);
  if (remaining < sizeof(VelaBinBlock)) {
    ET_LOG(Error, "Truncated block header in vela_bin_stream");
    return false;
  }

  const auto* candidate = reinterpret_cast<const VelaBinBlock*>(*cursor);
  const size_t padded_size =
      (static_cast<size_t>(candidate->size) + kVelaBlockAlignment - 1) &
      ~(kVelaBlockAlignment - 1);
  if (padded_size > remaining - sizeof(VelaBinBlock)) {
    ET_LOG(
        Error,
        "Block %.16s extends past the end of vela_bin_stream",
        candidate->name);
    return false;
  }

  *cursor += sizeof(VelaBinBlock) + padded_size;
  *block = candidate;
  return true;
}

bool validate_io_block(const VelaBinBlock& block) {
  if (block.size < sizeof(int)) {
    return false;
  }

  int count = 0;
  std::memcpy(&count, block.data, sizeof(count));
  if (count < 0) {
    return false;
  }
  const size_t maximum_count = (block.size - sizeof(int)) / sizeof(VelaIO);
  return static_cast<size_t>(count) <= maximum_count;
}
bool mark_block_seen(bool* seen, const char* name) {
  if (*seen) {
    ET_LOG(Error, "Duplicate %s block in vela_bin_stream", name);
    return false;
  }
  *seen = true;
  return true;
}

bool vela_bin_validate(const char* data, int size) {
  if (data == nullptr || size < static_cast<int>(2 * sizeof(VelaBinBlock))) {
    ET_LOG(Error, "vela_bin_stream is null or too small");
    return false;
  }
  if (reinterpret_cast<uintptr_t>(data) % kVelaBlockAlignment != 0) {
    ET_LOG(Error, "Vela bin pointer is not aligned to 16 bytes: %p", data);
    return false;
  }
  if (static_cast<size_t>(size) % kVelaBlockAlignment != 0) {
    ET_LOG(Error, "Vela bin size is not aligned to 16 bytes: %d", size);
    return false;
  }

  const char* cursor = data;
  const char* const end = data + size;
  bool first = true;
  bool saw_cmd_data = false;
  bool saw_weight_data = false;
  bool saw_scratch_size = false;
  bool saw_inputs = false;
  bool saw_outputs = false;
  bool saw_vela_model = false;
  while (cursor < end) {
    const VelaBinBlock* block = nullptr;
    if (!next_block(&cursor, end, &block)) {
      return false;
    }
    if (first) {
      first = false;
      if (!block_name_is(*block, "vela_bin_stream")) {
        ET_LOG(Error, "Incorrect header in vela_bin_stream");
        return false;
      }
    }

    // COP1 and COP2 are the command-stream magic values from the Ethos-U
    // driver ABI. Vela writes one of them at the start of cmd_data.
    if (block_name_is(*block, "vela_bin_stream")) {
      if (reinterpret_cast<const char*>(block) != data) {
        ET_LOG(Error, "Duplicate header in vela_bin_stream");
        return false;
      }
    } else if (block_name_is(*block, "cmd_data")) {
      if (!mark_block_seen(&saw_cmd_data, "cmd_data") || block->size < 4 ||
          (std::memcmp(block->data, "COP1", 4) != 0 &&
           std::memcmp(block->data, "COP2", 4) != 0)) {
        ET_LOG(Error, "Invalid command data in vela_bin_stream");
        return false;
      }
    } else if (block_name_is(*block, "weight_data")) {
      if (!mark_block_seen(&saw_weight_data, "weight_data")) {
        return false;
      }
    } else if (block_name_is(*block, "scratch_size")) {
      if (!mark_block_seen(&saw_scratch_size, "scratch_size") ||
          block->size < sizeof(uint32_t)) {
        ET_LOG(Error, "Invalid scratch_size block in vela_bin_stream");
        return false;
      }
    } else if (block_name_is(*block, "inputs")) {
      if (!mark_block_seen(&saw_inputs, "inputs") ||
          !validate_io_block(*block)) {
        ET_LOG(Error, "Invalid inputs block in vela_bin_stream");
        return false;
      }
    } else if (block_name_is(*block, "outputs")) {
      if (!mark_block_seen(&saw_outputs, "outputs") ||
          !validate_io_block(*block)) {
        ET_LOG(Error, "Invalid outputs block in vela_bin_stream");
        return false;
      }
    } else if (block_name_is(*block, "vela_model")) {
      if (!mark_block_seen(&saw_vela_model, "vela_model")) {
        return false;
      }
    } else if (block_name_is(*block, "vela_end_stream")) {
      if (cursor != end) {
        ET_LOG(Error, "vela_end_stream is not the final block");
        return false;
      }
      if (!(saw_cmd_data && saw_weight_data && saw_scratch_size && saw_inputs &&
            saw_outputs)) {
        ET_LOG(Error, "vela_bin_stream is missing a required block");
        return false;
      }
      return true;
    }
  }

  ET_LOG(Error, "vela_bin_stream has no footer");
  return false;
}

bool vela_bin_read(const char* data, VelaHandles* handles, int size) {
  if (handles == nullptr || !vela_bin_validate(data, size)) {
    return false;
  }
  *handles = {};

  const char* cursor = data;
  const char* const end = data + size;
  while (cursor < end) {
    const VelaBinBlock* block = nullptr;
    if (!next_block(&cursor, end, &block)) {
      return false;
    }

    if (block_name_is(*block, "vela_bin_stream")) {
      if (reinterpret_cast<const char*>(block) != data) {
        return false;
      }
    } else if (block_name_is(*block, "cmd_data")) {
      handles->cmd_data = block->data;
      handles->cmd_data_size = block->size;
    } else if (block_name_is(*block, "weight_data")) {
      handles->weight_data = block->data;
      handles->weight_data_size = block->size;
    } else if (block_name_is(*block, "scratch_size")) {
      std::memcpy(&handles->scratch_data_size, block->data, sizeof(uint32_t));
    } else if (block_name_is(*block, "inputs")) {
      handles->inputs =
          reinterpret_cast<VelaIOs*>(const_cast<char*>(block->data));
    } else if (block_name_is(*block, "outputs")) {
      handles->outputs =
          reinterpret_cast<VelaIOs*>(const_cast<char*>(block->data));
    } else if (block_name_is(*block, "vela_model")) {
      handles->vela_model_data = block->data;
      handles->vela_model_size = block->size;
    } else if (block_name_is(*block, "vela_end_stream")) {
      return cursor == end;
    } else {
      ET_LOG(
          Debug,
          "Skipping unknown block in vela_bin_stream: %.16s",
          block->name);
    }
  }
  return false;
}

} // namespace arm
} // namespace backends
} // namespace executorch
