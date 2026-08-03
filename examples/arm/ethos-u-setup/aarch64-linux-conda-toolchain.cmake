# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(CMAKE_VERSION VERSION_LESS 3.20)
  message(FATAL_ERROR "This toolchain file requires at least CMake 3.20")
endif()

set(CONDA_TOOLCHAIN_ROOT
    ""
    CACHE
      PATH
      "Micromamba or conda environment with the linux-aarch64 cross toolchain"
)
if(CONDA_TOOLCHAIN_ROOT STREQUAL "" AND DEFINED ENV{CONDA_PREFIX})
  set(CONDA_TOOLCHAIN_ROOT "$ENV{CONDA_PREFIX}")
endif()
if(CONDA_TOOLCHAIN_ROOT STREQUAL "")
  message(
    FATAL_ERROR
      "CONDA_TOOLCHAIN_ROOT is required. Activate the cross environment or "
      "pass -DCONDA_TOOLCHAIN_ROOT=/path/to/env."
  )
endif()

set(CONDA_LINUX_SYSROOT
    ""
    CACHE PATH "linux-aarch64 sysroot from the micromamba or conda environment"
)
if(CONDA_LINUX_SYSROOT STREQUAL "" AND DEFINED ENV{CONDA_BUILD_SYSROOT})
  set(CONDA_LINUX_SYSROOT "$ENV{CONDA_BUILD_SYSROOT}")
endif()
if(CONDA_LINUX_SYSROOT STREQUAL "")
  set(CONDA_LINUX_SYSROOT
      "${CONDA_TOOLCHAIN_ROOT}/aarch64-conda-linux-gnu/sysroot"
  )
endif()
if(NOT EXISTS "${CONDA_LINUX_SYSROOT}")
  message(
    FATAL_ERROR
      "The linux-aarch64 sysroot was not found. Install "
      "sysroot_linux-aarch64 or pass "
      "-DCONDA_LINUX_SYSROOT=/path/to/sysroot."
  )
endif()

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    CONDA_TOOLCHAIN_ROOT CONDA_LINUX_SYSROOT
)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_SYSROOT
    "${CONDA_LINUX_SYSROOT}"
    CACHE PATH "linux-aarch64 sysroot"
)
set(CMAKE_C_COMPILER
    "${CONDA_TOOLCHAIN_ROOT}/bin/aarch64-conda-linux-gnu-gcc"
    CACHE FILEPATH "linux-aarch64 C compiler"
)
set(CMAKE_CXX_COMPILER
    "${CONDA_TOOLCHAIN_ROOT}/bin/aarch64-conda-linux-gnu-g++"
    CACHE FILEPATH "linux-aarch64 C++ compiler"
)
set(CMAKE_AR
    "${CONDA_TOOLCHAIN_ROOT}/bin/aarch64-conda-linux-gnu-gcc-ar"
    CACHE FILEPATH "linux-aarch64 archiver"
)
set(CMAKE_RANLIB
    "${CONDA_TOOLCHAIN_ROOT}/bin/aarch64-conda-linux-gnu-gcc-ranlib"
    CACHE FILEPATH "linux-aarch64 ranlib"
)
set(CMAKE_STRIP
    "${CONDA_TOOLCHAIN_ROOT}/bin/aarch64-conda-linux-gnu-strip"
    CACHE FILEPATH "linux-aarch64 strip"
)

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
