# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Platform target for NXP i.MX93 (Cortex-M33 + Ethos-U65).
#
# Provides the ethosu_target_init and ethosu_target_common targets
# that the executor_runner CMakeLists.txt links against.

cmake_minimum_required(VERSION 3.20)

set(ETHOS_SDK_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U SDK"
)

# --- CMSIS and Ethos-U driver from the SDK ---
set(CMSIS_PATH "${ETHOS_SDK_PATH}/core_software/cmsis_6")
set(CORTEX_DFP_PATH "${ETHOS_SDK_PATH}/core_software/Cortex_DFP")
set(ETHOSU_DRIVER_PATH "${ETHOS_SDK_PATH}/core_software/core_driver")

# Build the Ethos-U core driver
add_subdirectory(${ETHOSU_DRIVER_PATH} ethosu_core_driver)

# --- Interface library matching core_platform's ethosu_target_common ---
add_library(ethosu_target_common INTERFACE)
target_include_directories(ethosu_target_common INTERFACE
    ${CMSIS_PATH}/CMSIS/Core/Include
    ${CMSIS_PATH}/CMSIS/Core/Include/m-profile
    ${CORTEX_DFP_PATH}/Device/ARMCM33/Include
)
# Force-include the CMSIS device header (same as cmsis.cmake does for Corstone).
# This makes DWT, CoreDebug, __CORTEX_M etc. available in all translation units.
target_compile_options(ethosu_target_common INTERFACE
    "$<$<COMPILE_LANGUAGE:C>:-includeARMCM33.h>"
    "$<$<COMPILE_LANGUAGE:CXX>:-includeARMCM33.h>"
)
# M33 has no DSP/MVE — define __CM33_REV, __FPU_PRESENT for CMSIS
target_compile_definitions(ethosu_target_common INTERFACE
    ARMCM33
    __CM33_REV=0x0000U
    __FPU_PRESENT=1U
    __MPU_PRESENT=1U
    __SAUREGION_PRESENT=0U
    __DSP_PRESENT=0U
    __NVIC_PRIO_BITS=3U
    __Vendor_SysTickConfig=0U
    # Model placement: MODEL=1 → DDR (ITCM is only 128KB, too small for code+model)
    ETHOSU_MODEL=1
)

# --- CMSIS startup for Cortex-M33 ---
set(CMSIS_STARTUP_SRC
    "${CORTEX_DFP_PATH}/Device/ARMCM33/Source/startup_ARMCM33.c"
)

# --- Platform init library ---
add_library(ethosu_target_init STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_table.c
    ${CMSIS_STARTUP_SRC}
)
target_link_libraries(ethosu_target_init PUBLIC
    ethosu_target_common
    ethosu_core_driver
)
target_include_directories(ethosu_target_init PRIVATE
    ${CMSIS_PATH}/CMSIS/Core/Include
    ${CMSIS_PATH}/CMSIS/Core/Include/m-profile
    ${CORTEX_DFP_PATH}/Device/ARMCM33/Include
)

# retarget.c — minimal stubs for _write/_read so printf works
# (nosys.specs provides these, but we add a hook for trace buffer output)
