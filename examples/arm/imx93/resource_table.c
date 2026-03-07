/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Remoteproc resource table for i.MX93 M33 firmware.
 * RPMsg vdev required for the kernel's virtio/rpmsg subsystem.
 */

#include <stdint.h>
#include <string.h>

#define RSC_VDEV    3
#define VIRTIO_ID_RPMSG 7
#define VRING_ALIGN 0x1000

struct fw_rsc_vdev_vring {
    uint32_t da;
    uint32_t align;
    uint32_t num;
    uint32_t notifyid;
    uint32_t reserved;
};

struct fw_rsc_vdev {
    uint32_t type;
    uint32_t id;
    uint32_t notifyid;
    uint32_t dfeatures;
    uint32_t gfeatures;
    uint32_t config_len;
    uint8_t  status;
    uint8_t  num_of_vrings;
    uint8_t  reserved[2];
    struct fw_rsc_vdev_vring vring[2];
};

struct resource_table {
    uint32_t ver;
    uint32_t num;
    uint32_t reserved[2];
    uint32_t offset[1];
    struct fw_rsc_vdev vdev;
};

__attribute__((section(".resource_table"), used))
struct resource_table resources = {
    .ver = 1,
    .num = 1,
    .reserved = {0, 0},
    .offset = {
        __builtin_offsetof(struct resource_table, vdev),
    },
    .vdev = {
        .type = RSC_VDEV,
        .id = VIRTIO_ID_RPMSG,
        .notifyid = 0,
        .dfeatures = 1, /* VIRTIO_RPMSG_F_NS */
        .gfeatures = 0,
        .config_len = 0,
        .status = 0,
        .num_of_vrings = 2,
        .reserved = {0},
        .vring = {
            {.da = 0xA4000000, .align = VRING_ALIGN, .num = 8, .notifyid = 0, .reserved = 0},
            {.da = 0xA4008000, .align = VRING_ALIGN, .num = 8, .notifyid = 1, .reserved = 0},
        },
    },
};

/* Trace to DTCM — read from A55 at 0x20200000+offset */
char trace_buf[0x1000] __attribute__((section(".data")));
volatile unsigned int trace_idx __attribute__((section(".data"))) = 0;

void trace_init(void) {
    trace_idx = 0;
    memset(trace_buf, 0, sizeof(trace_buf));
}

void trace_write(const char *buf, int len) {
    for (int i = 0; i < len; i++) {
        if (trace_idx >= sizeof(trace_buf) - 1)
            break;
        trace_buf[trace_idx++] = buf[i];
    }
    trace_buf[trace_idx] = '\0';
}
