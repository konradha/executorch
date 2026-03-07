/*
 * SPDX-License-Identifier: MIT
 *
 * A55-side helper for loading a PTE model into DDR and signaling the M33
 * via the DTCM mailbox. Run on the i.MX93 Linux side after remoteproc start.
 *
 * Usage:
 *   gcc -o ddr_loader ddr_loader.c
 *   sudo ./ddr_loader <model.pte> <mailbox_a55_addr> <diag_a55_addr>
 *
 * Addresses come from the ELF:
 *   MAILBOX = nm output for ddr_pte_mailbox + 0x200000
 *   DIAG    = nm output for executorch_diag_words + 0x200000
 *
 * Example:
 *   arm-none-eabi-nm arm_executor_runner | grep -E 'ddr_pte_mailbox|executorch_diag_words'
 *   # ddr_pte_mailbox = 0x200051F0  =>  A55: 0x202051F0
 *   # executorch_diag_words = 0x200050F0  =>  A55: 0x202050F0
 *   sudo ./ddr_loader model.pte 0x202051F0 0x202050F0
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* DMA heap allocation ioctl — from linux/dma-heap.h */
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)

/* Get physical address from pagemap for a userspace virtual address */
static uint64_t virt_to_phys(void *vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) { perror("pagemap"); return 0; }

    uintptr_t va = (uintptr_t)vaddr;
    uint64_t page_size = sysconf(_SC_PAGESIZE);
    uint64_t offset = (va / page_size) * 8;

    uint64_t entry = 0;
    if (pread(fd, &entry, 8, offset) != 8) {
        perror("pagemap read");
        close(fd);
        return 0;
    }
    close(fd);

    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "page not present for %p\n", vaddr);
        return 0;
    }

    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return pfn * page_size + (va % page_size);
}

/* Write a 32-bit word to a physical address via /dev/mem */
static int devmem_write32(uint64_t phys, uint32_t value) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return -1; }

    uint64_t page_size = sysconf(_SC_PAGESIZE);
    uint64_t page_base = phys & ~(page_size - 1);
    uint64_t page_off = phys - page_base;

    void *map = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, page_base);
    if (map == MAP_FAILED) { perror("mmap /dev/mem"); close(fd); return -1; }

    *(volatile uint32_t *)((char *)map + page_off) = value;

    munmap(map, page_size);
    close(fd);
    return 0;
}

/* Read a 32-bit word from a physical address via /dev/mem */
static uint32_t devmem_read32(uint64_t phys) {
    int fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return 0; }

    uint64_t page_size = sysconf(_SC_PAGESIZE);
    uint64_t page_base = phys & ~(page_size - 1);
    uint64_t page_off = phys - page_base;

    void *map = mmap(NULL, page_size, PROT_READ, MAP_SHARED, fd, page_base);
    if (map == MAP_FAILED) { perror("mmap /dev/mem read"); close(fd); return 0; }

    uint32_t val = *(volatile uint32_t *)((char *)map + page_off);

    munmap(map, page_size);
    close(fd);
    return val;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.pte> <mailbox_a55_addr> <diag_a55_addr>\n", argv[0]);
        fprintf(stderr, "\nAddresses from ELF (M33 addr + 0x200000):\n");
        fprintf(stderr, "  arm-none-eabi-nm arm_executor_runner | "
                        "grep -E 'ddr_pte_mailbox|executorch_diag_words'\n");
        return 1;
    }

    const char *pte_path = argv[1];
    uint64_t mailbox_addr = strtoull(argv[2], NULL, 0);
    uint64_t diag_addr = strtoull(argv[3], NULL, 0);

    /* Read PTE file */
    int pte_fd = open(pte_path, O_RDONLY);
    if (pte_fd < 0) { perror(pte_path); return 1; }

    struct stat st;
    fstat(pte_fd, &st);
    size_t pte_size = st.st_size;
    printf("PTE: %s (%zu bytes)\n", pte_path, pte_size);

    /* Allocate CMA buffer via DMA heap */
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
    if (heap_fd < 0) {
        /* fallback name */
        heap_fd = open("/dev/dma_heap/reserved", O_RDWR);
    }
    if (heap_fd < 0) { perror("dma_heap open"); close(pte_fd); return 1; }

    struct dma_heap_allocation_data alloc = {
        .len = pte_size,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        perror("DMA_HEAP_IOCTL_ALLOC");
        close(heap_fd);
        close(pte_fd);
        return 1;
    }
    close(heap_fd);

    int dma_fd = alloc.fd;
    void *cma_buf = mmap(NULL, pte_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         dma_fd, 0);
    if (cma_buf == MAP_FAILED) {
        perror("mmap dma buf");
        close(dma_fd);
        close(pte_fd);
        return 1;
    }

    /* Copy PTE into CMA buffer */
    size_t total = 0;
    while (total < pte_size) {
        ssize_t n = read(pte_fd, (char *)cma_buf + total, pte_size - total);
        if (n <= 0) { perror("read pte"); return 1; }
        total += n;
    }
    close(pte_fd);

    /* Get physical address */
    uint64_t phys = virt_to_phys(cma_buf);
    if (phys == 0) {
        fprintf(stderr, "Failed to get physical address\n");
        return 1;
    }
    printf("CMA buffer: virt=%p phys=0x%lx size=%zu\n",
           cma_buf, (unsigned long)phys, pte_size);

    /* Write mailbox to DTCM:
     *   offset +0: magic  = 0x50544544 ("PTED")
     *   offset +4: pte_phys_addr
     *   offset +8: pte_size
     *   offset +12: status (0 = pending)
     *
     * Write fields BEFORE magic so M33 sees a consistent snapshot. */
    devmem_write32(mailbox_addr + 12, 0);                   /* status */
    devmem_write32(mailbox_addr + 8, (uint32_t)pte_size);   /* size */
    devmem_write32(mailbox_addr + 4, (uint32_t)phys);       /* phys addr */
    devmem_write32(mailbox_addr + 0, 0x50544544u);          /* magic last */

    printf("Mailbox written at 0x%lx: magic=PTED phys=0x%lx size=%zu\n",
           (unsigned long)mailbox_addr, (unsigned long)phys, pte_size);

    /* Poll diag slot[1] (stage) for completion */
    printf("Polling M33 diag stage (slot[1] at 0x%lx)...\n",
           (unsigned long)(diag_addr + 4));

    uint32_t prev_stage = 0;
    for (int i = 0; i < 120; i++) {
        sleep(1);
        uint32_t stage = devmem_read32(diag_addr + 1 * 4);
        uint32_t mbox_status = devmem_read32(mailbox_addr + 12);
        if (stage != prev_stage) {
            printf("  stage=0x%08x status=%u (t=%ds)\n", stage, mbox_status, i);
            prev_stage = stage;
        }
        /* kStageRunModelDone = 0xA1000023 */
        if (stage == 0xA1000023u) {
            printf("Inference complete!\n");
            /* Read output summary: argmax in slot[12], count in slot[38] */
            uint32_t argmax = devmem_read32(diag_addr + 12 * 4);
            uint32_t count = devmem_read32(diag_addr + 38 * 4);
            printf("  argmax=%u total_floats=%u\n", argmax, count);
            /* Dump first 10 output values (slots 2..11, as float bits) */
            printf("  first outputs (IEEE754 hex): ");
            for (int s = 2; s < 12 && s < (int)(2 + (count < 10 ? count : 10)); s++) {
                printf("0x%08x ", devmem_read32(diag_addr + s * 4));
            }
            printf("\n");
            break;
        }
        /* Check for fault */
        if ((stage & 0xFA170000u) == 0xFA170000u) {
            printf("FAULT detected: stage=0x%08x\n", stage);
            break;
        }
    }

    /* Keep CMA buffer mapped — M33/NPU may still be DMA-ing from it.
     * The buffer is freed when this process exits. */
    printf("Done. Ctrl-C to release CMA buffer.\n");
    pause();

    munmap(cma_buf, pte_size);
    close(dma_fd);
    return 0;
}
