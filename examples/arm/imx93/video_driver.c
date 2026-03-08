/*
 * SPDX-License-Identifier: MIT
 *
 * A55-side video inference driver for i.MX93 Cortex-M33 + Ethos-U65.
 * Loads a PTE model via DDR mailbox, then feeds preprocessed frames through
 * the NPU one at a time, collecting outputs and timing.
 *
 * Usage:
 *   gcc -o video_driver video_driver.c -O2 -lm
 *   sudo ./video_driver <model.pte> <frames_dir> <output_dir> \
 *        <mailbox_addr> <diag_addr> <video_mbox_addr> \
 *        <input_size> <output_size>
 *
 * frames_dir contains frame_0000.bin, frame_0001.bin, ... (raw float32 tensors)
 * output_dir receives out_0000.bin, out_0001.bin, ... + timing.csv
 *
 * Addresses from ELF (M33 addr + 0x200000):
 *   arm-none-eabi-nm arm_executor_runner | \
 *     grep -E 'ddr_pte_mailbox|executorch_diag_words|video_mailbox'
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* DMA heap allocation ioctl */
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)

/* VideoMailbox — must match firmware struct */
struct video_mailbox {
    uint32_t magic;          /* 0x56494446 ("VIDF") */
    uint32_t input_phys;
    uint32_t input_size;
    uint32_t output_phys;
    uint32_t output_size;
    uint32_t command;        /* 0=idle, 1=infer, 0xFF=stop */
    uint32_t status;         /* 0=idle, 1=busy, 2=done, 0xEE=error */
    uint32_t frame_id;
    uint32_t infer_cycles;
    uint32_t output_actual;
};

/* ---- /dev/mem helpers ---- */

static void devmem_write32(uint64_t phys, uint32_t value) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem write"); return; }
    uint64_t ps = sysconf(_SC_PAGESIZE);
    uint64_t base = phys & ~(ps - 1);
    void *map = mmap(NULL, ps, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
    if (map != MAP_FAILED) {
        *(volatile uint32_t *)((char *)map + (phys - base)) = value;
        munmap(map, ps);
    }
    close(fd);
}

static uint32_t devmem_read32(uint64_t phys) {
    int fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (fd < 0) { perror("/dev/mem read"); return 0; }
    uint64_t ps = sysconf(_SC_PAGESIZE);
    uint64_t base = phys & ~(ps - 1);
    void *map = mmap(NULL, ps, PROT_READ, MAP_SHARED, fd, base);
    uint32_t val = 0;
    if (map != MAP_FAILED) {
        val = *(volatile uint32_t *)((char *)map + (phys - base));
        munmap(map, ps);
    }
    close(fd);
    return val;
}

/* ---- CMA allocation ---- */

struct cma_buf {
    int dma_fd;
    void *virt;
    uint64_t phys;
    size_t size;
};

static uint64_t virt_to_phys(void *vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    uintptr_t va = (uintptr_t)vaddr;
    uint64_t ps = sysconf(_SC_PAGESIZE);
    uint64_t entry = 0;
    if (pread(fd, &entry, 8, (va / ps) * 8) != 8) { close(fd); return 0; }
    close(fd);
    if (!(entry & (1ULL << 63))) return 0;
    return (entry & ((1ULL << 55) - 1)) * ps + (va % ps);
}

static int cma_alloc(struct cma_buf *buf, size_t size) {
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
    if (heap_fd < 0) heap_fd = open("/dev/dma_heap/reserved", O_RDWR);
    if (heap_fd < 0) { perror("dma_heap"); return -1; }

    struct dma_heap_allocation_data alloc = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        perror("DMA_HEAP_IOCTL_ALLOC");
        close(heap_fd);
        return -1;
    }
    close(heap_fd);

    buf->dma_fd = alloc.fd;
    buf->size = size;
    buf->virt = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.fd, 0);
    if (buf->virt == MAP_FAILED) { perror("mmap cma"); close(alloc.fd); return -1; }
    buf->phys = virt_to_phys(buf->virt);
    if (buf->phys == 0) { fprintf(stderr, "virt_to_phys failed\n"); return -1; }
    return 0;
}

static void cma_free(struct cma_buf *buf) {
    if (buf->virt && buf->virt != MAP_FAILED) munmap(buf->virt, buf->size);
    if (buf->dma_fd >= 0) close(buf->dma_fd);
}

/* ---- Frame counting ---- */

static int count_frames(const char *dir) {
    int count = 0;
    char path[512];
    for (int i = 0; i < 100000; i++) {
        snprintf(path, sizeof(path), "%s/frame_%04d.bin", dir, i);
        if (access(path, F_OK) != 0) break;
        count++;
    }
    return count;
}

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr,
            "Usage: %s <model.pte> <frames_dir> <output_dir> "
            "<pte_mbox_addr> <diag_addr> <video_mbox_addr> "
            "<input_size> <output_size>\n\n"
            "Addresses from ELF (M33 addr + 0x200000):\n"
            "  arm-none-eabi-nm arm_executor_runner | \\\n"
            "    grep -E 'ddr_pte_mailbox|executorch_diag_words|video_mailbox'\n",
            argv[0]);
        return 1;
    }

    const char *pte_path = argv[1];
    const char *frames_dir = argv[2];
    const char *output_dir = argv[3];
    uint64_t pte_mbox_addr = strtoull(argv[4], NULL, 0);
    uint64_t diag_addr = strtoull(argv[5], NULL, 0);
    uint64_t vmbox_addr = strtoull(argv[6], NULL, 0);
    uint32_t input_size = strtoul(argv[7], NULL, 0);
    uint32_t output_size = strtoul(argv[8], NULL, 0);

    /* Count frames */
    int num_frames = count_frames(frames_dir);
    if (num_frames == 0) {
        fprintf(stderr, "No frame_XXXX.bin files in %s\n", frames_dir);
        return 1;
    }
    printf("Found %d frames in %s\n", num_frames, frames_dir);

    /* Create output directory */
    mkdir(output_dir, 0755);

    /* Read PTE file */
    int pte_fd = open(pte_path, O_RDONLY);
    if (pte_fd < 0) { perror(pte_path); return 1; }
    struct stat st;
    fstat(pte_fd, &st);
    size_t pte_size = st.st_size;

    /* Allocate CMA buffers: PTE + input + output */
    struct cma_buf pte_buf = {0}, in_buf = {0}, out_buf = {0};

    printf("Allocating CMA: PTE=%zu, input=%u, output=%u\n",
           pte_size, input_size, output_size);

    if (cma_alloc(&pte_buf, pte_size) < 0) return 1;
    if (cma_alloc(&in_buf, input_size) < 0) return 1;
    if (cma_alloc(&out_buf, output_size) < 0) return 1;

    printf("CMA PTE:    phys=0x%lx\n", (unsigned long)pte_buf.phys);
    printf("CMA input:  phys=0x%lx\n", (unsigned long)in_buf.phys);
    printf("CMA output: phys=0x%lx\n", (unsigned long)out_buf.phys);

    /* Copy PTE to CMA */
    size_t total = 0;
    while (total < pte_size) {
        ssize_t n = read(pte_fd, (char *)pte_buf.virt + total, pte_size - total);
        if (n <= 0) { perror("read pte"); return 1; }
        total += n;
    }
    close(pte_fd);

    /* Write PTE mailbox (same protocol as ddr_loader) */
    devmem_write32(pte_mbox_addr + 12, 0);
    devmem_write32(pte_mbox_addr + 8, (uint32_t)pte_size);
    devmem_write32(pte_mbox_addr + 4, (uint32_t)pte_buf.phys);
    devmem_write32(pte_mbox_addr + 0, 0x50544544u); /* PTED */
    printf("PTE mailbox written (%zu bytes)\n", pte_size);

    /* Wait for M33 to finish initial inference (stage = kStageRunModelDone) */
    printf("Waiting for initial model run...\n");
    for (int i = 0; i < 300; i++) {
        usleep(100000); /* 100ms */
        uint32_t stage = devmem_read32(diag_addr + 1 * 4);
        if (stage == 0xA1000023u) {
            printf("Initial inference done (t=%dms)\n", i * 100);
            break;
        }
        if ((stage & 0xFA170000u) == 0xFA170000u) {
            fprintf(stderr, "FAULT: stage=0x%08x\n", stage);
            return 1;
        }
    }

    /* Set up video mailbox — write fields before magic */
    devmem_write32(vmbox_addr + 36, 0);                     /* output_actual */
    devmem_write32(vmbox_addr + 32, 0);                     /* infer_cycles */
    devmem_write32(vmbox_addr + 28, 0);                     /* frame_id */
    devmem_write32(vmbox_addr + 24, 0);                     /* status = idle */
    devmem_write32(vmbox_addr + 20, 0);                     /* command = idle */
    devmem_write32(vmbox_addr + 16, output_size);           /* output_size */
    devmem_write32(vmbox_addr + 12, (uint32_t)out_buf.phys);/* output_phys */
    devmem_write32(vmbox_addr + 8,  input_size);            /* input_size */
    devmem_write32(vmbox_addr + 4,  (uint32_t)in_buf.phys); /* input_phys */
    devmem_write32(vmbox_addr + 0,  0x56494446u);           /* magic = "VIDF" */
    printf("Video mailbox activated\n");

    /* Wait for M33 to see video mailbox and set status=idle */
    usleep(500000);

    /* Open timing CSV */
    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/timing.csv", output_dir);
    FILE *csv = fopen(csv_path, "w");
    fprintf(csv, "frame,wall_ms,m33_cycles,argmax,output_bytes\n");

    /* Process frames */
    double total_wall = 0;
    uint64_t total_cycles = 0;

    for (int f = 0; f < num_frames; f++) {
        char frame_path[512];
        snprintf(frame_path, sizeof(frame_path), "%s/frame_%04d.bin", frames_dir, f);

        /* Read frame into CMA input buffer */
        int ffd = open(frame_path, O_RDONLY);
        if (ffd < 0) { perror(frame_path); continue; }
        ssize_t got = read(ffd, in_buf.virt, input_size);
        close(ffd);
        if (got != (ssize_t)input_size) {
            fprintf(stderr, "Frame %d: expected %u bytes, got %zd\n",
                    f, input_size, got);
            continue;
        }

        /* Signal M33: command=1 (infer) */
        double t_start = time_ms();
        devmem_write32(vmbox_addr + 24, 0);  /* status = idle */
        devmem_write32(vmbox_addr + 20, 1);  /* command = infer */

        /* Poll for completion */
        uint32_t status = 0;
        for (int w = 0; w < 10000; w++) {
            usleep(100); /* 0.1ms */
            status = devmem_read32(vmbox_addr + 24); /* status */
            if (status == 2 || status == 0xEEu) break;
        }
        double t_end = time_ms();
        double wall_ms = t_end - t_start;

        if (status != 2) {
            fprintf(stderr, "Frame %d: timeout/error (status=0x%x)\n", f, status);
            continue;
        }

        /* Read results */
        uint32_t m33_cycles = devmem_read32(vmbox_addr + 32);
        uint32_t frame_id = devmem_read32(vmbox_addr + 28);
        uint32_t out_actual = devmem_read32(vmbox_addr + 36);
        uint32_t argmax = devmem_read32(diag_addr + 12 * 4);

        total_wall += wall_ms;
        total_cycles += m33_cycles;

        /* Save output tensor */
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/out_%04d.bin", output_dir, f);
        int ofd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (ofd >= 0) {
            write(ofd, out_buf.virt, out_actual);
            close(ofd);
        }

        fprintf(csv, "%d,%.2f,%u,%u,%u\n", f, wall_ms, m33_cycles, argmax, out_actual);

        if (f % 10 == 0 || f == num_frames - 1) {
            double fps = (f + 1) / (total_wall / 1000.0);
            printf("  frame %d/%d: argmax=%u wall=%.1fms cycles=%u (%.1f FPS avg)\n",
                   f, num_frames, argmax, wall_ms, m33_cycles, fps);
        }
    }

    fclose(csv);

    /* Send stop command */
    devmem_write32(vmbox_addr + 20, 0xFF);  /* command = stop */

    /* Summary */
    double avg_wall = total_wall / num_frames;
    double avg_fps = num_frames / (total_wall / 1000.0);
    printf("\n=== Video inference complete ===\n");
    printf("  Frames:     %d\n", num_frames);
    printf("  Total time: %.1f ms\n", total_wall);
    printf("  Avg frame:  %.1f ms\n", avg_wall);
    printf("  Avg FPS:    %.1f\n", avg_fps);
    printf("  Avg cycles: %lu\n", (unsigned long)(total_cycles / num_frames));
    printf("  Timing:     %s\n", csv_path);
    printf("  Outputs:    %s/out_XXXX.bin\n", output_dir);

    cma_free(&out_buf);
    cma_free(&in_buf);
    /* Keep PTE buffer alive until M33 is done */
    cma_free(&pte_buf);
    return 0;
}
