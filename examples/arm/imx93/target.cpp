/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Platform target for NXP i.MX93 Cortex-M33 + Ethos-U65.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "ARMCM33.h"
#include "ethosu_driver.h"

extern "C" {
void trace_init(void);
void trace_write(const char *buf, int len);
extern uint32_t __StackLimit;
extern uint32_t __StackTop;
__attribute__((weak)) void _GLOBAL__sub_I__ZN9__gnu_cxx9__freeresEv(void);
extern volatile uint32_t executorch_diag_words[];
extern volatile uint32_t executorch_fault_words[];
extern volatile uint32_t executorch_persist_words[];
}

namespace {

constexpr uintptr_t kNpuBaseAddr = 0x4A900000;
constexpr int kNpuIrqNum = 178;

constexpr uintptr_t kFastMemAddr = 0x20480000;
constexpr size_t kFastMemSize = 0x4000; // 16KB
constexpr size_t kDiagWordCount = 64;
constexpr size_t kPersistDiagBase = 0;
constexpr size_t kPersistFaultBase = kDiagWordCount;
constexpr size_t kPersistWordCount = 2 * kDiagWordCount;

constexpr uintptr_t kShcsrAddr = 0xE000ED24;
constexpr uintptr_t kCfsrAddr = 0xE000ED28;
constexpr uintptr_t kHfsrAddr = 0xE000ED2C;
constexpr uintptr_t kMmfarAddr = 0xE000ED34;
constexpr uintptr_t kBfarAddr = 0xE000ED38;

enum FaultKind : uint32_t {
  kFaultHard = 1,
  kFaultBus = 2,
  kFaultMemManage = 3,
  kFaultUsage = 4,
  kFaultSbrk = 5,
};

struct ethosu_driver ethosu_drv;

} // namespace

// DDR PTE mailbox: A55 writes PTE location here via devmem after remoteproc
// start, M33 polls for it during boot. See arm_executor_runner.cpp.
struct DdrPteMailbox {
  volatile uint32_t magic;           // 0x50544544 ("PTED") when valid
  volatile uint32_t pte_phys_addr;
  volatile uint32_t pte_size;
  volatile uint32_t status;          // written by M33: 1=running, 0xDEAD=error
};

extern "C" {
__attribute__((section(".data")))
DdrPteMailbox ddr_pte_mailbox = {0, 0, 0, 0};

struct ethosu_driver* executorch_get_ethosu_driver() {
  return &ethosu_drv;
}
}

namespace {

void npu_irq_handler() {
  ethosu_irq_handler(&ethosu_drv);
}

void trace_hex(const char* prefix, int prefix_len, uint32_t val) {
  char buf[32];
  int i = 0;
  for (int j = 0; j < prefix_len; j++) buf[i++] = prefix[j];
  buf[i++] = '0'; buf[i++] = 'x';
  for (int s = 28; s >= 0; s -= 4) {
    int nibble = (val >> s) & 0xF;
    buf[i++] = nibble < 10 ? '0' + nibble : 'A' + nibble - 10;
  }
  buf[i++] = '\n';
  trace_write(buf, i);
}

inline void diag_store(volatile uint32_t* words, size_t slot, uintptr_t value) {
  if (slot < kDiagWordCount) {
    const uint32_t value32 = static_cast<uint32_t>(value);
    words[slot] = value32;
    if (words == executorch_diag_words) {
      executorch_persist_words[kPersistDiagBase + slot] = value32;
    } else if (words == executorch_fault_words) {
      executorch_persist_words[kPersistFaultBase + slot] = value32;
    }
  }
}

inline uint32_t read_reg32(uintptr_t addr) {
  return *reinterpret_cast<volatile uint32_t*>(addr);
}

} // namespace

// Ethos-U fast scratch buffer — overrides weak symbols in EthosUBackend.cpp
extern "C" {
__attribute__((section(".bss.ethosu_scratch"), aligned(16)))
uint8_t ethosu_fast_scratch_buf[kFastMemSize];
unsigned char* ethosu_fast_scratch = ethosu_fast_scratch_buf;
size_t ethosu_fast_scratch_size = kFastMemSize;

__attribute__((section(".data")))
volatile uint32_t executorch_diag_words[kDiagWordCount] = {0};
__attribute__((section(".data")))
volatile uint32_t executorch_fault_words[kDiagWordCount] = {0};
__attribute__((section(".noinit.persist_diag"), aligned(16)))
volatile uint32_t executorch_persist_words[kPersistWordCount];

}


// Baremetal stubs normally provided by CRT files.
extern "C" {
void* __dso_handle = nullptr;
void _fini() {}
void _init() {}

int _write(int fd, const char *buf, int len) {
  (void)fd;
  trace_write(buf, len);
  return len;
}

// Minimal vprintf that bypasses newlib's vfprintf (which bus-faults on M33).
// Newlib's _vfprintf_r accesses uninitialized FILE structures via _REENT
// causing bus lockups that crash the entire SoC.
// Supports: %d, %u, %x, %s, %p, %c, %zu, %ld, %lu, %lx, %%.
static int mini_vprintf(const char *fmt, __builtin_va_list ap) {
  int total = 0;
  const char *seg = fmt;
  for (const char *p = fmt; *p; p++) {
    if (*p != '%') continue;
    if (p > seg) { _write(1, seg, p - seg); total += p - seg; }
    p++;
    int is_long = 0;
    if (*p == 'l') { is_long = 1; p++; }
    else if (*p == 'z') { is_long = (sizeof(size_t) > sizeof(int)); p++; }
    char buf[20];
    int len = 0;
    switch (*p) {
      case 'd': case 'i': {
        long val = is_long ? __builtin_va_arg(ap, long) : __builtin_va_arg(ap, int);
        if (val < 0) { buf[len++] = '-'; val = -val; }
        char tmp[12]; int ti = 0;
        do { tmp[ti++] = '0' + (int)(val % 10); val /= 10; } while (val);
        while (ti--) buf[len++] = tmp[ti];
        _write(1, buf, len); total += len; break;
      }
      case 'u': {
        unsigned long val = is_long ? __builtin_va_arg(ap, unsigned long) : __builtin_va_arg(ap, unsigned int);
        char tmp[12]; int ti = 0;
        do { tmp[ti++] = '0' + (int)(val % 10); val /= 10; } while (val);
        while (ti--) buf[len++] = tmp[ti];
        _write(1, buf, len); total += len; break;
      }
      case 'x': case 'X': case 'p': {
        unsigned long val;
        if (*p == 'p') { val = (unsigned long)__builtin_va_arg(ap, void*); buf[len++] = '0'; buf[len++] = 'x'; }
        else { val = is_long ? __builtin_va_arg(ap, unsigned long) : __builtin_va_arg(ap, unsigned int); }
        char tmp[8]; int ti = 0;
        do { int n = val & 0xF; tmp[ti++] = n < 10 ? '0' + n : 'a' + n - 10; val >>= 4; } while (val);
        while (ti--) buf[len++] = tmp[ti];
        _write(1, buf, len); total += len; break;
      }
      case 's': {
        const char *s = __builtin_va_arg(ap, const char*);
        if (!s) s = "(null)";
        int slen = 0; while (s[slen]) slen++;
        _write(1, s, slen); total += slen; break;
      }
      case 'c': {
        char c = (char)__builtin_va_arg(ap, int);
        _write(1, &c, 1); total++; break;
      }
      case '%': _write(1, "%", 1); total++; break;
      default: buf[0] = '%'; buf[1] = *p; _write(1, buf, 2); total += 2; break;
    }
    seg = p + 1;
  }
  if (*seg) { int slen = 0; while (seg[slen]) slen++; _write(1, seg, slen); total += slen; }
  return total;
}

int printf(const char *fmt, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, fmt);
  int r = mini_vprintf(fmt, ap);
  __builtin_va_end(ap);
  return r;
}

int puts(const char *s) {
  int len = 0; while (s[len]) len++;
  _write(1, s, len);
  _write(1, "\n", 1);
  return 0;
}

int vprintf(const char *fmt, __builtin_va_list ap) {
  return mini_vprintf(fmt, ap);
}

int fprintf(FILE *stream, const char *fmt, ...) {
  (void)stream;
  __builtin_va_list ap;
  __builtin_va_start(ap, fmt);
  int r = mini_vprintf(fmt, ap);
  __builtin_va_end(ap);
  return r;
}

int vfprintf(FILE *stream, const char *fmt, __builtin_va_list ap) {
  (void)stream;
  return mini_vprintf(fmt, ap);
}

void executorch_diag_reset() {
  for (size_t i = 0; i < kDiagWordCount; ++i) {
    diag_store(executorch_diag_words, i, 0);
    diag_store(executorch_fault_words, i, 0);
  }
  diag_store(executorch_diag_words, 0, 0x45544447); // ETDG
}

void executorch_diag_mark(uint32_t slot, uintptr_t value) {
  diag_store(executorch_diag_words, slot, value);
}

uintptr_t executorch_diag_read_sp() {
  uintptr_t sp = 0;
  __asm volatile("mov %0, sp" : "=r"(sp));
  return sp;
}

// Minimal vsnprintf for ET_LOG message formatting.
static int mini_vsnprintf(char *buf, size_t size, const char *fmt, __builtin_va_list ap) {
  if (!size) return 0;
  char *out = buf;
  char *end = buf + size - 1;
  const char *seg = fmt;

  auto emit = [&](const char *s, int len) {
    for (int i = 0; i < len && out < end; i++) *out++ = s[i];
  };

  for (const char *p = fmt; *p; p++) {
    if (*p != '%') continue;
    emit(seg, p - seg);
    p++;
    int is_long = 0;
    if (*p == 'l') { is_long = 1; p++; }
    else if (*p == 'z') { is_long = (sizeof(size_t) > sizeof(int)); p++; }
    char tmp[20];
    int len = 0;
    switch (*p) {
      case 'd': case 'i': {
        long val = is_long ? __builtin_va_arg(ap, long) : __builtin_va_arg(ap, int);
        if (val < 0) { tmp[len++] = '-'; val = -val; }
        char rev[12]; int ri = 0;
        do { rev[ri++] = '0' + (int)(val % 10); val /= 10; } while (val);
        while (ri--) tmp[len++] = rev[ri];
        emit(tmp, len); break;
      }
      case 'u': {
        unsigned long val = is_long ? __builtin_va_arg(ap, unsigned long) : __builtin_va_arg(ap, unsigned int);
        char rev[12]; int ri = 0;
        do { rev[ri++] = '0' + (int)(val % 10); val /= 10; } while (val);
        while (ri--) tmp[len++] = rev[ri];
        emit(tmp, len); break;
      }
      case 'x': case 'X': case 'p': {
        unsigned long val;
        if (*p == 'p') { val = (unsigned long)__builtin_va_arg(ap, void*); tmp[len++] = '0'; tmp[len++] = 'x'; }
        else { val = is_long ? __builtin_va_arg(ap, unsigned long) : __builtin_va_arg(ap, unsigned int); }
        char rev[8]; int ri = 0;
        do { int n = val & 0xF; rev[ri++] = n < 10 ? '0' + n : 'a' + n - 10; val >>= 4; } while (val);
        while (ri--) tmp[len++] = rev[ri];
        emit(tmp, len); break;
      }
      case 's': {
        const char *s = __builtin_va_arg(ap, const char*);
        if (!s) s = "(null)";
        int slen = 0; while (s[slen]) slen++;
        emit(s, slen); break;
      }
      case 'c': { char c = (char)__builtin_va_arg(ap, int); emit(&c, 1); break; }
      case '%': emit("%", 1); break;
      default: { char b[2] = {'%', *p}; emit(b, 2); break; }
    }
    seg = p + 1;
  }
  if (*seg) { int slen = 0; while (seg[slen]) slen++; emit(seg, slen); }
  *out = '\0';
  return out - buf;
}

int snprintf(char *buf, size_t size, const char *fmt, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, fmt);
  int r = mini_vsnprintf(buf, size, fmt, ap);
  __builtin_va_end(ap);
  return r;
}

int vsnprintf(char *buf, size_t size, const char *fmt, __builtin_va_list ap) {
  return mini_vsnprintf(buf, size, fmt, ap);
}

void* _sbrk(ptrdiff_t increment) {
  uintptr_t lr = 0;
  __asm volatile("mov %0, lr" : "=r"(lr));
  diag_store(executorch_diag_words, 1, 0x5342524B); // SBRK
  diag_store(executorch_diag_words, 2, increment);
  diag_store(executorch_diag_words, 3, lr);
  diag_store(executorch_diag_words, 4, executorch_diag_read_sp());
  diag_store(executorch_fault_words, 0, kFaultSbrk);
  diag_store(executorch_fault_words, 1, increment);
  diag_store(executorch_fault_words, 2, lr);
  diag_store(executorch_fault_words, 3, executorch_diag_read_sp());
  while (1) {
    __asm volatile("wfi");
  }
}

void fault_handler_c(uint32_t* stack, uint32_t exc_return, uint32_t fault_kind) {
  const uint32_t cfsr = read_reg32(kCfsrAddr);
  const uint32_t hfsr = read_reg32(kHfsrAddr);
  const uint32_t mmfar = read_reg32(kMmfarAddr);
  const uint32_t bfar = read_reg32(kBfarAddr);

  diag_store(executorch_diag_words, 1, 0xFA170000u | fault_kind);
  diag_store(executorch_diag_words, 24, executorch_diag_read_sp());

  diag_store(executorch_fault_words, 0, fault_kind);
  diag_store(executorch_fault_words, 1, cfsr);
  diag_store(executorch_fault_words, 2, hfsr);
  diag_store(executorch_fault_words, 3, bfar);
  diag_store(executorch_fault_words, 4, mmfar);
  diag_store(executorch_fault_words, 5, exc_return);
  diag_store(
      executorch_fault_words,
      6,
      reinterpret_cast<uintptr_t>(stack));
  if (stack != nullptr) {
    diag_store(executorch_fault_words, 7, stack[0]);
    diag_store(executorch_fault_words, 8, stack[1]);
    diag_store(executorch_fault_words, 9, stack[2]);
    diag_store(executorch_fault_words, 10, stack[3]);
    diag_store(executorch_fault_words, 11, stack[4]);
    diag_store(executorch_fault_words, 12, stack[5]);
    diag_store(executorch_fault_words, 13, stack[6]);
    diag_store(executorch_fault_words, 14, stack[7]);
  }
  while (1) {
    __asm volatile("wfi");
  }
}

__attribute__((naked)) void HardFault_Handler() {
  __asm volatile(
      "tst lr, #4\n"
      "ite eq\n"
      "mrseq r0, msp\n"
      "mrsne r0, psp\n"
      "mov r1, lr\n"
      "movs r2, #1\n"
      "b fault_handler_c\n");
}

__attribute__((naked)) void BusFault_Handler() {
  __asm volatile(
      "tst lr, #4\n"
      "ite eq\n"
      "mrseq r0, msp\n"
      "mrsne r0, psp\n"
      "mov r1, lr\n"
      "movs r2, #2\n"
      "b fault_handler_c\n");
}

__attribute__((naked)) void MemManage_Handler() {
  __asm volatile(
      "tst lr, #4\n"
      "ite eq\n"
      "mrseq r0, msp\n"
      "mrsne r0, psp\n"
      "mov r1, lr\n"
      "movs r2, #3\n"
      "b fault_handler_c\n");
}

__attribute__((naked)) void UsageFault_Handler() {
  __asm volatile(
      "tst lr, #4\n"
      "ite eq\n"
      "mrseq r0, msp\n"
      "mrsne r0, psp\n"
      "mov r1, lr\n"
      "movs r2, #4\n"
      "b fault_handler_c\n");
}

}

extern "C" void target_init();
extern "C" void npu_init();

// Called from CMSIS startup before main — keep minimal,
// .bss zero table hasn't been processed yet by __cmsis_start.
extern "C" void SystemInit() {
  target_init();
}

void target_init() {
  trace_init();
  executorch_diag_reset();
  trace_write("target_init: start\n", 19);

  // Enable FPU (CP10/CP11 full access)
  *reinterpret_cast<volatile uint32_t*>(0xE000ED88) |= (0xFu << 20);
  __asm volatile("dsb"); __asm volatile("isb");

  // Enable BusFault, MemManage, UsageFault as separate exceptions
  *reinterpret_cast<volatile uint32_t*>(kShcsrAddr) |= (7u << 16);

  diag_store(
      executorch_diag_words,
      28,
      reinterpret_cast<uintptr_t>(&__StackLimit));
  diag_store(
      executorch_diag_words,
      29,
      reinterpret_cast<uintptr_t>(&__StackTop));
  diag_store(executorch_diag_words, 30, kFastMemAddr);

  trace_write("target_init: done\n", 18);
}

// Walk .init_array manually with tracing to identify crashing constructors.
extern "C" void run_init_array() {
  typedef void (*ctor_t)();
  extern ctor_t __init_array_start[];
  extern ctor_t __init_array_end[];
  int count = __init_array_end - __init_array_start;
  char buf[40];
  int len = 0;
  // "init_array: N entries\n"
  const char *prefix = "init_array: ";
  for (int i = 0; prefix[i]; i++) buf[len++] = prefix[i];
  if (count >= 10) buf[len++] = '0' + count / 10;
  buf[len++] = '0' + count % 10;
  buf[len++] = '\n';
  trace_write(buf, len);

  for (int i = 0; i < count; i++) {
    len = 0;
    buf[len++] = 'c'; buf[len++] = 't'; buf[len++] = 'o'; buf[len++] = 'r';
    buf[len++] = '[';
    if (i >= 10) buf[len++] = '0' + i / 10;
    buf[len++] = '0' + i % 10;
    buf[len++] = ']'; buf[len++] = '\n';
    trace_write(buf, len);
    // Skip only libstdc++'s optional freeres ctor. Skipping by position is
    // brittle and can drop real static initializers such as kernel
    // registrations.
    if (
        _GLOBAL__sub_I__ZN9__gnu_cxx9__freeresEv != nullptr &&
        __init_array_start[i] == _GLOBAL__sub_I__ZN9__gnu_cxx9__freeresEv) {
      trace_write("ctor:skip freeres\n", 18);
      continue;
    }
    __init_array_start[i]();
  }
  trace_write("init_array: done\n", 17);
}

// Override weak ethosu semaphore/mutex to use NOP instead of WFE/WFI.
// WFE hangs on bare metal M33 without configured event sources.
struct ethosu_semaphore_t {
  volatile int count;
  volatile int allocated;
};
constexpr int kSemaphorePoolSize = 4;
__attribute__((section(".bss"))) ethosu_semaphore_t
    g_semaphore_pool[kSemaphorePoolSize];

extern "C" void* ethosu_semaphore_create(void) {
  trace_write("sem_create: enter\n", 18);
  for (int i = 0; i < kSemaphorePoolSize; ++i) {
    if (!g_semaphore_pool[i].allocated) {
      g_semaphore_pool[i].allocated = 1;
      g_semaphore_pool[i].count = 0;
      trace_hex("sem_create: ptr=", 16, reinterpret_cast<uintptr_t>(&g_semaphore_pool[i]));
      return &g_semaphore_pool[i];
    }
  }
  trace_write("sem_create: fail\n", 17);
  return nullptr;
}

extern "C" int ethosu_semaphore_take(void *sem, uint64_t timeout) {
  (void)timeout;
  auto *s = static_cast<ethosu_semaphore_t*>(sem);
  while (s->count == 0) {
    // Poll NPU STATUS register for cmd_end_reached (bit 5) when a job is
    // running. Without IRQs, nothing else will signal the semaphore.
    if (ethosu_drv.job.state == ETHOSU_JOB_RUNNING) {
      volatile uint32_t *status_reg =
          reinterpret_cast<volatile uint32_t*>(kNpuBaseAddr + 0x0004);
      if (*status_reg & (1u << 5)) {
        ethosu_irq_handler(&ethosu_drv);
        continue;
      }
    }
    __asm volatile("nop");
  }
  s->count--;
  return 0;
}

extern "C" int ethosu_semaphore_give(void *sem) {
  auto* s = static_cast<ethosu_semaphore_t*>(sem);
  s->count++;
  return 0;
}

extern "C" void ethosu_semaphore_destroy(void* sem) {
  auto* s = static_cast<ethosu_semaphore_t*>(sem);
  s->count = 0;
  s->allocated = 0;
}

// Called from main() after __libc_init_array.
void npu_init() {
  trace_write("npu_init: start\n", 16);

  int err = ethosu_init(
      &ethosu_drv,
      reinterpret_cast<void*>(kNpuBaseAddr),
      nullptr, 0,
      /*secure=*/0,
      /*privilege=*/0);
  if (err) {
    trace_write("npu_init: ethosu_init FAIL\n", 27);
  } else {
    trace_write("npu_init: ethosu_init ok\n", 25);
  }

  // NPU IRQ 178 (GIC SPI number) is WRONG for M33 NVIC — it points to
  // a different peripheral and causes SoC reboot. Use polling mode instead.
  // TODO: find correct M33 NVIC IRQ number from GPC interrupt map.
  trace_write("npu_init: polling mode\n", 23);
}
