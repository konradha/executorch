/*
 * Linker script for ExecuTorch on NXP i.MX93 Cortex-M33 + Ethos-U65.
 *
 * Memory layout:
 *   ITCM:  .text (vectors FIRST!), .resource_table (code — 128KB, PT_LOAD)
 *   DTCM:  .rodata, .data, model (loaded by remoteproc, PT_LOAD)
 *          .bss, heap, stack (zeroed/reserved, not loaded)
 *   DDR:   tensor_arena, ethosu_scratch, method_pool (NOLOAD, no segment)
 *
 * ITCM and DTCM are in the remoteproc ATT. DDR (0xA8000000) is NOT in
 * the ATT — the M33 accesses it directly on the system bus at runtime.
 * Using PHDRS to prevent .ddr from creating a PT_LOAD segment, since
 * remoteproc can't ioremap the 128MB DDR carveout.
 *
 * All PROGBITS content (model data, GOT) goes in DTCM.
 * DDR only holds zero-filled runtime buffers (NOLOAD).
 *
 * Boot: loaded by Linux remoteproc from A55, entry at Reset_Handler.
 */

#ifndef ETHOSU_MODEL
  #define ETHOSU_MODEL 1
#endif

__STACK_SIZE = 0x00010000;
__HEAP_SIZE  = 0x00001000;

MEMORY
{
  ITCM   (rx)  : ORIGIN = 0x0FFE0000, LENGTH = 0x00020000  /* 128KB M33 code TCM */
  DTCM   (rw)  : ORIGIN = 0x20000000, LENGTH = 0x00020000  /* 128KB M33 data TCM */
  DDR    (rw)  : ORIGIN = 0xA8000000, LENGTH = 0x08000000  /* 128MB reserved region */
}

PHDRS
{
  rom_code PT_LOAD;
  rom_data PT_LOAD;
  rom_stack PT_LOAD;
  /* No PT_LOAD for DDR — remoteproc can't load there (not in ATT) */
}

ENTRY(Reset_Handler)

SECTIONS
{
  /* Vector table MUST be at the very start of ITCM (0x0FFE0000).
   * The M33 reads initial SP and reset vector from VTOR base.
   * If anything else is here, the M33 faults immediately on boot. */
  .text :
  {
    _vectors = .;
    KEEP(*(.vectors))
    . = ALIGN(16);

    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    KEEP(*(.eh_frame*))

    *(.ARM.extab* .gnu.linkonce.armextab.*)
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;

    /* Copy/zero tables for CMSIS startup.
     * Copy table is empty — remoteproc loads .data/.rodata to DTCM directly.
     * Zero table zeroes .bss. */
    . = ALIGN(4);
    __copy_table_start__ = .;
    __copy_table_end__ = .;
    __zero_table_start__ = .;
    LONG (ADDR(.bss))
    LONG (SIZEOF(.bss) / 4)
    __zero_table_end__ = .;

#if (ETHOSU_MODEL == 0)
    . = ALIGN(16);
    *(network_model_sec)
#endif
  } > ITCM :rom_code

  /* Resource table — remoteproc finds it by ELF section name, not address.
   * Must NOT be at ITCM base (that's the vector table).
   * Do NOT read /sys/kernel/debug/remoteproc/remoteproc0/resource_table
   * as rproc_rsc_table_show crashes with ITCM addresses. */
  .resource_table :
  {
    KEEP(*(.resource_table))
  } > ITCM :rom_code

  /*
   * .rodata and .data live directly in DTCM (loaded by remoteproc).
   * No LMA in ITCM, no startup copy needed.
   */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata.*)
    . = ALIGN(4);
  } > DTCM :rom_data

  .data :
  {
    . = ALIGN(4);
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)
    *(.tm_clone_table)

    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    __init_array_start = .;
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    __init_array_end = .;

    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))

    /* Model data and GOT in DTCM. */
#if (ETHOSU_MODEL == 1)
    . = ALIGN(16);
    *(network_model_sec)
#endif
    *(.got*)

    . = ALIGN(4);
    __data_end__ = .;
  } > DTCM :rom_data

  .persist (NOLOAD) :
  {
    . = ALIGN(16);
    *(.noinit.persist_diag)
    . = ALIGN(16);
  } > DTCM :rom_data

  /* BSS in DTCM.
   * EXCLUDE_FILE skips .bss.* from runner and target objects —
   * their .bss.tensor_arena and .bss.ethosu_scratch go to .ddr below. */
  .bss (NOLOAD) :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(EXCLUDE_FILE(*arm_executor_runner.cpp* *target.cpp*) .bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > DTCM :rom_data

  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM :rom_data

  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM :rom_stack
  PROVIDE(__stack = __StackTop);

  /* DDR: M33 accesses DDR directly on the system bus at runtime.
   * NOLOAD with :NONE — no PT_LOAD segment, remoteproc can't load here. */
  .ddr (NOLOAD) :
  {
    . = ALIGN(16);
    *(.bss.ddr_pte)

    . = ALIGN(32);
    *(.bss.tensor_arena)

    . = ALIGN(16);
    *(.bss.ethosu_scratch)

    /* Catch any remaining .bss.* from excluded files */
    *arm_executor_runner.cpp*(.bss.*)
    *target.cpp*(.bss.*)

    . = ALIGN(4);
    *(input_data_sec)
    *(expected_output_data_sec)
    *(output_data_sec)

    . = ALIGN(16);
    *(sec_command_stream, sec_weight_data, sec_input_data)
    *(ethosu_core_in_queue)
    *(ethosu_core_out_queue)
    . = ALIGN(4);
  } > DDR :NONE

  /DISCARD/ :
  {
    *(.ARM.attributes)
    *(.comment)
  }

  ASSERT(__StackLimit >= __HeapLimit, "region DTCM overflowed with stack")
}
