#include <am.h>
#include "x86_64-qemu.h"
#include <klib.h>

// Multiprocesor configuration
struct mpconf {           // configuration table header
  uint8_t  signature[4];  // "PCMP"
  uint16_t length;        // total table length
  uint8_t  version;       // [14]
  uint8_t  checksum;      // all bytes must add up to 0
  uint8_t  product[20];   // product id
  uint32_t oemtable;      // OEM table pointer
  uint16_t oemlength;     // OEM table length
  uint16_t entry;         // entry count
  uint32_t lapicaddr;     // address of local APIC
  uint16_t xlength;       // extended table length
  uint8_t  xchecksum;     // extended table checksum
  uint8_t  reserved;
};


struct mpdesc {
  int      magic;
  uint32_t conf;     // MP config table addr
  uint8_t  length;   // 1
  uint8_t  specrev;  // [14]
  uint8_t  checksum; // all bytes add to 0
  uint8_t  type;     // config type
  uint8_t  imcrp;
  uint8_t  reserved[3];
};

static inline void *upcast(uint32_t ptr) {
  return (void *)(uintptr_t)ptr;
}

void bootcpu_init() {

  int32_t magic = 0x5a5aa5a5;
  int32_t step = 1L << 20;
  extern char end;
  uintptr_t st, ed;
  for (st = ed = ROUNDUP(&end, step); ; ed += step) {
    volatile uint32_t *ptr = (uint32_t *)ed;
    if ((*ptr = magic, *ptr) != magic) {
      break; // read-after-write fail
    }
  }
  _heap = RANGE(st, ed);

  for (char *st = (char *)0xf0000; st != (char *)0xffffff; st ++) {
    if (*(volatile uint32_t *)st == 0x5f504d5f) {
      uint32_t mpconf_ptr = ((volatile struct mpdesc *)st)->conf;
      struct mpconf *conf = upcast(mpconf_ptr);
      __am_lapic = upcast(conf->lapicaddr);
      for (volatile char *ptr = (char *)(conf + 1);
                 ptr < (char *)conf + conf->length; ptr += 8) {
        if (*ptr == '\0') {
          ptr += 12;
          if (++__am_ncpu > MAX_CPU) {
            panic("cannot support > MAX_CPU processors");
          }
        }
      }
      printf("There are %d cpus, lapic at %p\n", __am_ncpu, __am_lapic);
      return;
    }
  }
  panic("seems not an x86-qemu virtual machine");

  __am_ioapic_init();
}

// apic.c


// LAPIC/IOAPIC related code
// from xv6

#define ID      (0x0020/4)   // ID
#define VER     (0x0030/4)   // Version
#define TPR     (0x0080/4)   // Task Priority
#define EOI     (0x00B0/4)   // EOI
#define SVR     (0x00F0/4)   // Spurious Interrupt Vector
  #define ENABLE     0x00000100   // Unit Enable
#define ESR     (0x0280/4)   // Error Status
#define ICRLO   (0x0300/4)   // Interrupt Command
  #define INIT       0x00000500   // INIT/RESET
  #define STARTUP    0x00000600   // Startup IPI
  #define DELIVS     0x00001000   // Delivery status
  #define ASSERT     0x00004000   // Assert interrupt (vs deassert)
  #define DEASSERT   0x00000000
  #define LEVEL      0x00008000   // Level triggered
  #define BCAST      0x00080000   // Send to all APICs, including self.
  #define BUSY       0x00001000
  #define FIXED      0x00000000
#define ICRHI   (0x0310/4)   // Interrupt Command [63:32]
#define TIMER   (0x0320/4)   // Local Vector Table 0 (TIMER)
  #define X1         0x0000000B   // divide counts by 1
  #define PERIODIC   0x00020000   // Periodic
#define PCINT   (0x0340/4)   // Performance Counter LVT
#define LINT0   (0x0350/4)   // Local Vector Table 1 (LINT0)
#define LINT1   (0x0360/4)   // Local Vector Table 2 (LINT1)
#define ERROR   (0x0370/4)   // Local Vector Table 3 (ERROR)
  #define MASKED     0x00010000   // Interrupt masked
#define TICR    (0x0380/4)   // Timer Initial Count
#define TCCR    (0x0390/4)   // Timer Current Count
#define TDCR    (0x03E0/4)   // Timer Divide Configuration

#define IOAPIC_ADDR  0xFEC00000   // Default physical address of IO APIC
#define REG_ID     0x00  // Register index: ID
#define REG_VER    0x01  // Register index: version
#define REG_TABLE  0x10  // Redirection table base

#define INT_DISABLED   0x00010000  // Interrupt disabled
#define INT_LEVEL      0x00008000  // Level-triggered (vs edge-)
#define INT_ACTIVELOW  0x00002000  // Active low (vs high)
#define INT_LOGICAL    0x00000800  // Destination is CPU id (vs APIC ID)

volatile unsigned int *__am_lapic = NULL;  // Initialized in mp.c
struct IOAPIC {
    uint32_t reg, pad[3], data;
} __attribute__((packed));
typedef struct IOAPIC IOAPIC;

static volatile IOAPIC *ioapic;

static void lapicw(int index, int value) {
  __am_lapic[index] = value;
  __am_lapic[ID];
}

void __am_percpu_initlapic(void) {
  lapicw(SVR, ENABLE | (T_IRQ0 + IRQ_SPURIOUS));
  lapicw(TDCR, X1);
  lapicw(TIMER, PERIODIC | (T_IRQ0 + IRQ_TIMER));
  lapicw(TICR, 10000000); 
  lapicw(LINT0, MASKED);
  lapicw(LINT1, MASKED);
  if(((__am_lapic[VER]>>16) & 0xFF) >= 4)
    lapicw(PCINT, MASKED);
  lapicw(ERROR, T_IRQ0 + IRQ_ERROR);
  lapicw(ESR, 0);
  lapicw(ESR, 0);
  lapicw(EOI, 0);
  lapicw(ICRHI, 0);
  lapicw(ICRLO, BCAST | INIT | LEVEL);
  while(__am_lapic[ICRLO] & DELIVS) ;
  lapicw(TPR, 0);
}

// Acknowledge interrupt.
void __am_lapic_eoi(void) {
  if(__am_lapic)
    lapicw(EOI, 0);
}

void __am_lapic_bootap(uint32_t apicid, uint32_t addr) {
  int i;
  uint16_t *wrv;
  outb(0x70, 0xF);
  outb(0x71, 0x0A);
  wrv = (unsigned short*)((0x40<<4 | 0x67));
  wrv[0] = 0;
  wrv[1] = addr >> 4;

  lapicw(ICRHI, apicid<<24);
  lapicw(ICRLO, INIT | LEVEL | ASSERT);
  lapicw(ICRLO, INIT | LEVEL);
  
 for(i = 0; i < 2; i++){
    lapicw(ICRHI, apicid<<24);
    lapicw(ICRLO, STARTUP | (addr>>12));
  }
}

static unsigned int ioapicread(int reg) {
  ioapic->reg = reg;
  return ioapic->data;
}

static void ioapicwrite(int reg, unsigned int data) {
  ioapic->reg = reg;
  ioapic->data = data;
}

void __am_ioapic_init(void) {
  int i, maxintr;

  ioapic = (volatile IOAPIC*)IOAPIC_ADDR;
  maxintr = (ioapicread(REG_VER) >> 16) & 0xFF;

  for(i = 0; i <= maxintr; i++){
    ioapicwrite(REG_TABLE+2*i, INT_DISABLED | (T_IRQ0 + i));
    ioapicwrite(REG_TABLE+2*i+1, 0);
  }
}

void __am_ioapic_enable(int irq, int cpunum) {
  ioapicwrite(REG_TABLE+2*irq, T_IRQ0 + irq);
  ioapicwrite(REG_TABLE+2*irq+1, cpunum << 24);
}
