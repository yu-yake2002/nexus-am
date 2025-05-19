#include "../isa/amtest.h"
#include <csr.h>
#include <xsextra.h>
#include "../isa/utils.h"
#include <riscv_matrix.h>

const fp16_t srca[8 * 8] = {
  -0.4791228,  -5.50058298, -2.07270458, 5.3226858,   -8.84389546,
  4.59407695,  6.5989989,   -3.14833659, 5.81934332,  8.27158019,
  -7.44617388, 3.45992688,  4.65601025,  -1.30384156, -6.16177024,
  -2.78090041, -5.02383668, 2.47596348,  -2.01387443, 6.29004664,
  9.39431268,  0.01367022,  -1.1863497,  -2.91107541, 7.97764859,
  5.20841543,  -6.92063255, -1.59031739, -1.63604836, 5.33806209,
  -1.77626059, -6.37831689, -8.63225568, -6.51750765, 4.07559851,
  8.82386443,  -0.96819769, -2.65136036, 7.01084136,  -8.91837203,
  3.34552436,  -7.10714317, -4.92892601, -1.69535171, -0.44200629,
  4.95106719,  3.1458187,   1.2355903,   5.10646997,  3.48436152,
  0.71208435,  8.18292857,  -8.59622791, -1.60706513, -5.45765094,
  -1.78489762, 7.15678721,  -3.33436914, 6.6050622,   -0.54018179,
  6.62988256,  1.88233015,  -2.58375522, -5.98121322
};

const fp16_t srcb[8 * 8] = {
  -7.09247889, 0.09991281,  9.71906561,  -3.01612009, 9.19340953,
  9.54288417,  -9.91204777, 6.88476914,  -8.07182731, -4.45583986,
  0.07357176,  7.51683337,  3.30242458,  -4.26903074, -1.62946192,
  2.74305551,  5.8407685,   8.64020195,  -1.70596265, 1.11948438,
  -0.60827231, 2.710995,    -5.54557408, 6.26212106,  9.22451442,
  0.08672065,  -0.7108666,  8.5726977,   8.94694825,  -0.97666864,
  0.48978224,  -9.80966056, -3.21427914, -5.39927878, -3.55575748,
  -4.52220854, 9.98265771,  5.73339337,  -4.21182337, -0.78817531,
  6.93726156,  -0.72896213, 2.75900982,  5.41802047,  -0.91040382,
  -3.06361928, -3.51136412, -2.40916743, 3.64698973,  2.47729373,
  -9.05774253, -8.48844017, -4.126127,   9.52703889,  1.90001698,
  -9.51317452, 9.4533237,   -4.48583695, 1.73332576,  1.42924847,
  0.62548777,  7.19749661,  1.22851512,  -9.70041731
};

const fp16_t srcc[8 * 8] = {
  4.4160154,   3.5332977,   -4.19579848, 3.07497555,  1.20030683,
  3.41173536,  2.01856532,  -2.06981731, 2.41305312,  -0.77042081,
  -8.49611319, -6.30986548, -9.72178782, -4.91703216, 2.55078244,
  6.00616738,  -7.68156249, 9.33716417,  -6.00531342, -1.50450808,
  -8.36759781, -1.03462902, 3.64356792,  6.91081309,  -1.49179717,
  0.04204007,  9.76100764,  4.71724232,  -0.26571138, 2.65151656,
  -8.72104956, -3.52638119, 6.97061847,  8.35415848,  0.14319908,
  6.08000407,  9.08252654,  -3.40035253, 1.62632946,  -8.3082222,
  8.78537245,  6.96326183,  -3.82118672, -4.85217052, 4.37744263,
  4.15808431,  7.11142754,  -7.14811029, 4.68388129,  5.54947351,
  4.32283561,  6.07252715,  -7.86587985, -9.6808525,  -2.00948736,
  -9.59864288, 3.049945,    4.92879451,  6.27233823,  3.5829288,
  -3.26062172, 1.89711464,  -5.06760085, 3.4590756
};

const fp16_t answ[8 * 8] = {
  143.80824244, 85.42047692,  -30.61214998,  10.85229756,   -94.1527624,
  -13.06716935, 59.61981715,  -121.98474951, -189.97398697, -128.06072475,
  89.75293632,  79.84894886,  177.93038264,  -56.29081881,  -55.77700838,
  73.25616628,  -7.72265771,  -59.66488837,  -83.35231337,  47.42819173,
  107.96655256, -49.58482491, 18.20886036,   -63.11248645,  -179.69029879,
  -53.14832541, 126.19470112, 40.70740029,   62.40523473,   -49.22649626,
  -81.74674374, 105.14691544, 151.98248402,  137.04644802,  -180.29246723,
  -18.91261389, -57.09065993, -50.3482864,   95.28206144,   -119.68716505,
  56.91821513,  -2.74028875,  26.66440639,   -84.53077827,  -21.63374975,
  75.87926478,  4.00632367,   -71.33697747,  -0.31131816,   39.46757565,
  119.64992428, 161.75181043, 60.41833493,   -91.09096793,  -28.95737817,
  39.18428904,  -61.41659078, 60.78478557,   59.35502654,   -46.70304584,
  114.06239061, 67.44158298,  -154.25685373, 163.0858631
};

/*
 * RISC-V 64 SV39 Virutal Memory test
 */

#define EXCEPTION_LOAD_ACCESS_FAULT 5
#define EXCEPTION_STORE_ACCESS_FAULT 7
#define EXCEPTION_LOAD_PAGE_FAULT 13
#define EXCEPTION_STORE_PAGE_FAULT 15

extern volatile uint64_t load_page_fault_to_be_reported;
extern volatile uint64_t store_page_fault_to_be_reported;
extern volatile uint64_t load_access_fault_to_be_reported;
extern volatile uint64_t store_access_fault_to_be_reported;

inline int inst_is_compressed(uint64_t addr){
  uint8_t byte = *(uint8_t*)addr;
  return (byte & 0x3) != 0x3;
}

extern _Context* store_page_fault_handler(_Event* ev, _Context *c);

extern _Context* load_page_fault_handler(_Event* ev, _Context *c);

extern _Context* load_access_fault_handler(_Event* ev, _Context *c);

extern _Context* store_access_fault_handler(_Event* ev, _Context *c);

extern void* sv39_pgalloc(size_t pg_size);

extern void sv39_pgfree(void *ptr);

extern _AddressSpace kas;
#include <riscv.h>

void matrix_sv39_test() {
  asm volatile (
    "lui a0, 0x2002\n"
    "addiw a0, a0, 512\n"
    "csrs mstatus, a0\n"
    "csrs sstatus, a0\n"::
  );
  printf("start sv39 test\n");
  _vme_init(sv39_pgalloc, sv39_pgfree);
  printf("sv39 setup done\n");
  _map(&kas, (void *)0x900000000UL, (void *)0x80020000, PTE_R | PTE_A | PTE_D);
  _map(&kas, (void *)0x900010000UL, (void *)0x80030000, PTE_R | PTE_A | PTE_D);
  _map(&kas, (void *)0x900020000UL, (void *)0x80040000, PTE_R | PTE_A | PTE_D);
  _map(&kas, (void *)0xa00000000UL, (void *)0x80020000, PTE_W | PTE_R | PTE_A | PTE_D);
  _map(&kas, (void *)0xa00010000UL, (void *)0x80030000, PTE_W | PTE_R | PTE_A | PTE_D);
  _map(&kas, (void *)0xa00020000UL, (void *)0x80040000, PTE_W | PTE_R | PTE_A | PTE_D);
  _map(&kas, (void *)0xb00000000UL, (void *)0x80020000, PTE_A | PTE_D);
  printf("memory map done\n");
  fp16_t *w_ptr_0 = (fp16_t *)(0xa00000000UL);
  fp16_t *w_ptr_1 = (fp16_t *)(0xa00010000UL);
  fp16_t *w_ptr_2 = (fp16_t *)(0xa00020000UL);
  fp16_t *r_ptr_0 = (fp16_t *)(0x900000000UL);
  fp16_t *r_ptr_1 = (fp16_t *)(0x900010000UL);
  fp16_t *r_ptr_2 = (fp16_t *)(0x900020000UL);
  // fp16_t *fault_ptr = (fp16_t *)(0xb00000000UL);

  irq_handler_reg(EXCEPTION_STORE_PAGE_FAULT, &store_page_fault_handler);
  irq_handler_reg(EXCEPTION_LOAD_PAGE_FAULT, &load_page_fault_handler);
  asm volatile("sfence.vma");
  printf("test sv39 data write\n");
  for (int i = 0; i < 8 * 8; ++i) {
    w_ptr_0[i] = srca[i];
    w_ptr_1[i] = srcb[i];
    w_ptr_2[i] = srcc[i];
  }

  printf("test sv39 data read\n");
  for (int i = 0; i < 8 * 8; ++i) {
    assert(r_ptr_0[i] == srca[i]);
    assert(r_ptr_1[i] == srcb[i]);
    assert(r_ptr_2[i] == srcc[i]);
  }

  SET_MBA0_F16();
  msettilem(8);
  msettilek(8);
  msettilen(8);
  mfloat16_t ts1 = mla_m(r_ptr_0, 8 * sizeof(fp16_t));
  mfloat16_t ts2 = mlb_m(r_ptr_1, 8 * sizeof(fp16_t));
  mfloat16_t td  = mlc_m(r_ptr_2, 8 * sizeof(fp16_t));
  mfloat16_t md  = mfma_mm(td, ts1, ts2);
  msc_m(md, w_ptr_0, 8 * sizeof(fp16_t));

  // printf("test sv39 store page fault\n");
  // store_page_fault_to_be_reported = 1;
  // *fault_ptr = 'b';
  // if(store_page_fault_to_be_reported){
  //   printf("error @ access -1!\n");
  //   _halt(1);
  // }
  // for (int i = 0; i < 8 * 8; ++i) {
  //   store_page_fault_to_be_reported = 1;
  //   fault_ptr[i] = srcb[i];
  //   if(store_page_fault_to_be_reported){
  //     printf("error @ access %d!\n", i);
  //     _halt(1);
  //   }
  // }

  // printf("test sv39 load page fault\n");
  // for (int i = 0; i < 8 * 8; ++i) {
  //   load_page_fault_to_be_reported = 1;
  //   w_ptr[i] = fault_ptr[i];
  //   if(load_page_fault_to_be_reported){
  //     _halt(1);
  //   }
  // }
  
  _halt(0);
}
