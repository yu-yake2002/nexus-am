#include <riscv_matrix.h>
#include <klib.h>
#include "../isa/utils.h"

const fp16_t ls_fp16_src[8 * 8] = {
    -104, -10,  -90,  3,   94,   -2,  -118, 29,   -85, 38,  -33, 9,    71,
    78,   -99,  60,   -24, -74,  105, 113,  -109, 6,   -90, -11, -121, 11,
    36,   43,   -119, 2,   -112, 93,  75,   -81,  -33, 93,  76,  -37,  -22,
    -48,  -111, 3,    -20, -95,  1,   35,   118,  61,  -17, 44,  -67,  32,
    26,   57,   99,   125, -111, -20, 47,   18,   58,  -91, 101, -55};

fp16_t fp16_buffer[200];

int main() {
	printf("hello riscv matrix clang ==================== \n"
         "AME info:\n");
  size_t cfg;
  asm volatile("csrr %0, mlenb" : "=r"(cfg));
  printf("mlenb:  %lu\n", cfg);
  asm volatile("csrr %0, mrlenb" : "=r"(cfg));
  printf("mrlenb: %lu\n", cfg);
  asm volatile("csrr %0, mamul" : "=r"(cfg));
  printf("mamul:  %lu\n", cfg);

  const size_t M = 8;
  const size_t K = 8;
  const size_t stride = K * sizeof(fp16_t);
  SET_MBA0_F16();
  msettilem(M);
  msettilek(K);
  mfloat16_t ms = mla_m(ls_fp16_src, stride);

  printf("test\n");

  msa_m(ms, fp16_buffer, stride);

  int i = 0;
  int j = 0;
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++)
      printf("%f  ", fp16_buffer[i * 10 + j]);
    printf("\n");
  }

  return 0;
}
