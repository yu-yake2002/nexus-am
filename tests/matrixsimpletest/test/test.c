#include <riscv_matrix.h>
#include <klib.h>
#include "../isa/utils.h"

const int8_t ls_i8_src[8 * 8] = {
    -104, -10,  -90,  3,   94,   -2,  -118, 29,   -85, 38,  -33, 9,    71,
    78,   -99,  60,   -24, -74,  105, 113,  -109, 6,   -90, -11, -121, 11,
    36,   43,   -119, 2,   -112, 93,  75,   -81,  -33, 93,  76,  -37,  -22,
    -48,  -111, 3,    -20, -95,  1,   35,   118,  61,  -17, 44,  -67,  32,
    26,   57,   99,   125, -111, -20, 47,   18,   58,  -91, 101, -55};

int8_t i8_buffer[200];

int main() {
	printf("hello riscv matrix clang ==================== \n");

  const size_t M = 8;
  const size_t K = 8;
  const size_t stride = K * sizeof(int8_t);
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  mint8_t ms = mla_m(ls_i8_src, stride);

  printf("test\n");

  msa_m(ms, i8_buffer, stride);

  int i = 0;
  int j = 0;
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++)
      printf("%d  ", i8_buffer[i * 10 + j]);
    printf("\n");
  }

  return 0;
}
