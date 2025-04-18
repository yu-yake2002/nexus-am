#include "../isa/amtest.h"

void sv39_test();
void matrix_sv39_test();

_Context *simple_trap(_Event ev, _Context *ctx) {
  switch(ev.event) {
    case _EVENT_IRQ_TIMER:
      printf("t"); break;
    case _EVENT_IRQ_IODEV:
      printf("d"); read_key(); break;
    case _EVENT_YIELD:
      printf("y"); break;
  }
  return ctx;
}

int main() {
  IOE;
  CTE(simple_trap);
  asm volatile (
    "lui a0, 0x2002\n"
    "addiw a0, a0, 512\n"
    "csrs mstatus, a0\n"
    "csrs sstatus, a0\n"::
  );
  // sv39_test();
  matrix_sv39_test();
  return 0;
}
