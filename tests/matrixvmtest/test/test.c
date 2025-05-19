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
  // sv39_test();
  matrix_sv39_test();
  return 0;
}
