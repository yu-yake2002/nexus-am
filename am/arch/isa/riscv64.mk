MARCH ?= rv64gc

ifeq ($(TOOLCHAIN), LLVM)
CROSS_COMPILE := 
else ifeq ($(LINUX_GNU_TOOLCHAIN),1)
CROSS_COMPILE := riscv64-linux-gnu-
else
CROSS_COMPILE := riscv64-unknown-linux-gnu-
endif

ifeq ($(TOOLCHAIN), LLVM)
COMMON_FLAGS  := -fno-pic -march=rv64gv0p10zfh0p1 -mcmodel=medany -menable-experimental-extensions
else
COMMON_FLAGS  := -fno-pic -march=$(MARCH) -mcmodel=medany
endif
CFLAGS        += $(COMMON_FLAGS) -static
ASFLAGS       += $(COMMON_FLAGS) -O0
LDFLAGS       += -melf64lriscv
