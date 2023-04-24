# Add the path to the m5op library
M5OP_PATH = /home/zhouzikai/gem5/util/m5/src/abi/arm64

CC := aarch64-linux-gnu-gcc-10
CFLAGS := -march=armv8.2-a+sve2 -flax-vector-conversions -I /home/zhouzikai/gem5/include/gem5/ -O3 -g
LDFLAGS := -lstdc++ -L /home/zhouzikai/gem5/include/gem5 $(M5OP_PATH)/m5op_arm64.o


SRC_DIR := src/
TARGET_BASE := out/

SRC ?= $(SRC_DIR)os_ws9_is6.cpp
TARGET := $(TARGET_BASE)$(basename $(notdir $(SRC)))

OBJ := $(SRC:.cpp=.o)

.PHONY: all clean

all: $(TARGET)
	@rm -f $(OBJ)

$(TARGET): $(OBJ)
	mkdir -p $(TARGET_BASE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.INTERMEDIATE: $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)