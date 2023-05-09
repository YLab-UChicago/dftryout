# Add the path to the m5op library
M5OP_PATH = /home/zhouzikai/gem5/util/m5/src/abi

CC := arm-linux-gnueabihf-g++
CFLAGS := -march=armv7-a -flax-vector-conversions -I /home/zhouzikai/gem5/include/gem5/ -O3
LDFLAGS := -lstdc++ -L /home/zhouzikai/gem5/include/gem5 $(M5OP_PATH)/m5op_arm.o

SRC_DIR := src/
TARGET_DIR := out/

SRCS := $(wildcard $(SRC_DIR)*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SRCS))
TARGET := $(addprefix $(TARGET_DIR), $(notdir $(basename $(SRCS))))

.PHONY: all clean

all: $(TARGET)

$(TARGET_DIR)%: $(SRC_DIR)%.o
	mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(SRC_DIR)%.o: $(SRC_DIR)%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build m5op_arm.o from m5op.S
$(M5OP_PATH)/m5op_arm.o: $(M5OP_PATH)/m5op.S
	$(CC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) $(M5OP_PATH)/m5op_arm.o