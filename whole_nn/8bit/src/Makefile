# Add the path to the m5op library
M5OP_PATH = /home/zhouzikai/gem5/util/m5/src/abi/arm64

CC := aarch64-linux-gnu-gcc-10
CFLAGS := -march=armv8.2-a+sve2 -flax-vector-conversions -I /home/zhouzikai/gem5/include/gem5/ -O3
LDFLAGS := -lstdc++ -L /home/zhouzikai/gem5/include/gem5 $(M5OP_PATH)/m5op_arm64.o

SRC_DIR := ./
TARGET_DIR := ../out/

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

clean:
	rm -f $(OBJS) $(TARGET)
