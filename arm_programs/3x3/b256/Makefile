
CC := aarch64-linux-gnu-gcc-10
CFLAGS := -march=armv8.2-a+sve2
LDFLAGS := -lstdc++

SRC_DIR := src/
TARGET_BASE := out/

SRC ?= $(SRC_DIR)os_ws9_is6.cpp
TARGET := $(TARGET_BASE)$(basename $(notdir $(SRC)))

OBJ := $(SRC:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)