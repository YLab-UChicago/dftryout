# Compiler
CXX = g++ 

# Compiler flags
CXXFLAGS = -std=c++11 -lm

# Target file
TARGET = output

# Source file
SOURCE = optimized_transformer.cpp

# Build rule
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
