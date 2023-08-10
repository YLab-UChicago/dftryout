CXX = g++ 
CXXFLAGS = -std=c++11 -lm
TARGET = output
SOURCE = optimized_transformer.cpp
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
