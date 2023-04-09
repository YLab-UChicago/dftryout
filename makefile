CXX = g++
CXXFLAGS = -I -O3 -march=native -ftree-vectorize -msse4.2

all: main

main: main.cpp resnet20.cpp
	$(CXX) $(CXXFLAGS) -o main main.cpp resnet20.cpp

run: main
	./main

clean:
	rm -f main
