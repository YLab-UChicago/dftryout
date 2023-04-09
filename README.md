# dataflow_tryout
Manually implementing individual dataflow

To run, simply compile by the following command: 
```
g++ -I eigen -O3 -march=native -ftree-vectorize -msse4.2 main.cpp -o main
```

or alternatively, perform 
```
make
```

TODOs:
0. start implementing different dataflows in stationaries.cpp
    note that now when we refer to stationaries, it is meant to refer to the function that performs the convolution operation
1. construct resnet & write script to automate other classes
2. replace matrix multiplication operation from Eigen to our own intrinsics
3. Potentially integrate PyTorch/Tensorflow but replace their kernels / intrinsics multiplications by our own implementations 

To install Eigen:
```
http://eigen.tuxfamily.org/index.php?title=Main_Page#Download
```

Some internal documentations:

This makefile defines a few different targets:

all is the default target and it depends on the main target.
main is the target for building the executable and it depends on main.cpp and resnet.cpp files.
run is the target for running the executable.
clean is the target for cleaning up the build files.
You can use the makefile by running the make command in the terminal.
For example, to build the program, you can run make or make all
To run the program, you can run make run
And to clean the build files, you can run make clean

It's worth noting that this is a simple example of a makefile, in practice, makefiles can be much more complex and include additional targets and dependencies depending on the project.
Also, you may need to adjust the makefile according to your system's settings and the path of the libraries and headers.