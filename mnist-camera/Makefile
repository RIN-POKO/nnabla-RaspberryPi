# Detect architecture
ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    # for aarch64 (e.g. Jetson Nano, Raspberry Pi 4 64-bit OS)

    # Compiler and Flags
    CXX = g++
    CXXFLAGS = -I$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/include -std=c++14 `pkg-config --cflags opencv4`

    # Library Paths
    LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib

    # HDF5 Library Paths
    HDF5_LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib -lhdf5 -lhdf5_hl -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib

    # Libraries to Link
    LDLIBS = -lnnabla -lnnabla_utils -lnnabla_cli $(HDF5_LDFLAGS) `pkg-config --libs opencv4`

    # Export library path for runtime
    export LD_LIBRARY_PATH := $(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib:$(LD_LIBRARY_PATH)

else ifeq ($(ARCH), armv7l)
    # for armv7l (e.g. Raspberry Pi 32-bit OS)

    # Compiler and Flags
    CXX = g++
    CXXFLAGS = -I$(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/include -std=c++14 `pkg-config --cflags opencv4`

    # Library Paths
    LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/lib -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/lib

    # HDF5 Library Paths
    HDF5_LDFLAGS = -L$(shell pwd)/usr/lib/arm-linux-gnueabihf

    # Libraries to Link
    LDLIBS = -lnnabla -lnnabla_utils -lnnabla_cli $(HDF5_LDFLAGS) `pkg-config --libs opencv4`

    # Export library path for runtime
    export LD_LIBRARY_PATH := $(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/lib:$(LD_LIBRARY_PATH)

else ifeq ($(ARCH), x86_64)
    # for x86_64 (e.g. PC)

    # Compiler and Flags
    CXX = g++
    CXXFLAGS = -I$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/include -std=c++14 `pkg-config --cflags opencv4`

    # Library Paths
    LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib

    # HDF5 Library Paths
    HDF5_LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib -lhdf5 -lhdf5_hl

    # Libraries to Link
    LDLIBS = -lnnabla -lnnabla_utils -lnnabla_cli $(HDF5_LDFLAGS) `pkg-config --libs opencv4`

    # Export library path for runtime
    export LD_LIBRARY_PATH := $(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib:$(LD_LIBRARY_PATH)

else
    $(error Unsupported architecture: $(ARCH))
endif

# Targets
all: mnist_runtime

mnist_runtime: mnist_runtime.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

mnist_runtime.o: mnist_runtime.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f mnist_runtime.o mnist_runtime

