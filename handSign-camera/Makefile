# Detect architecture
ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    # for aarch64 (e.g. Jetson Nano, Raspberry Pi 4 64-bit OS)

    # Compiler and Flags
    CXX = g++
    CXXFLAGS = -I$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/include -I/usr/include/libcamera -I$(shell pwd)/LCCV/include -std=c++17 `pkg-config --cflags opencv4`

    # Library Paths
    LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib -L/usr/lib/aarch64-linux-gnu \
              -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib -Wl,-rpath,/usr/lib/aarch64-linux-gnu

    # HDF5 Library Paths
    HDF5_LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib -lhdf5 -lhdf5_hl -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib

    # Libraries to Link
    LDLIBS = -lnnabla -lnnabla_utils -lnnabla_cli $(HDF5_LDFLAGS) -lcamera -lcamera-base -lcamera_app `pkg-config --libs opencv4`

    # Export library path for runtime
    export LD_LIBRARY_PATH := $(shell pwd)/nnabla-cpplib-1.38.0-Linux_aarch64/lib:/usr/lib/aarch64-linux-gnu:$(LD_LIBRARY_PATH)

else ifeq ($(ARCH), armv7l)
    # for armv7l (e.g. Raspberry Pi 32-bit OS)

    # Compiler and Flags
    CXX = g++
    CXXFLAGS = -I$(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/include -I/usr/include/libcamera -I$(shell pwd)/LCCV/include -std=c++17 `pkg-config --cflags opencv4`

    # Library Paths
    LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/lib -L/usr/lib/arm-linux-gnueabihf \
              -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/lib -Wl,-rpath,/usr/lib/arm-linux-gnueabihf

    # HDF5 Library Paths
    HDF5_LDFLAGS = -L$(shell pwd)/usr/lib/arm-linux-gnueabihf

    # Libraries to Link
    LDLIBS = -lnnabla -lnnabla_utils -lnnabla_cli $(HDF5_LDFLAGS) -lcamera -lcamera-base -lcamera_app `pkg-config --libs opencv4`

    # Export library path for runtime
    export LD_LIBRARY_PATH := $(shell pwd)/nnabla-cpplib-1.40.0.-Linux_armv7l/lib:/usr/lib/arm-linux-gnueabihf:$(LD_LIBRARY_PATH)

else ifeq ($(ARCH), x86_64)
    # for x86_64 (e.g. PC)

    # Compiler and Flags
    CXX = g++
    CXXFLAGS = -I$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/include -I/usr/include/libcamera -I$(shell pwd)/LCCV/include -std=c++17 `pkg-config --cflags opencv4`

    # Library Paths
    LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib -L/usr/lib/x86_64-linux-gnu \
              -Wl,-rpath,$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu

    # HDF5 Library Paths
    HDF5_LDFLAGS = -L$(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib -lhdf5 -lhdf5_hl

    # Libraries to Link
    LDLIBS = -lnnabla -lnnabla_utils -lnnabla_cli $(HDFLDFLAGS) -lcamera -lcamera-base -lcamera_app `pkg-config --libs opencv4`

    # Export library path for runtime
    export LD_LIBRARY_PATH := $(shell pwd)/nnabla-cpplib-1.39.0-Rocky8.9_x86_64/lib:/usr/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH)

else
    $(error Unsupported architecture: $(ARCH))
endif

# Targets
all: mnist_runtime

# Include lccv.o, libcamera_app.o, and libcamera_app_options.o in the build process
mnist_runtime: mnist_runtime.o lccv.o libcamera_app.o libcamera_app_options.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

mnist_runtime.o: mnist_runtime.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

lccv.o: LCCV/src/lccv.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

libcamera_app.o: LCCV/src/libcamera_app.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

libcamera_app_options.o: LCCV/src/libcamera_app_options.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f mnist_runtime.o lccv.o libcamera_app.o libcamera_app_options.o mnist_runtime
