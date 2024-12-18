# Install script for directory: /home/pi/Desktop/nnabla/third_party/hdf5-master/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/hdf5.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5api_adpt.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5public.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Apublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5ACpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Cpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Dpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Epubgen.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Epublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5ESpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Fpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDcore.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDdirect.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDfamily.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDhdfs.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDlog.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDmirror.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDmpi.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDmpio.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDmulti.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDros3.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDs3comms.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDsec2.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDsplitter.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDstdio.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5FDwindows.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Gpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Ipublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Lpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Mpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5MMpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Opublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Ppublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5PLextern.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5PLpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Rpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Spublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Tpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5VLconnector.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5VLconnector_passthru.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5VLnative.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5VLpassthru.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5VLpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Zpublic.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5Epubgen.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5version.h"
    "/home/pi/Desktop/nnabla/third_party/hdf5-master/src/H5overflow.h"
    "/home/pi/Desktop/nnabla/build/third_party/hdf5-master/src/H5pubconf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/pi/Desktop/nnabla/build/third_party/hdf5-master/bin/libhdf5.so.200.2.0"
    "/home/pi/Desktop/nnabla/build/third_party/hdf5-master/bin/libhdf5.so.200"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so.200.2.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so.200"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/pi/Desktop/nnabla/build/third_party/hdf5-master/bin/libhdf5.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libhdf5.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/pi/Desktop/nnabla/build/third_party/hdf5-master/CMakeFiles/hdf5.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/pi/Desktop/nnabla/build/third_party/hdf5-master/CMakeFiles/h5cc")
endif()

