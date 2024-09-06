#
#  This is a SAMPLE CMakeLists.txt suitable for direct use with a new PETSc application or
#  you can add fragments of the material below to an existing application's CMakeLists.txt
#
#  Usage:  To build ex1 from ex1.c
#    rm -fr build
#    mkdir build
#    cd build
#    cmake ..
#    make
#    ./ex1
#
#  By default it gets both the compiler information and the library information from PETSc
#
cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED 17 )

# Set verbose output during configuration
# Set in vscode plugin instead
# set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "")
  add_compile_options(-g -O0)
  add_compile_definitions(DEBUG)
  # message(CMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG}")
endif()

option(USE_SLEPC "Use SLEPc" ON)

# set root of location to find PETSc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

# Remove the lines below if you do not wish to have PETSc determine the compilers
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
SET(CMAKE_C_COMPILER ${C_COMPILER})
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
if (CXX_COMPILER)
  SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
endif (CXX_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=fcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE FORTRAN_COMPILER)
if (FORTRAN_COMPILER)
  SET(CMAKE_Fortran_COMPILER ${FORTRAN_COMPILER})
  enable_language(Fortran)
endif (FORTRAN_COMPILER)

# Check cmake compilers
# message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
# message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
# message("CMAKE_Fortran_COMPILER: ${CMAKE_Fortran_COMPILER}")

find_package(PkgConfig REQUIRED)
# Check PETSc and SLEPc libraries
if (USE_SLEPC)
# Check SLEPc
  set(SLEPC $ENV{SLEPC_DIR}/$ENV{SLEPC_ARCH})
  set(ENV{PKG_CONFIG_PATH} ${SLEPC}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH})
  pkg_check_modules(SLEPC REQUIRED SLEPc)
  # if (SLEPC_FOUND)
  #   message("SLEPC_LIBRARIES: ${SLEPC_LIBRARIES}")
  #   message("SLEPC_LINK_LIBRARIES: ${SLEPC_LINK_LIBRARIES}")
  #   message("SLEPC_LIBRARY_DIRS: ${SLEPC_LIBRARY_DIRS}")
  #   message("SLEPC_LDFLAGS: ${SLEPC_LDFLAGS}")
  #   message("SLEPC_LDFLAGS_OTHER: ${SLEPC_LDFLAGS_OTHER}")
  #   message("SLEPC_INCLUDE_DIRS: ${SLEPC_INCLUDE_DIRS}")
  #   message("SLEPC_CFLAGS: ${SLEPC_CFLAGS}")
  #   message("SLEPC_CFLAGS_OTHER: ${SLEPC_CFLAGS_OTHER}")
  #   message("SLEPC_VERSION: ${SLEPC_VERSION}")
  #   message("SLEPC_PREFIX: ${SLEPC_PREFIX}")
  # endif (SLEPC_FOUND)
else ()
  pkg_check_modules(PETSC REQUIRED PETSc)
  # All variables from pkg-config are available in PETSC
  # if (PETSC_FOUND)
  #   message("PETSC_LIBRARIES: ${PETSC_LIBRARIES}")
  #   message("PETSC_LINK_LIBRARIES: ${PETSC_LINK_LIBRARIES}")
  #   message("PETSC_LIBRARY_DIRS: ${PETSC_LIBRARY_DIRS}")
  #   message("PETSC_LDFLAGS: ${PETSC_LDFLAGS}")
  #   message("PETSC_LDFLAGS_OTHER: ${PETSC_LDFLAGS_OTHER}")
  #   message("PETSC_INCLUDE_DIRS: ${PETSC_INCLUDE_DIRS}")
  #   message("PETSC_CFLAGS: ${PETSC_CFLAGS}")
  #   message("PETSC_CFLAGS_OTHER: ${PETSC_CFLAGS_OTHER}")
  #   message("PETSC_VERSION: ${PETSC_VERSION}")
  #   message("PETSC_PREFIX: ${PETSC_PREFIX}")
  # endif (PETSC_FOUND)
endif (USE_SLEPC)

# Check gsl
find_package(GSL REQUIRED)
# if (GSL_FOUND)
#   message("GSL_LIBRARIES: ${GSL_LIBRARIES}")
#   message("GSL_INCLUDE_DIRS: ${GSL_INCLUDE_DIRS}")
# endif (GSL_FOUND)

find_package(HDF5 REQUIRED COMPONENTS C)

#  tells CMake to build the application ex1 from the source file ex1.c
#  this must appear AFTER the compilers are set
project(matexpre CXX)
add_subdirectory(src)
