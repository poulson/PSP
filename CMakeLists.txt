#
#  Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
#  of a sweeping preconditioner for 3d Helmholtz equations.
#
#  Copyright (C) 2011 Jack Poulson, Lexing Ying, and
#  The University of Texas at Austin
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
cmake_minimum_required(VERSION 2.8)
project(PSP)

set(PSP_VERSION_MAJOR 0)
set(PSP_VERSION_MINOR 2)

option(RELEASE "Avoid unnecessary assertions." OFF)
option(VIEW_MATRICES "Print/draw the preconditioner matrices." OFF)
option(BUILT_PETSC_WITH_X11 "Was petsc-dev built with X11 support?" OFF)

if(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))
  message(WARNING "CXX compiler, ${CMAKE_CXX_COMPILER} does not appear to be an MPI compiler. Please manually set it via the environment variable CXX or through the '-DCMAKE_CXX_COMPILER' cmake flag. You should probably also clean this build folder first.")
endif(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))
if(NOT (CMAKE_C_COMPILER MATCHES "mpi"))
  message(WARNING "C compiler, ${CMAKE_C_COMPILER} does not appear to be an MPI compiler. Please manually set it via the environment variable CC or through the '-DCMAKE_C_COMPILER' cmake flag. You should probably also clean this build folder first.")
endif(NOT (CMAKE_C_COMPILER MATCHES "mpi"))

# We need PETSc's GMRES, so go ahead and use its dependencies, which we must 
# link anyways.
#if(NOT PETSC_DIR)    
#  message(
#    FATAL_ERROR
#    "PETSC_DIR must be set to the location of the top-level PETSc folder."
#  )
#endif(NOT PETSC_DIR)
#if(NOT PETSC_ARCH_SUBDIR)
#  message(
#    FATAL_ERROR
#    "PETSC_ARCH_SUBDIR must be set to the architecture-specific PETSc subdirectory."
#  )
#endif(NOT PETSC_ARCH_SUBDIR)
#set(PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
#set(PETSC_ARCH_INCLUDE_DIR "${PETSC_DIR}/${PETSC_ARCH_SUBDIR}/include")
#set(PETSC_LIB_DIR "${PETSC_DIR}/${PETSC_ARCH_SUBDIR}/lib")
#find_library(PETSC_LIB NAMES petsc HINTS ${PETSC_LIB_DIR})
#if(NOT PETSC_LIB)
#  message(
#    FATAL_ERROR
#    "Could not find PETSc using PETSC_DIR and PETSC_ARCH_SUBDIR. Please make sure it was built."
#  )
#endif(NOT PETSC_LIB)

# X11 library
#find_library(X11_LIB NAMES X11)
#if(NOT X11_LIB)
#  if(BUILT_PETSC_WITH_X11)
#    message(FATAL_ERROR "Could not find X11 lib.")
#  else(BUILT_PETSC_WITH_X11)
#    set(X11_LIB "")
#  endif(BUILT_PETSC_WITH_X11)
#endif(NOT X11_LIB)

# Dynamic library
find_library(DL_LIB NAMES dl)
if(NOT DL_LIB)
  message(FATAL_ERROR "Could not find dl library.")
endif(NOT DL_LIB)

option(BUILD_TESTS "Build the test drivers" ON)
option(AVOID_COMPLEX_MPI "Avoid complex MPI routines for robustness" ON)
mark_as_advanced(AVOID_COMPLEX_MPI)

set(CXX_FLAGS "-O3" CACHE STRING "CXX flags")
set(C_FLAGS "-O3" CACHE STRING "C flags")

set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
set(CMAKE_C_FLAGS "${C_FLAGS}")

# Query the size of a void pointer in order to determine whether or not this is
# a 32-bit system
if(${CMAKE_SIZEOF_VOID_P} MATCHES 4)
  set(32_BIT_SYSTEM TRUE)
  message(STATUS "Detected 32-bit system")
else(${CMAKE_SIZEOF_VOID_P} MATCHES 4)
  set(32_BIT_SYSTEM FALSE)    
  message(STATUS "Detected 64-bit system")
endif(${CMAKE_SIZEOF_VOID_P} MATCHES 4)

if(MATH_LIBS)
  set(NEED_MATH FALSE)
else(MATH_LIBS)
  set(NEED_MATH TRUE)
endif(MATH_LIBS)

if(NEED_MATH)
  set(MATH_DESC "BLAS/LAPACK link flags")
  # Look for the base Intel libs first
  if(32_BIT_SYSTEM)
    set(INTEL_ARCH_SUBDIRS lib/32)
    set(MKL_ARCH_SUBDIRS lib/32)
  else(32_BIT_SYSTEM)
    set(INTEL_ARCH_SUBDIRS lib/em64t lib/intel64)
    set(MKL_ARCH_SUBDIRS lib/em64t lib/intel64 mkl/lib/em64t)
  endif(32_BIT_SYSTEM)
  # There is a cyclic dependency between MKL_CORE and MKL_SEQUENTIAL with
  # version 11.1/072, so we sandwich MKL_CORE between MKL_SEQUENTIALs
  set(MKL_REQUIRED 
      MKL_INTEL MKL_SEQUENTIAL MKL_CORE MKL_SEQUENTIAL GUIDE IOMP5 PTHREAD)
  find_library(PTHREAD_LIB pthread)
  find_library(IOMP5_LIB 
               NAMES libiomp5.a iomp5.lib 
               PATHS ${INTEL_ROOT} ${MKL_ROOT}
               PATH_SUFFIXES ${INTEL_ARCH_SUBDIRS})
  find_library(GUIDE_LIB 
               NAMES libguide.a guide.lib
               PATHS ${INTEL_ROOT} ${MKL_ROOT}
               PATH_SUFFIXES ${INTEL_ARCH_SUBDIRS})
  find_library(MKL_CORE_LIB 
               NAMES libmkl_core.a mkl_core.lib
               PATHS ${INTEL_ROOT} ${MKL_ROOT}
               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
  find_library(MKL_SEQUENTIAL_LIB 
               NAMES libmkl_sequential.a mkl_sequential.lib
               PATHS ${INTEL_ROOT} ${MKL_ROOT}
               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
  find_library(MKL_INTEL_LIB 
               NAMES libmkl_intel_lp64.a mkl_intel_lp64.lib
               PATHS ${INTEL_ROOT} ${MKL_ROOT}
               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
  set(MKL_FOUND TRUE)
  set(MATH_LIBS "")
  foreach(NAME ${MKL_REQUIRED})
    if( ${NAME}_LIB )
      message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
      list(APPEND MATH_LIBS ${${NAME}_LIB})
    else( ${NAME}_LIB )
      message(STATUS "Could not find ${NAME}_LIB")
      set(MKL_FOUND FALSE)
    endif( ${NAME}_LIB )
  endforeach(NAME)
  if(MKL_FOUND)
    set(NEED_MATH FALSE)
    message(STATUS "Using MKL's math libraries.")
  else(MKL_FOUND)
    # Look for default BLAS/LAPACK
    set(REFERENCE_REQUIRED LAPACK BLAS)
    find_library(BLAS_LIB
                 NAMES blas blas.a blas.lib
                 PATHS ${REFERENCE_ROOT})
    find_library(LAPACK_LIB
                 NAMES lapack lapack.a lapack.lib
                       reflapack reflapack.a reflapack.lib
                 PATHS ${REFERENCE_ROOT})
    set(REFERENCE_FOUND ON)
    set(MATH_LIBS "")
    foreach(NAME ${REFERENCE_REQUIRED})
      if( ${NAME}_LIB )
        message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
        list(APPEND MATH_LIBS ${${NAME}_LIB})
      else( ${NAME}_LIB )
        message(STATUS "Could not find ${NAME}_LIB")
        set(REFERENCE_FOUND OFF)
      endif( ${NAME}_LIB )
    endforeach(NAME)
    message(STATUS "REFERENCE_FOUND=${REFERENCE_FOUND}")
    if(REFERENCE_FOUND)
      message(STATUS "WARNING: Using reference BLAS/LAPACK.")
      message(STATUS "MATH_LIBS=${MATH_LIBS}")
    else(REFERENCE_FOUND)
      set(MATH_LIBS "" CACHE STRING ${MATH_DESC})
      message(FATAL_ERROR 
        "Could not find BLAS/LAPACK libs. Please provide the root directory of MKL with -DMKL_ROOT, the directory of reference implementations with -DREFERENCE_ROOT, or manually specify all math libraries with -DMATH_LIBS. There are numerous idiosyncratic library dependencies for BLAS/LAPACK, so you will almost certainly need to manually specify -DMATH_LIBS.")
    endif(REFERENCE_FOUND)
  endif(MKL_FOUND)
  # Append the standard math libraries to the link list.
  list(APPEND MATH_LIBS m)
endif(NEED_MATH)

# Attempt to detect the BLAS/LAPACK underscore conventions. 
# We currently only handle whether or not there is an underscore appended.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS})
check_function_exists(daxpy HAVE_DAXPY)
if(HAVE_DAXPY)
  set(BLAS_POST FALSE)
else(HAVE_DAXPY)
  check_function_exists(daxpy_ HAVE_DAXPY_POST)
  if(HAVE_DAXPY_POST)
    set(BLAS_POST TRUE)
  else(HAVE_DAXPY_POST)
    message(FATAL_ERROR "Could not determine BLAS format.")
  endif(HAVE_DAXPY_POST)
endif(HAVE_DAXPY)
check_function_exists(dpotrf HAVE_DPOTRF)
if(HAVE_DPOTRF)
  set(LAPACK_POST FALSE)
else(HAVE_DPOTRF)
  check_function_exists(dpotrf_ HAVE_DPOTRF_POST)
  if(HAVE_DPOTRF_POST)
    set(LAPACK_POST TRUE)
  else(HAVE_DPOTRF_POST)
    message(FATAL_ERROR "Could not determine LAPACK format.")
  endif(HAVE_DPOTRF_POST)
endif(HAVE_DPOTRF)

# Look for restrict support
include(CheckCXXSourceCompiles)
set(RESTRICT_CODE
    "int main(void)
     {
         int* RESTRICT a;
         return 0;
     }
    ")
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict__")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict__)
if(HAVE___restrict__)
  set(RESTRICT "__restrict__")
  message(STATUS "Using __restrict__ keyword.")
else(HAVE___restrict__)
  set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict")
  check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict)
  if(HAVE___restrict)
    set(RESTRICT "__restrict")
    message(STATUS "Using __restrict keyword.")
  else(HAVE___restrict)
    set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=restrict")
    check_cxx_source_compiles("${RESTRICT_CODE}" HAVE_restrict)
    if(HAVE_restrict)
      set(RESTRICT "restrict")
      message(STATUS "Using restrict keyword.")
    else(HAVE_restrict)
      set(RESTRICT "")
      message(STATUS "Could not find a restrict keyword.")
    endif(HAVE_restrict)
  endif(HAVE___restrict)
endif(HAVE___restrict__)

# Store all of our system-specific definitions
configure_file( 
  ${CMAKE_SOURCE_DIR}/include/psp/config.h.cmake
  ${CMAKE_BINARY_DIR}/include/psp/config.h 
)

# Grab all of the .c, .cpp, .h, and .hpp PSP files
file(GLOB_RECURSE PSP_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "src/*.c" "src/*.cpp")
file(GLOB_RECURSE PSP_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "include/*.h" "include/*.hpp")
set(PSP_SRC "${PSP_CPP};${PSP_HEADERS}")

# Create the lists of test executables and configure the compile flags
# for the various build modes
if(BUILD_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)
  set(HMATRIX_TOOLS_TESTS MatrixAdd Compress ConvertSubmatrix)
  set(QUASI2D_TESTS Invert LowRankConstructor Pack)
endif(BUILD_TESTS)

# Create a dummy library in order to be able to force the math libraries
# to be linked last
add_library(cmake-dummy-lib STATIC cmake/CMakeDummyFunction.cpp)
set(EXTERNAL_LIBS "${PETSC_LIB}")
list(APPEND EXTERNAL_LIBS "${MATH_LIBS}")
list(APPEND EXTERNAL_LIBS "${X11_LIB}")
list(APPEND EXTERNAL_LIBS "${DL_LIB}")
target_link_libraries(cmake-dummy-lib ${EXTERNAL_LIBS})

# Copy the headers into the build directory
foreach(HEADER ${PSP_HEADERS})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}
            ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
  )
endforeach(HEADER)

# Make sure the PSP headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

# Make sure that the PETSc includes can be found
#include_directories("${PETSC_INCLUDE_DIR}")
#include_directories("${PETSC_ARCH_INCLUDE_DIR}")

# Build the PSP source
add_library(psp STATIC ${PSP_SRC})

# Build the test drivers if necessary
if(BUILD_TESTS)
  # Build the hmatrix_tools tests
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/hmatrix_tools")
  foreach(TEST ${HMATRIX_TOOLS_TESTS})
    add_executable(${TEST} ${TEST_DIR}/hmatrix_tools/${TEST}.cpp)
    target_link_libraries(${TEST} psp cmake-dummy-lib)
    set_target_properties(${TEST} 
                          PROPERTIES OUTPUT_NAME ${TEST}
                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
  endforeach(TEST)
  # Build the Quasi2dHMatrix tests
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/quasi2d_hmatrix")
  foreach(TEST ${QUASI2D_TESTS})
    add_executable(${TEST} ${TEST_DIR}/quasi2d_hmatrix/${TEST}.cpp)
    target_link_libraries(${TEST} psp cmake-dummy-lib)
    set_target_properties(${TEST}
                          PROPERTIES OUTPUT_NAME ${TEST}
                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
  endforeach(TEST)
endif(BUILD_TESTS)

