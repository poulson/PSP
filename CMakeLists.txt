cmake_minimum_required(VERSION 2.8)
project(Sweeper)

set(Sweeper_VERSION_MAJOR 0)
set(Sweeper_VERSION_MINOR 1)

option(BUILD_TESTS ON)
option(AVOID_COMPLEX_MPI "Avoid complex MPI routines for robustness" ON)
mark_as_advanced(AVOID_COMPLEX_MPI)

set(CXX_FLAGS "-O3" CACHE STRING "CXX optimization flags")

if(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))
  message(WARNING "CXX compiler, ${CMAKE_CXX_COMPILER} does not appear to be an MPI compiler. Please manually set it via the environment variable CXX or through the '-DCMAKE_CXX_COMPILER' cmake flag. You should probably also clean this build folder first.")
endif(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))

# Query the size of a void pointer in order to determine whether or not this is
# a 32-bit system
if(${CMAKE_SIZEOF_VOID_P} MATCHES 4)
  set(32_BIT_SYSTEM TRUE)
  message(STATUS "Detected 32-bit system")
else(${CMAKE_SIZEOF_VOID_P} MATCHES 4)
  set(32_BIT_SYSTEM FALSE)    
  message(STATUS "Detected 64-bit system")
endif(${CMAKE_SIZEOF_VOID_P} MATCHES 4)

# Since the CMake find_blas for some reason requires a Fortran compiler, we
# must build our own.
include(${PROJECT_SOURCE_DIR}/cmake/find_library_list.cmake)

if(MATH_LIBS)
  set(NEED_MATH FALSE)
else(MATH_LIBS)
  set(NEED_MATH TRUE)
endif(MATH_LIBS)

if(NEED_MATH)
  set(MATH_DESC "BLAS/LAPACK/BLACS/ScaLAPACK link flags")
  if(APPLE)
    set(MATH_LIBS "-framework vecLib" CACHE STRING ${MATH_DESC})
  elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Look for the base Intel libs first
    if(32_BIT_SYSTEM)
      set(INTEL_ARCH_SUBDIR lib/32)
    else(32_BIT_SYSTEM)
      set(INTEL_ARCH_SUBDIR lib/em64t)
    endif(32_BIT_SYSTEM)
    set(INTEL_LIBS guide iomp5)
    find_library_list(FOUND_INTEL_LIBS INTEL_LIBS "" ${INTEL_ARCH_SUBDIR})
    message(STATUS "FOUND_INTEL_LIBS=${FOUND_INTEL_LIBS}")
    set(MKL_LIBS mkl_scalapack_lp64 mkl_blacs_lp64 mkl_intel_lp64 
                 mkl_intel_thread mkl_core)
    find_library_list(FOUND_MKL_LIBS MKL_LIBS "" ${INTEL_ARCH_SUBDIR})
    message(STATUS "FOUND_MKL_LIBS=${FOUND_MKL_LIBS}")
    find_library(PTHREAD_LIB pthread DOC "Pthreads library")
    if(FOUND_INTEL_LIBS AND FOUND_MKL_LIBS AND FOUND_PTHREAD_LIB)
      set(MATH_LIBS ${FOUND_MKL_LIBS} ${FOUND_INTEL_LIBS} ${PTHREAD_LIB} 
          CACHE STRING ${MATH_DESC})
      message(STATUS "Using MKL for BLAS/LAPACK/BLACS/ScaLAPACK.")
      set(NEED_MATH FALSE)
    else(FOUND_INTEL_LIBS AND FOUND_MKL_LIBS AND FOUND_PTHREAD_LIB)
      # Look for default BLAS/LAPACK/BLACS/ScaLAPACK
      # TODO: Add appropriate BLACS/ScaLAPACK libs
      set(REFERENCE_LIBS scalapack blacs blacsF77 lapack blas)
      find_library_list(REF_MATH_LIBS REFERENCE_LIBS "" "")
      if(REF_MATH_LIBS)
        set(MATH_LIBS ${REF_MATH_LIBS} CACHE STRING ${MATH_DESC})
        message(STATUS "WARNING: Using reference BLAS/LAPACK/BLACS/ScaLAPACK.")
      else(REF_MATH_LIBS)
        set(MATH_LIBS "" CACHE STRING ${MATH_DESC})
        message(FATAL_ERROR 
          "Could not find BLAS/LAPACK/BLACS/ScaLAPACK libs. Please manually specify MATH_LIBS.")
      endif(REF_MATH_LIBS)
    endif(FOUND_INTEL_LIBS AND FOUND_MKL_LIBS AND FOUND_PTHREAD_LIB)
  endif(APPLE)
endif(NEED_MATH)

# Attempt to detect the BLAS/LAPACK underscore conventions. 
# We currently only handle whether or not there is an underscore appended.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS})
check_function_exists(daxpy HAVE_DAXPY)
if(HAVE_DAXPY)
  set(BLAS_POST FALSE)
  set(BLAS_DEFS "")
else(HAVE_DAXPY)
  check_function_exists(daxpy_ HAVE_DAXPY_POST)
  if(HAVE_DAXPY_POST)
    set(BLAS_POST TRUE)
    set(BLAS_DEFS "-DBLAS_POST")
  else(HAVE_DAXPY_POST)
    message(FATAL_ERROR "Could not determine BLAS format.")
  endif(HAVE_DAXPY_POST)
endif(HAVE_DAXPY)
check_function_exists(dpotrf HAVE_DPOTRF)
if(HAVE_DPOTRF)
  set(LAPACK_POST FALSE)
  set(LAPACK_DEFS "")
else(HAVE_DPOTRF)
  check_function_exists(dpotrf_ HAVE_DPOTRF_POST)
  if(HAVE_DPOTRF_POST)
    set(LAPACK_POST TRUE)
    set(LAPACK_DEFS "-DLAPACK_POST")
  else(HAVE_DPOTRF_POST)
    message(FATAL_ERROR "Could not determine LAPACK format.")
  endif(HAVE_DPOTRF_POST)
endif(HAVE_DPOTRF)

#add_subdirectory(contrib/ParMetis)
#add_subdirectory(contrib/MUMPS)

#configure_file( ${CMAKE_SOURCE_DIR}/include/sweeper/config.h.cmake
#                ${CMAKE_BINARY_DIR}/include/sweeper/config.h )

# Grab all of the .c, .cpp, .h, and .hpp Sweeper files
file(GLOB_RECURSE SWEEPER_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "src/*.c" "src/*.cpp")
file(GLOB_RECURSE SWEEPER_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "include/*.h" "include/*.hpp")
set(SWEEPER_SRC "${SWEEPER_CPP};${SWEEPER_HEADERS}")

# Create the lists of test executables and configure the compile flags
# for the various build modes
#if(BUILD_TESTS)
#  set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
#
#  set(CORE_TESTS DifferentGrids DistMatrix Matrix)
#
#  set(CMAKE_CXX_FLAGS_PURERELEASE
#      "${CXX_PURE_RELEASE_FLAGS}")
#endif(BUILD_TESTS)

# Create a dummy library in order to be able to force the math libraries
# to be linked last
#add_library(cmake-dummy-lib STATIC cmake/CMakeDummyFunction.cpp)
#target_link_libraries(cmake-dummy-lib ${MATH_LIBS})

# Copy the headers into the build directory
#foreach(HEADER ${SWEEPER_HEADERS})
#  add_custom_command(
#    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
#    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}
#            ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
#  )
#endforeach(HEADER)

# Make sure the Sweeper headers can be found
#include_directories("${PROJECT_BINARY_DIR}/include")

# Build the main library
#add_library(sweeper STATIC ${SWEEPER_SRC})
#target_link_libraries(sweeper zmumps parmetis metis)

# Build the test drivers if necessary
#if(BUILD_TESTS)
  # Build the core tests
#  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/core/")
#  foreach(TEST ${CORE_TESTS})
#    add_executable(core-${TEST} ${TEST_DIR}/core/${TEST}.cpp)
#    target_link_libraries(core-${TEST} sweeper cmake-dummy-lib)
#    set_target_properties(core-${TEST} 
#                          PROPERTIES OUTPUT_NAME ${TEST}
#                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
#  endforeach(TEST)
#endif(BUILD_TESTS)

