cmake_minimum_required(VERSION 2.8)
project(PSP)

set(PSP_VERSION_MAJOR 0)
set(PSP_VERSION_MINOR 1)

option(BUILD_TESTS "Build the test drivers" ON)
option(AVOID_COMPLEX_MPI "Avoid complex MPI routines for robustness" ON)
mark_as_advanced(AVOID_COMPLEX_MPI)

set(CXX_FLAGS "-O3" CACHE STRING "CXX optimization flags")

if(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))
  message(WARNING "CXX compiler, ${CMAKE_CXX_COMPILER} does not appear to be an MPI compiler. Please manually set it via the environment variable CXX or through the '-DCMAKE_CXX_COMPILER' cmake flag. You should probably also clean this build folder first.")
endif(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))

# We need an F90 compiler is needed for MUMPS
enable_language(Fortran)

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
  set(MATH_DESC "BLAS/LAPACK/BLACS/ScaLAPACK link flags")
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
  set(MKL_REQUIRED MKL_SCALAPACK MKL_BLACS
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
  message(STATUS 
          "WARNING: Defaulting to looking for the OpenMPI MKL BLACS library...")
# TODO: Find a way to choose the correct BLACS library
#  find_library(MKL_BLACS_LIB 
#               NAMES libmkl_blacs_lp64.a mkl_blacs_lp64.lib
#               PATHS ${INTEL_ROOT} ${MKL_ROOT}
#               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
  find_library(MKL_BLACS_LIB
               NAMES libmkl_blacs_openmpi_lp64.a mkl_blacs_openmpi_lp64.lib
               PATHS ${INTEL_ROOT} ${MKL_ROOT}
               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
  find_library(MKL_SCALAPACK_LIB 
               NAMES libmkl_scalapack_lp64.a mkl_scalapack_lp64.lib
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
    # Look for default BLAS/LAPACK/BLACS/ScaLAPACK
    # TODO: Add appropriate BLACS/ScaLAPACK libs
    set(REFERENCE_REQUIRED 
        SCALAPACK BLACSF BLACSC BLACS BLACSC BLACSF LAPACK BLAS)
    find_library(BLAS_LIB
                 NAMES blas blas.a blas.lib
                 PATHS ${REFERENCE_ROOT})
    find_library(LAPACK_LIB
                 NAMES lapack lapack.a lapack.lib
                       reflapack reflapack.a reflapack.lib
                 PATHS ${REFERENCE_ROOT})
    find_library(BLACS_LIB
                 NAMES blacs blacs.a blacs.lib
                 PATHS ${REFERENCE_ROOT})
    find_library(BLACSC_LIB
                 NAMES blacsC blacsC.a blacsC.lib
                       blacsCinit blacsCinit.a blacsCinit.lib
                 PATHS ${REFERENCE_ROOT})
    find_library(BLACSF_LIB
                 NAMES blacsF77 blacsF77.a blacsF77.lib
                       blacsF77init blacsF77init.a blacsF77init.lib
                 PATHS ${REFERENCE_ROOT})
    find_library(SCALAPACK_LIB
                 NAMES scalapack scalapack.a scalapack.lib
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
      message(STATUS "WARNING: Using reference BLAS/LAPACK/BLACS/ScaLAPACK.")
      message(STATUS "MATH_LIBS=${MATH_LIBS}")
    else(REFERENCE_FOUND)
      set(MATH_LIBS "" CACHE STRING ${MATH_DESC})
      message(FATAL_ERROR 
        "Could not find BLAS/LAPACK/BLACS/ScaLAPACK libs. Please provide the root directory of MKL with -DMKL_ROOT, the directory of reference implementations with -DREFERENCE_ROOT, or manually specify all math libraries with -DMATH_LIBS. There are numerous idiosyncratic library dependencies for BLAS/LAPACK/BLACS/ScaLAPACK, so you will almost certainly need to manually specify -DMATH_LIBS.")
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

add_subdirectory(contrib/ParMetis)
add_subdirectory(contrib/MUMPS)

#configure_file( ${CMAKE_SOURCE_DIR}/include/psp/config.h.cmake
#                ${CMAKE_BINARY_DIR}/include/psp/config.h )

# Grab all of the .c, .cpp, .h, and .hpp PSP files
file(GLOB_RECURSE PSP_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "src/*.c" "src/*.cpp")
file(GLOB_RECURSE PSP_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "include/*.h" "include/*.hpp")
set(PSP_SRC "${PSP_CPP};${PSP_HEADERS}")

# Create the lists of test executables and configure the compile flags
# for the various build modes
if(BUILD_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
  set(TESTS PanelTest ScaLAPACK-Cholesky)

  set(CMAKE_CXX_FLAGS_PURERELEASE "${CXX_PURE_RELEASE_FLAGS}")
endif(BUILD_TESTS)

# Create a dummy library in order to be able to force the math libraries
# to be linked last
add_library(cmake-dummy-lib STATIC cmake/CMakeDummyFunction.cpp)
target_link_libraries(cmake-dummy-lib ${MATH_LIBS})

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

# Build the main library (assuming it doesn't stay header-only)
add_library(psp STATIC ${PSP_SRC})
target_link_libraries(psp zmumps parmetis metis)

# Build the test drivers if necessary
if(BUILD_TESTS)
 # Build the core tests
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/")
  foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST_DIR}/${TEST}.cpp)
    target_link_libraries(${TEST} psp cmake-dummy-lib)
    set_target_properties(${TEST} 
                          PROPERTIES OUTPUT_NAME ${TEST}
                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
  endforeach(TEST)
endif(BUILD_TESTS)

