cmake_minimum_required(VERSION 2.8)
project(PSP)

set(PSP_VERSION_MAJOR 0)
set(PSP_VERSION_MINOR 1)

if(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))
  message(WARNING "CXX compiler, ${CMAKE_CXX_COMPILER} does not appear to be an MPI compiler. Please manually set it via the environment variable CXX or through the '-DCMAKE_CXX_COMPILER' cmake flag. You should probably also clean this build folder first.")
endif(NOT (CMAKE_CXX_COMPILER MATCHES "mpi"))
if(NOT (CMAKE_C_COMPILER MATCHES "mpi"))
  message(WARNING "C compiler, ${CMAKE_C_COMPILER} does not appear to be an MPI compiler. Please manually set it via the environment variable CC or through the '-DCMAKE_C_COMPILER' cmake flag. You should probably also clean this build folder first.")
endif(NOT (CMAKE_C_COMPILER MATCHES "mpi"))

# AMD (linked but not used)
if(NOT AMD_LIB)
  message(
    FATAL_ERROR
    "AMD_LIB must be set to the location of the AMD library."
  )
endif(NOT AMD_LIB)

# Chaco (linked but not used)
if(NOT CHACO_LIB)
  message(
    FATAL_ERROR
    "CHACO_LIB must be set to the location of the Chaco library."
  )
endif(NOT CHACO_LIB)

# HYPRE (linked but not used)
if(NOT HYPRE_LIB)
  message(
    FATAL_ERROR
    "HYPRE_LIB must be set to the location of the HYPRE library."
  )
endif(NOT HYPRE_LIB)

# Metis + ParMetis
if(NOT METIS_LIB)
  message(
    FATAL_ERROR
    "METIS_LIB must be set to the location of the Metis library."
  )
endif(NOT METIS_LIB)
if(NOT PARMETIS_LIB)
  message(
    FATAL_ERROR
    "PARMETIS_LIB must be set to the location of the ParMetis library."
  )
endif(NOT PARMETIS_LIB)

# MUMPS
if(NOT MUMPS_LIBS)
  message(
    FATAL_ERROR
    "MUMPS_LIBS must be a semicolon delimited list of the mumps libraries."
  )
endif(NOT MUMPS_LIBS)

# PETSc
if(NOT PETSC_INCLUDE_DIR)
  message(
    FATAL_ERROR 
    "PETSC_INCLUDE_DIR must be set to the location of the PETSc include folder."
  )
endif(NOT PETSC_INCLUDE_DIR)
if(NOT PETSC_ARCH_INCLUDE_DIR)
  message(
    FATAL_ERROR 
    "PETSC_ARCH_INCLUDE_DIR must be set to the location of the PETSc arch include folder."
  )
endif(NOT PETSC_ARCH_INCLUDE_DIR)
if(NOT PETSC_LIB)
  message(
    FATAL_ERROR
    "PETSC_LIB must be set to the full path of a PETSc library."
  )
endif(NOT PETSC_LIB)

# PLAPACK (linked but not used)
if(NOT PLAPACK_LIB)
  message(
    FATAL_ERROR
    "PLAPACK_LIB must be set to the location of the PLAPACK library."
  )
endif(NOT PLAPACK_LIB)

# PORD (linked but not used)
if(NOT PORD_LIB)
  message(
    FATAL_ERROR
    "PORD_LIB must be set to the location of the PORD library."
  )
endif(NOT PORD_LIB)

# Prometheus (linked but not used)
if(NOT PROMETHEUS_LIBS)
  message(
    FATAL_ERROR
    "PROMETHEUS_LIBS must be the semicolon-delimited list of Prometheus libs."
  )
endif(NOT PROMETHEUS_LIBS)

# ScaLAPACK + BLACS
if(NOT SCALAPACK_LIB)
  message(
    FATAL_ERROR
    "SCALAPACK_LIB must be set to the location of the ScaLAPACK library."
  )
endif(NOT SCALAPACK_LIB)
if(NOT BLACS_LIBS)
  message(
    FATAL_ERROR
    "BLACS_LIBS must be a semicolon delimited list of the BLACS libraries."
  )
endif(NOT BLACS_LIBS)

# SPAI (linked but not used)
if(NOT SPAI_LIB)
  message(
    FATAL_ERROR
    "SPAI_LIB must be set to the location of the SPAI library."
  )
endif(NOT SPAI_LIB)

# SPOOLES (linked but not used)
if(NOT SPOOLES_LIB)
  message(
    FATAL_ERROR
    "SPOOLES_LIB must be set to the location of the SPOOLES library."
  )
endif(NOT SPOOLES_LIB)

# SuperLU (linked but not used)
if(NOT SUPERLU_LIB)
  message(
    FATAL_ERROR
    "SUPERLU_LIB must be set to the location of the SuperLU library."
  )
endif(NOT SUPERLU_LIB)
if(NOT SUPERLU_DIST_LIB)
  message(
    FATAL_ERROR
    "SUPERLU_DIST_LIB must point to the distributed SuperLU library."
  )
endif(NOT SUPERLU_DIST_LIB)

# Triangle (linked but not used)
if(NOT TRIANGLE_LIB)
  message(
    FATAL_ERROR
    "TRIANGLE_LIB must be set to the location of the Triangle library."
  )
endif(NOT TRIANGLE_LIB)

# UMFPACK (linked but not used)
if(NOT UMFPACK_LIB)
  message(
    FATAL_ERROR
    "UMFPACK_LIB must be set to the location of the UMFPACK library."
  )
endif(NOT UMFPACK_LIB)

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
# For now, default to the user specifying the BLACS/ScaLAPACK libs
#
#  message(STATUS 
#          "WARNING: Defaulting to looking for the OpenMPI MKL BLACS library...")
# TODO: Find a way to choose the correct BLACS library
#  find_library(MKL_BLACS_LIB 
#               NAMES libmkl_blacs_lp64.a mkl_blacs_lp64.lib
#               NAMES libmkl_blacs_openmpi_lp64.a mkl_blacs_openmpi_lp64.lib
#               PATHS ${INTEL_ROOT} ${MKL_ROOT}
#               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
#  find_library(MKL_SCALAPACK_LIB 
#               NAMES libmkl_scalapack_lp64.a mkl_scalapack_lp64.lib
#               PATHS ${INTEL_ROOT} ${MKL_ROOT}
#               PATH_SUFFIXES ${MKL_ARCH_SUBDIRS})
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
# For now, default to the user specifying the BLACS/ScaLAPACK libs
#
#    find_library(BLACS_LIB
#                 NAMES blacs blacs.a blacs.lib
#                 PATHS ${REFERENCE_ROOT})
#    find_library(BLACSC_LIB
#                 NAMES blacsC blacsC.a blacsC.lib
#                       blacsCinit blacsCinit.a blacsCinit.lib
#                 PATHS ${REFERENCE_ROOT})
#    find_library(BLACSF_LIB
#                 NAMES blacsF77 blacsF77.a blacsF77.lib
#                       blacsF77init blacsF77init.a blacsF77init.lib
#                 PATHS ${REFERENCE_ROOT})
#    find_library(SCALAPACK_LIB
#                 NAMES scalapack scalapack.a scalapack.lib
#                 PATHS ${REFERENCE_ROOT})
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

check_function_exists(mkl_set_num_threads HAVE_MKL)

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
  set(TESTS PanelTest PetscTest)
endif(BUILD_TESTS)

# Create a dummy library in order to be able to force the math libraries
# to be linked last
add_library(cmake-dummy-lib STATIC cmake/CMakeDummyFunction.cpp)
set(EXTERNAL_LIBS "${PETSC_LIB}")
list(APPEND EXTERNAL_LIBS "${AMD_LIB}")
list(APPEND EXTERNAL_LIBS "${HYPRE_LIB}")
list(APPEND EXTERNAL_LIBS "${PROMETHEUS_LIBS}")
list(APPEND EXTERNAL_LIBS "${SPAI_LIB}")
list(APPEND EXTERNAL_LIBS "${SPOOLES_LIB}")
list(APPEND EXTERNAL_LIBS "${TRIANGLE_LIB}")
list(APPEND EXTERNAL_LIBS "${UMFPACK_LIB}")
list(APPEND EXTERNAL_LIBS "${MUMPS_LIBS}")
list(APPEND EXTERNAL_LIBS "${SUPERLU_DIST_LIB};${SUPERLU_LIB}")
list(APPEND EXTERNAL_LIBS "${CHACO_LIB}")
list(APPEND EXTERNAL_LIBS "${PARMETIS_LIB};${METIS_LIB}")
list(APPEND EXTERNAL_LIBS "${PORD_LIB}")
list(APPEND EXTERNAL_LIBS "${PLAPACK_LIB}")
list(APPEND EXTERNAL_LIBS "${SCALAPACK_LIB};${BLACS_LIBS}")
list(APPEND EXTERNAL_LIBS "${MATH_LIBS}")
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
include_directories("${PETSC_INCLUDE_DIR}")
include_directories("${PETSC_ARCH_INCLUDE_DIR}")

# Build the main library (assuming it doesn't stay header-only)
add_library(psp STATIC ${PSP_SRC})

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

