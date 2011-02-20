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

# We need PETSc's GMRES, so go ahead and use its dependencies, which we must link
# anyways.
if(NOT PETSC_DIR)    
  message(
    FATAL_ERROR
    "PETSC_DIR must be set to the location of the top-level PETSc folder."
  )
endif(NOT PETSC_DIR)
if(NOT PETSC_ARCH_DIR)
  message(
    FATAL_ERROR
    "PETSC_ARCH_DIR must be set to the architecture-specific PETSc directory."
  )
endif(NOT PETSC_ARCH_DIR)
set(PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
set(PETSC_ARCH_INCLUDE_DIR "${PETSC_ARCH_DIR}/include")
set(PETSC_LIB_DIR "${PETSC_ARCH_DIR}/lib")
find_library(PETSC_LIB NAMES petsc HINTS ${PETSC_LIB_DIR})
if(NOT PETSC_LIB)
  message(
    FATAL_ERROR
    "Could not find PETSc using PETSC_ARCH_DIR. Please make sure it was built."
  )
endif(NOT PETSC_LIB)

# AMD (linked but not used)
find_library(AMD_LIB NAMES amd HINTS ${PETSC_LIB_DIR})
if(NOT AMD_LIB )
  message(WARNING "Could not find AMD using PETSC_ARCH_DIR.")
endif(NOT AMD_LIB)

# Chaco (linked but not used)
find_library(CHACO_LIB NAMES chaco HINTS ${PETSC_LIB_DIR})
if(NOT CHACO_LIB )
  message(WARNING "Could not find Chaco using PETSC_ARCH_DIR.")
endif(NOT CHACO_LIB)

# HYPRE (linked but not used
find_library(HYPRE_LIB NAMES HYPRE HINTS ${PETSC_LIB_DIR})
if(NOT HYPRE_LIB )
  message(WARNING "Could not find HYPRE using PETSC_ARCH_DIR.")
endif(NOT HYPRE_LIB)

# Metis + ParMetis
find_library(METIS_LIB NAMES metis HINTS ${PETSC_LIB_DIR})
if(NOT METIS_LIB)
  message(
    FATAL_ERROR
    "Could not find Metis library in PETSC_ARCH_DIR. Make sure PETSc built it."
  )
endif(NOT METIS_LIB)
find_library(PARMETIS_LIB NAMES parmetis HINTS ${PETSC_LIB_DIR})
if(NOT PARMETIS_LIB)
  message(
    FATAL_ERROR
    "Could not find ParMetis lib in PETSC_ARCH_DIR. Make sure PETSc built it."
  )
endif(NOT PARMETIS_LIB)

# MUMPS
find_library(CMUMPS_LIB NAMES cmumps HINTS ${PETSC_LIB_DIR})
find_library(DMUMPS_LIB NAMES dmumps HINTS ${PETSC_LIB_DIR})
find_library(SMUMPS_LIB NAMES smumps HINTS ${PETSC_LIB_DIR})
find_library(ZMUMPS_LIB NAMES zmumps HINTS ${PETSC_LIB_DIR})
find_library(MUMPS_COMMON_LIB NAMES mumps_common HINTS ${PETSC_LIB_DIR})
if(NOT CMUMPS_LIB OR NOT DMUMPS_LIB OR NOT SMUMPS_LIB OR NOT ZMUMPS_LIB 
   OR NOT MUMPS_COMMON_LIB)
  message(
    FATAL_ERROR
    "Could not find MUMPS libraries in PETSC_ARCH_DIR. Make sure they exist."
  )
endif(NOT CMUMPS_LIB OR NOT DMUMPS_LIB OR NOT SMUMPS_LIB OR NOT ZMUMPS_LIB 
      OR NOT MUMPS_COMMON_LIB)
set(MUMPS_LIBS 
  "${CMUMPS_LIB};${DMUMPS_LIB};${SMUMPS_LIB};${ZMUMPS_LIB};${MUMPS_COMMON_LIB}")

# PLAPACK (linked but not used)
find_library(PLAPACK_LIB NAMES PLAPACK HINTS ${PETSC_LIB_DIR})
if(NOT PLAPACK_LIB)
  message(WARNING "Could not find PLAPACK using PETSC_ARCH_DIR.")
endif(NOT PLAPACK_LIB)

# PORD (linked but not used)
find_library(PORD_LIB NAMES pord HINTS ${PETSC_LIB_DIR})
if(NOT PORD_LIB)
  message(WARNING "Could not find PORD using PETSC_ARCH_DIR.")
endif(NOT PORD_LIB)

# Prometheus (linked but not used)
find_library(PROMETHEUS_MAIN_LIB NAMES prometheus HINTS ${PETSC_LIB_DIR})
find_library(PROMFEI_LIB NAMES promfei HINTS ${PETSC_LIB_DIR})
if(NOT PROMETHEUS_MAIN_LIB OR NOT PROMFEI_LIB)
  message(WARNING "Could not find Prometheus using PETSC_ARCH_DIR.")
endif(NOT PROMETHEUS_MAIN_LIB OR NOT PROMFEI_LIB)
set(PROMETHEUS_LIBS "${PROMFEI_LIB};${PROMETHEUS_MAIN_LIB}")

# ScaLAPACK
find_library(SCALAPACK_LIB NAMES scalapack HINTS ${PETSC_LIB_DIR})
if(NOT SCALAPACK_LIB)
  message(
    FATAL_ERROR
    "Could not find ScaLAPACK using PETSC_ARCH_DIR. Make sure PETSc built it."
  )
endif(NOT SCALAPACK_LIB)

# BLACS
find_library(BLACS_LIB NAMES blacs HINTS ${PETSC_LIB_DIR})
if(NOT BLACS_LIB)
  message(
    FATAL_ERROR
    "Could not find BLACS using PETSC_ARCH_DIR. Make sure PETSc build it."
  )
endif(NOT BLACS_LIB)

# SPAI (linked but not used)
find_library(SPAI_LIB NAMES spai HINTS ${PETSC_LIB_DIR})
if(NOT SPAI_LIB)
  message(WARNING "Could not find SPAI using PETSC_ARCH_DIR.")
endif(NOT SPAI_LIB)

# SPOOLES (linked but not used)
find_library(SPOOLES_LIB NAMES spooles HINTS ${PETSC_LIB_DIR})
if(NOT SPOOLES_LIB)
  message(WARNING "Could not find SPOOLES using PETSC_ARCH_DIR.")
endif(NOT SPOOLES_LIB)

# SuperLU (linked but not used)
find_library(SUPERLU_LIB NAMES superlu_4.0 superlu HINTS ${PETSC_LIB_DIR})
if(NOT SUPERLU_LIB)
  message(WARNING "Could not find SuperLU using PETSC_ARCH_DIR.")
endif(NOT SUPERLU_LIB)

# SuperLUDist
find_library(
  SUPERLU_DIST_LIB 
  NAMES superlu_dist_2.3 superlu_dist 
  HINTS ${PETSC_LIB_DIR}
  )
if(NOT SUPERLU_DIST_LIB)
  message(WARNING "Could not find SuperLUDist using PETSC_ARCH_DIR.")
endif(NOT SUPERLU_DIST_LIB)

# Triangle (linked but not used)
find_library(TRIANGLE_LIB NAMES triangle HINTS ${PETSC_LIB_DIR})
if(NOT TRIANGLE_LIB)
  message(WARNING "Could not find Triangle using PETSC_ARCH_DIR.")
endif(NOT TRIANGLE_LIB)

# UMFPACK (linked but not used)
find_library(UMFPACK_LIB NAMES umfpack HINTS ${PETSC_LIB_DIR})
if(NOT UMFPACK_LIB)
  message(WARNING "Could not find UMFPACK using PETSC_ARCH_DIR.")
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
list(APPEND EXTERNAL_LIBS "${SCALAPACK_LIB};${BLACS_LIB}")
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

