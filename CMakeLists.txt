#
#  Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
#  of a sweeping preconditioner for 3d Helmholtz equations.
#
#  Copyright (C) 2010-2012 Jack Poulson <jack.poulson@gmail.com>
#  Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
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
cmake_minimum_required(VERSION 2.8.5)
project(PSP)

set(PSP_VERSION_MAJOR 0)
set(PSP_VERSION_MINOR 1)

option(BUILD_TESTS "Build a collection of test executables" OFF)

# TODO: Think of how to avoid this mess...
add_subdirectory(external/clique)
include_directories("${PROJECT_BINARY_DIR}/external/clique/include")
if(HAVE_PARMETIS)
  include_directories(
    "${PROJECT_SOURCE_DIR}/external/clique/external/parmetis/include"
  )
  include_directories(
    "${PROJECT_SOURCE_DIR}/external/clique/external/parmetis/metis/include"
  )
endif()
include_directories(
  "${PROJECT_BINARY_DIR}/external/clique/external/elemental/include")
include_directories(${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

# Detect whether or not we can use mkdir to create directories.
# This should ideally exist on all POSIX machines.
include(CheckFunctionExists)
check_function_exists(mkdir HAVE_MKDIR)

# Create the PSP configuration header
configure_file( 
  ${PROJECT_SOURCE_DIR}/cmake/config.h.cmake
  ${PROJECT_BINARY_DIR}/include/psp/config.h 
) 
install(FILES ${PROJECT_BINARY_DIR}/include/psp/config.h 
        DESTINATION include/psp)

# Create a file which can be included in Makefile's.
# This is meant to be analogous to PETSc's 'conf/petscvariables' file
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/pspvariables.cmake
  ${PROJECT_BINARY_DIR}/conf/pspvariables @ONLY
)
install(FILES ${PROJECT_BINARY_DIR}/conf/pspvariables
        DESTINATION conf)

# Grab all of the .c, .cpp, .h, and .hpp PSP files
file(GLOB_RECURSE PSP_CPP RELATIVE ${PROJECT_SOURCE_DIR} 
     "src/*.c" "src/*.cpp")
file(GLOB_RECURSE PSP_HEADERS RELATIVE ${PROJECT_SOURCE_DIR} 
     "include/*.h" "include/*.hpp")
set(PSP_SRC "${PSP_CPP};${PSP_HEADERS}")

# The main library
add_library(psp ${LIBRARY_TYPE} ${PSP_SRC})
target_link_libraries(psp clique)
set(MPI_LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
install(TARGETS psp DESTINATION lib)

# Define the header-file preparation rules
set(PREPARED_HEADERS)
foreach(HEADER ${PSP_HEADERS})
  add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/${HEADER}
            ${PROJECT_BINARY_DIR}/${HEADER}
    DEPENDS "${PROJECT_SOURCE_DIR}/${HEADER}"
  )
  list(APPEND PREPARED_HEADERS ${PROJECT_BINARY_DIR}/${HEADER})

  get_filename_component(HEADER_PATH ${HEADER} PATH)
  install(FILES ${PROJECT_BINARY_DIR}/${HEADER} DESTINATION ${HEADER_PATH})
endforeach()
add_custom_target(prepare_psp_headers DEPENDS ${PREPARED_HEADERS})
add_dependencies(psp prepare_psp_headers)

# Make sure the PSP headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

# Build the test drivers if necessary
if(BUILD_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)
  set(TESTS UnitCube)

  # Build the tests
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/tests")
  foreach(TEST ${TESTS})
    add_executable(tests-${TEST} ${TEST_DIR}/${TEST}.cpp)
    target_link_libraries(tests-${TEST} psp)
    set_target_properties(tests-${TEST} PROPERTIES
                          OUTPUT_NAME ${TEST}
                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    if(MPI_LINK_FLAGS)
      set_target_properties(tests-${TEST} PROPERTIES
                            LINK_FLAGS ${MPI_LINK_FLAGS})
    endif()
    install(TARGETS tests-${TEST} DESTINATION bin/tests)
  endforeach()
endif()

################################################################################
# Uncomment if including PSP as a subproject in another build system           #
################################################################################
#set(USE_CUSTOM_ALLTOALLV_FOR_FACT ${USE_CUSTOM_ALLTOALLV_FOR_FACT} PARENT_SCOPE)
#set(USE_CUSTOM_ALLTOALLV_FOR_MULT ${USE_CUSTOM_ALLTOALLV_FOR_MULT} PARENT_SCOPE)
#set(USE_CUSTOM_ALLTOALLV_FOR_SOLVE ${USE_CUSTOM_ALLTOALLV_FOR_SOLVE} PARENT_SCOPE)
#
#set(LIBRARY_TYPE ${LIBRARY_TYPE} PARENT_SCOPE)
#set(MPI_C_COMPILER ${MPI_C_COMPILER} PARENT_SCOPE)
#set(MPI_C_INCLUDE_PATH ${MPI_C_INCLUDE_PATH} PARENT_SCOPE)
#set(MPI_C_COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS} PARENT_SCOPE)
#set(MPI_C_LINK_FLAGS ${MPI_C_LINK_FLAGS} PARENT_SCOPE)
#set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
#set(MPI_CXX_COMPILER ${MPI_CXX_COMPILER} PARENT_SCOPE)
#set(MPI_CXX_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH} PARENT_SCOPE)
#set(MPI_CXX_COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS} PARENT_SCOPE)
#set(MPI_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS} PARENT_SCOPE)
#set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES} PARENT_SCOPE)
#set(MATH_LIBS ${MATH_LIBS} PARENT_SCOPE)
#set(RESTRICT ${RESTRICT} PARENT_SCOPE)
#set(RELEASE ${RELEASE} PARENT_SCOPE)
#set(BLAS_POST ${BLAS_POST} PARENT_SCOPE)
#set(LAPACK_POST ${LAPACK_POST} PARENT_SCOPE)
#set(HAVE_F90_INTERFACE ${HAVE_F90_INTERFACE} PARENT_SCOPE)
#set(WITHOUT_PMRRR ${WITHOUT_PMRRR} PARENT_SCOPE)
#set(AVOID_COMPLEX_MPI ${AVOID_COMPLEX_MPI} PARENT_SCOPE)
#set(HAVE_REDUCE_SCATTER_BLOCK ${HAVE_REDUCE_SCATTER_BLOCK} PARENT_SCOPE)
#set(REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE ${REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE} PARENT_SCOPE)
#set(USE_BYTE_ALLGATHERS ${USE_BYTE_ALLGATHERS} PARENT_SCOPE)
