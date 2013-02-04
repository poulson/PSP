#
# Copyright (C) 2011-2013 Jack Poulson, Lexing Ying, 
# The University of Texas at Austin, and Stanford University
#
# This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
# GNU General Public License, which can be found in the LICENSE file in the 
# root directory, or at <http://www.gnu.org/licenses/>.
#
cmake_minimum_required(VERSION 2.8.10) # needed for Mercurial downloads
project(PSP)

set(PSP_VERSION_MAJOR 0)
set(PSP_VERSION_MINOR 1)

option(PSP_TESTS "Build a collection of test executables" OFF)
option(PSP_EXAMPLES "Build one or more example executables" OFF)
option(PRINT_RITZ_VALUES "Print Ritz values before restarting GMRES" OFF)

set(CLIQ_REV 371)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/external/clique-r${CLIQ_REV})
  find_package(Hg)
  if(HG_FOUND)
    execute_process(COMMAND
      ${HG_EXECUTABLE} clone https://bitbucket.org/poulson/clique
      ${PROJECT_SOURCE_DIR}/external/clique-r${CLIQ_REV})
  else()
    message(FATAL_ERROR
      "Could not find Mercurial, which is needed for checking out Clique")
  endif()
endif()
add_subdirectory(external/clique-r${CLIQ_REV})
include_directories(
  "${PROJECT_BINARY_DIR}/external/clique-r${CLIQ_REV}/include")
if(HAVE_PARMETIS)
  include_directories(
    "${PROJECT_SOURCE_DIR}/external/clique-r${CLIQ_REV}/external/parmetis/include")
  include_directories(
    "${PROJECT_SOURCE_DIR}/external/clique-r${CLIQ_REV}/external/parmetis/metis/include")
endif()
include_directories(
    "${PROJECT_BINARY_DIR}/external/clique-r${CLIQ_REV}/external/elemental-r${ELEM_REV}/include")
include_directories(${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

# Detect whether or not we can use mkdir to create directories.
# This should ideally exist on all POSIX machines.
include(CheckFunctionExists)
check_function_exists(mkdir HAVE_MKDIR)

# Create the PSP configuration header
configure_file( 
  ${PROJECT_SOURCE_DIR}/cmake/config.h.cmake
  ${PROJECT_BINARY_DIR}/include/psp/config.h)
install(FILES ${PROJECT_BINARY_DIR}/include/psp/config.h 
        DESTINATION include/psp)

# Create a file which can be included in Makefile's.
# This is meant to be analogous to PETSc's 'conf/petscvariables' file
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/pspvariables.cmake
  ${PROJECT_BINARY_DIR}/conf/pspvariables @ONLY)
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
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/${HEADER}
            ${PROJECT_BINARY_DIR}/${HEADER}
    DEPENDS "${PROJECT_SOURCE_DIR}/${HEADER}")
  list(APPEND PREPARED_HEADERS ${PROJECT_BINARY_DIR}/${HEADER})

  get_filename_component(HEADER_PATH ${HEADER} PATH)
  install(FILES ${PROJECT_BINARY_DIR}/${HEADER} DESTINATION ${HEADER_PATH})
endforeach()
add_custom_target(prepare_psp_headers DEPENDS ${PREPARED_HEADERS})
add_dependencies(psp prepare_psp_headers)

# Make sure the PSP headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

# Build the test drivers if necessary
if(PSP_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)
  set(TESTS Interpolate UnitCube)

  # Build the tests
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/tests")
  foreach(TEST ${TESTS})
    add_executable(tests-${TEST} ${TEST_DIR}/${TEST}.cpp)
    target_link_libraries(tests-${TEST} psp)
    set_target_properties(tests-${TEST} PROPERTIES
      OUTPUT_NAME ${TEST} RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    if(MPI_LINK_FLAGS)
      set_target_properties(tests-${TEST} PROPERTIES 
        LINK_FLAGS ${MPI_LINK_FLAGS})
    endif()
    install(TARGETS tests-${TEST} DESTINATION bin/tests)
  endforeach()
endif()

# Build the example drivers if necessary
if(PSP_EXAMPLES)
  set(EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
  set(EXAMPLES Overthrust Overthrust-sideways)

  # Build the examples
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/examples")
  foreach(EXAMPLE ${EXAMPLES})
    add_executable(examples-${EXAMPLE} ${EXAMPLE_DIR}/${EXAMPLE}.cpp)
    target_link_libraries(examples-${EXAMPLE} psp)
    set_target_properties(examples-${EXAMPLE} PROPERTIES
      OUTPUT_NAME ${EXAMPLE} RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    if(MPI_LINK_FLAGS)
      set_target_properties(examples-${EXAMPLE} PROPERTIES
        LINK_FLAGS ${MPI_LINK_FLAGS})
    endif()
    install(TARGETS examples-${EXAMPLE} DESTINATION bin/examples)
  endforeach()
endif()

# If PSP is a subproject, then pass some variables to the parent
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(./cmake/psp_sub.cmake)
endif()
