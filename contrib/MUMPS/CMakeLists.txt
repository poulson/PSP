# We only need to make the double-precision complex MUMPS with ParMetis
file(GLOB_RECURSE MUMPS_FCOMMON 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "mumps_*.F")
list(APPEND MUMPS_FCOMMON "src/tools_common_mod.F")
file(GLOB_RECURSE MUMPS_CCOMMON 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "mumps_*.c")
file(GLOB_RECURSE MUMPS_HCOMMON 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "mumps_*.h")
file(GLOB_RECURSE MUMPS_FZ 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "z*.F")
file(GLOB_RECURSE MUMPS_HZ
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "z*.h")
set(MUMPS_SRC 
   "${MUMPS_FCOMMON};${MUMPS_CCOMMON};${MUMPS_HCOMMON};${MUMPS_FZ};${MUMPS_HZ}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# You may need to add/remove:
#  -Dintel_
#  -DALLOW_NON_INIT
#  -DAdd_
if(Fortran_APPENDS_UNDERSCORE)
  set(FUNDERSCORE_DEF "-DAdd_")
else(Fortran_APPENDS_UNDERSCORE)
  set(FUNDERSCORE_DEF "")
endif(Fortran_APPENDS_UNDERSCORE)
set(MUMPS_C_FLAGS "${FUNDERSCORE_DEF} -Dparmetis -DMUMPS_ARITH=MUMPS_ARITH_z")
set(MUMPS_Fortran_FLAGS "-Dparmetis -DMUMPS_ARITH=MUMPS_ARITH_z")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MUMPS_C_FLAGS}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MUMPS_Fortran_FLAGS}")

add_library(zmumps STATIC ${MUMPS_SRC})
target_link_libraries(zmumps parmetis metis)

