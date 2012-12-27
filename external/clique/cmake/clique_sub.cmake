set(USE_CUSTOM_ALLTOALLV ${USE_CUSTOM_ALLTOALLV} PARENT_SCOPE)
set(BARRIER_IN_ALLTOALLV ${BARRIER_IN_ALLTOALLV} PARENT_SCOPE)
set(HAVE_PARMETIS ${HAVE_PARMETIS} PARENT_SCOPE)

set(LIBRARY_TYPE ${LIBRARY_TYPE} PARENT_SCOPE)
string(TOUPPER ${CMAKE_BUILD_TYPE} UPPER_BUILD_TYPE)
set(CMAKE_CXX_FLAGS_${UPPER_BUILD_TYPE} ${CMAKE_CXX_FLAGS_${UPPER_BUILD_TYPE}} PARENT_SCOPE)
set(MPI_C_COMPILER ${MPI_C_COMPILER} PARENT_SCOPE)
set(MPI_C_INCLUDE_PATH ${MPI_C_INCLUDE_PATH} PARENT_SCOPE)
set(MPI_C_COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS} PARENT_SCOPE)
set(MPI_C_LINK_FLAGS ${MPI_C_LINK_FLAGS} PARENT_SCOPE)
set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
set(MPI_CXX_COMPILER ${MPI_CXX_COMPILER} PARENT_SCOPE)
set(MPI_CXX_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH} PARENT_SCOPE)
set(MPI_CXX_COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS} PARENT_SCOPE)
set(MPI_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS} PARENT_SCOPE)
set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES} PARENT_SCOPE)
set(MPI_LINK_FLAGS ${MPI_LINK_FLAGS} PARENT_SCOPE)
set(MATH_LIBS ${MATH_LIBS} PARENT_SCOPE)
set(RESTRICT ${RESTRICT} PARENT_SCOPE)
set(RELEASE ${RELEASE} PARENT_SCOPE)
set(BLAS_POST ${BLAS_POST} PARENT_SCOPE)
set(LAPACK_POST ${LAPACK_POST} PARENT_SCOPE)
set(HAVE_OPENMP ${HAVE_OPENMP} PARENT_SCOPE)
set(HAVE_F90_INTERFACE ${HAVE_F90_INTERFACE} PARENT_SCOPE)
set(WITHOUT_PMRRR ${WITHOUT_PMRRR} PARENT_SCOPE)
set(AVOID_COMPLEX_MPI ${AVOID_COMPLEX_MPI} PARENT_SCOPE)
set(HAVE_REDUCE_SCATTER_BLOCK ${HAVE_REDUCE_SCATTER_BLOCK} PARENT_SCOPE)
set(REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE ${REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE} PARENT_SCOPE)
set(USE_BYTE_ALLGATHERS ${USE_BYTE_ALLGATHERS} PARENT_SCOPE)
