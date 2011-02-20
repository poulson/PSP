set(CMAKE_SYSTEM_NAME BlueGeneP-static)

set(CMAKE_C_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlc_r)
set(CMAKE_CXX_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlcxx_r)

set(CXX_FLAGS "-g -O4")

# The remainder of the file is for linking 
# BLAS/LAPACK/BLACS/ScaLAPACK/METIS/PORD/MUMPS/PETSc

set(PETSC_DIR "/soft/apps/petsc-3.1-p2")
set(PETSC_ARCH_DIR "${PETSC_DIR}/bgp-ibm-cxx-opt")
set(PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
set(PETSC_ARCH_INCLUDE_DIR "${PETSC_ARCH_DIR}/include")
set(PETSC_LIB "${PETSC_ARCH_DIR}/lib/libpetsc.a")

set(CMUMPS_LIB "${PETSC_ARCH_DIR}/lib/libcmumps.a")
set(DMUMPS_LIB "${PETSC_ARCH_DIR}/lib/libdmumps.a")
set(SMUMPS_LIB "${PETSC_ARCH_DIR}/lib/libsmumps.a")
set(ZMUMPS_LIB "${PETSC_ARCH_DIR}/lib/libzmumps.a")
set(MUMPS_COMMON_LIB "${PETSC_ARCH_DIR}/lib/libmumps_common.a")
set(MUMPS_LIBS 
  "${CMUMPS_LIB};${DMUMPS_LIB};${SMUMPS_LIB};${ZMUMPS_LIB};${MUMPS_COMMON_LIB}")

set(SCALAPACK_LIB "${PETSC_ARCH_DIR}/lib/libscalapack.a")
set(BLACS_LIBS "${PETSC_ARCH_DIR}/lib/libblacs.a")

set(METIS_LIB "${PETSC_ARCH_DIR}/lib/libmetis.a")
set(PARMETIS_LIB "${PETSC_ARCH_DIR}/lib/libparmetis.a")

# Linked but not used...
set(AMD_LIB "${PETSC_ARCH_DIR}/lib/libamd.a")
set(CHACO_LIB "${PETSC_ARCH_DIR}/lib/libchaco.a")
set(HYPRE_LIB "${PETSC_ARCH_DIR}/lib/libHYPRE.a")
set(PLAPACK_LIB "${PETSC_ARCH_DIR}/lib/libPLAPACK.a")
set(PORD_LIB "${PETSC_ARCH_DIR}/lib/libpord.a")
set(PROMETHEUS_LIBS 
  "${PETSC_ARCH_DIR}/lib/libpromfei.a;${PETSC_ARCH_DIR}/lib/libprometheus.a")
set(SPAI_LIB "${PETSC_ARCH_DIR}/lib/libspai.a")
set(SPOOLES_LIB "${PETSC_ARCH_DIR}/lib/libspooles.a")
set(SUPERLU_LIB "${PETSC_ARCH_DIR}/lib/libsuperlu_4.0.a")
set(SUPERLU_DIST_LIB "${PETSC_ARCH_DIR}/lib/libsuperlu_dist_2.3.a")
set(TRIANGLE_LIB "${PETSC_ARCH_DIR}/lib/libtriangle.a")
set(UMFPACK_LIB "${PETSC_ARCH_DIR}/lib/libumfpack.a")

set(LAPACK "-L/soft/apps/LAPACK -llapack_bgp")
set(ESSL_DIR "/soft/apps/ESSL-4.3.1-1")
set(IBMCMP_DIR "/soft/apps/ibmcmp-dec2010")
set(XLF_DIR "${IBMCMP_DIR}/xlf/bg/11.1/bglib")
set(XLSMP_DIR "${IBMCMP_DIR}/xlsmp/bg/1.7/bglib")
set(ESSL "-L${ESSL_DIR}/lib -lesslbg")
set(XLF "-L${XLF_DIR} -lxlfmath -lxlf90_r")
set(XLOMP_SER "-L${XLSMP_DIR} -lxlomp_ser")

set(MATH_LIBS "${LAPACK};${ESSL};${XLF};${XLOMP_SER}")
