set(CMAKE_SYSTEM_NAME BlueGeneP-static)

set(CMAKE_C_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlc_r)
set(CMAKE_CXX_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlcxx_r)

set(CXX_FLAGS "-g -O4")

set(PETSC_DIR "/soft/apps/petsc-3.1-p2")
set(PETSC_ARCH_DIR "${PETSC_DIR}/bgp-ibm-cxx-opt")

set(LAPACK "-L/soft/apps/LAPACK -llapack_bgp")
set(ESSL_DIR "/soft/apps/ESSL-4.3.1-1")
set(IBMCMP_DIR "/soft/apps/ibmcmp-dec2010")
set(XLF_DIR "${IBMCMP_DIR}/xlf/bg/11.1/bglib")
set(XLSMP_DIR "${IBMCMP_DIR}/xlsmp/bg/1.7/bglib")
set(ESSL "-L${ESSL_DIR}/lib -lesslbg")
set(XLF "-L${XLF_DIR} -lxlfmath -lxlf90_r")
set(XLOMP_SER "-L${XLSMP_DIR} -lxlomp_ser")

set(MATH_LIBS "${LAPACK};${ESSL};${XLF};${XLOMP_SER}")
