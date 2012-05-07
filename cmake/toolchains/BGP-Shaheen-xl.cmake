set(CMAKE_SYSTEM_NAME BlueGeneP-static)

# The serial XL compilers
set(BGXLC_BASE /usr)
set(BGXLF_BASE /usr)
set(CMAKE_C_COMPILER       ${BGXLC_BASE}/bin/bgxlc_r)
set(CMAKE_CXX_COMPILER     ${BGXLC_BASE}/bin/bgxlC_r)
set(CMAKE_Fortran_COMPILER ${BGXLF_BASE}/bin/bgxlf_r)

# The MPI wrappers for the XL C and C++ compilers
set(BGMPI_BASE /bgsys/drivers/ppcfloor/comm)
set(MPI_C_COMPILER   ${BGMPI_BASE}/bin/mpixlc_r)
set(MPI_CXX_COMPILER ${BGMPI_BASE}/bin/mpixlcxx_r)

set(CXX_PURE_DEBUG_FLAGS "-g")
set(CXX_PURE_RELEASE_FLAGS "-O3")
set(CXX_HYBRID_DEBUG_FLAGS "-g")
set(CXX_HYBRID_RELEASE_FLAGS "-O3")

set(CMAKE_THREAD_LIBS_INIT "-qthreaded")
set(OpenMP_CXX_FLAGS "-qsmp=omp:noauto -qthreaded")

set(ESSL_BASE "/opt/share/ibmmath")
set(IBMCMP_BASE "/opt/ibmcmp")
set(XLF_BASE "${IBMCMP_BASE}/xlf/bg/11.1/bglib")
set(XLSMP_BASE "${IBMCMP_BASE}/xlsmp/bg/1.7/bglib")
set(BGP_LAPACK "-L/opt/share/lapack/3.4.0/bgp-pdc/lib -llapack")
set(PURE_ESSL "-L${ESSL_BASE}/lib -lesslbg")
set(THREADED_ESSL "-L${ESSL_BASE}/lib -lesslsmpbg")
set(XLF_LIBS "-L${XLF_BASE} -lxlfmath -lxlf90_r")
set(XLOMP_SER "-L${XLSMP_BASE} -lxlomp_ser")
set(XLSMP "-L${XLSMP_BASE} -lxlsmp")

if(CMAKE_BUILD_TYPE MATCHES PureDebug OR
   CMAKE_BUILD_TYPE MATCHES PureRelease)
  set(MATH_LIBS "${BGP_LAPACK};${PURE_ESSL};${XLF_LIBS};${XLOMP_SER}")
else(CMAKE_BUILD_TYPE MATCHES PureDebug OR 
     CMAKE_BUILD_TYPE MATCHES PureRelease)
  set(MATH_LIBS "${BGP_LAPACK};${THREADED_ESSL};${XLF_LIBS};${XLSMP}")
endif(CMAKE_BUILD_TYPE MATCHES PureDebug OR 
      CMAKE_BUILD_TYPE MATCHES PureRelease)

# Make sure we can find the ESSL headers
set(ESSL_INC "${ESSL_BASE}/include")
include_directories(${ESSL_INC})
