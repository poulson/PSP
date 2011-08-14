# The serial Intel compilers
set(CMAKE_C_COMPILER       /opt/apps/intel/11.1/bin/intel64/icc)
set(CMAKE_CXX_COMPILER     /opt/apps/intel/11.1/bin/intel64/icpc)

# The MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   /opt/apps/intel11_1/mvapich2/1.6/bin/mpicc)
set(MPI_CXX_COMPILER /opt/apps/intel11_1/mvapich2/1.6/bin/mpicxx)

set(MATH_LIBS "-L/opt/apps/intel/11.1/mkl/lib/em64t -lmkl_intel_lp64 -lmkl_sequential -lmkl_core /opt/apps/intel/11.1/lib/intel64/libifcore.a /opt/apps/intel/11.1/lib/intel64/libsvml.a -lm")

