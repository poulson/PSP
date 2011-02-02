#!/bin/sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx.mpich \
      -DCMAKE_C_COMPILER=/usr/bin/mpicc.mpich \
      -DCMAKE_Fortran_COMPILER=/usr/bin/mpif90.mpich \
      -DFortran_FLAGS="-O3 -I/usr/lib/mpich/include" \
      -DINTEL_ROOT=/opt/intel/Compiler/11.1/072 ..
make

