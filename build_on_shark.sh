#!/bin/sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx.openmpi \
      -DCMAKE_C_COMPILER=/usr/bin/mpicc.openmpi \
      -DCMAKE_Fortran_COMPILER=/usr/bin/mpif90.openmpi \
      -DINTEL_ROOT=/opt/intel/Compiler/11.1/072 ..
make

