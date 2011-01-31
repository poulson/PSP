#!/bin/sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=`which mpicxx` \
      -DCMAKE_C_COMPILER=`which mpicc` \
      -DCMAKE_Fortran_COMPILER=`which mpif90` \
      -DINTEL_ROOT=/opt/intel/Compiler/11.1/072 ..
make

