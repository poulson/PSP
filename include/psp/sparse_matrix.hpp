/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef PSP_SPARSE_MATRIX_HPP
#define PSP_SPARSE_MATRIX_HPP 1

namespace psp {

// A simple Compressed Sparse Row (CSR) data structure
struct SparseMatrix
{
    int m; // height of matrix
    int n; // width of matrix
    std::vector<PetscScalar> nonzeros;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;

    // TODO: Routines for outputting in Matlab and PETSc formats?
};

} // namespace psp

#endif // PSP_SPARSE_MATRIX_HPP
