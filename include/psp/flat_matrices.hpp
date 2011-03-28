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
#ifndef PSP_FLAT_MATRICES_HPP
#define PSP_FLAT_MATRICES_HPP 1

#include <complex>
#include <vector>

namespace psp {

// A basic dense matrix representation that is used for storing blocks 
// whose sources and targets are too close to represent as low rank
template<typename Scalar>
struct DenseMatrix
{
    bool symmetric;
    int m; // height of matrix
    int n; // width of matrix
    int ldim; // leading dimension of matrix
    std::vector<Scalar> buffer; // column-major buffer
};

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
struct SparseMatrix
{
    bool symmetric;
    int m; // height of matrix
    int n; // width of matrix
    std::vector<Scalar> nonzeros;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;
    // TODO: Routines for outputting in Matlab and PETSc formats?
};

// A basic low-rank matrix representation that is used for the blocks with
// sufficiently separated sources and targets
template<typename Scalar>
struct FactorMatrix
{
    int m; // height of matrix
    int n; // width of matrix
    int r; // rank of matrix
    // A = U V^H
    std::vector<Scalar> U; // buffer for m x r left set of vectors
    std::vector<Scalar> V; // buffer for n x r right set of vectors
};

} // namespace psp

#endif // PSP_FLAT_MATRICES_HPP
