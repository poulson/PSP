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
#ifndef PSP_SPLIT_DENSE_MATRIX_HPP
#define PSP_SPLIT_DENSE_MATRIX_HPP 1

#include "psp/classes/dense_matrix.hpp"
#include "psp/classes/vector.hpp"

namespace psp {

// A wrapper for a dense matrix that is split between two processes. 
// For now, the source process will always own the matrix. This may need
// to change for triangular/symmetric matrices.
template<typename Scalar>
struct SplitDenseMatrix
{
    int height, width;
    bool ownSourceSide;
    int localOffset;
    int partner;

    DenseMatrix<Scalar> D;

    // Temporary storage for the product D x
    mutable Vector<Scalar> z;
};

} // namespace psp

#endif // PSP_SPLIT_DENSE_MATRIX_HPP
