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
#ifndef PSP_LOW_RANK_MATRIX_HPP
#define PSP_LOW_RANK_MATRIX_HPP 1

#include "psp/dense_matrix.hpp"

namespace psp {

// NOTE: We have two different factorized forms since Hermitian-transposes are
//       more natural, but plain transposes will almost certainly be faster for
//       complex symmetric problems.

// A basic low-rank matrix representation that is used for the blocks with
// sufficiently separated sources and targets. 
//
// U and V will always be assumed to be of general type 
// (they should be non-square except in pathological cases).
template<typename Scalar,bool Conjugated>
struct LowRankMatrix
{
    // If Conjugated == true, then A = U V^H, otherwise, A = U V^T. 
    DenseMatrix<Scalar> U;
    DenseMatrix<Scalar> V;

    int Height() const { return U.Height(); }
    int Width() const { return V.Height(); }
    int Rank() const { return U.Width(); }
};

} // namespace psp

#endif // PSP_LOW_RANK_MATRIX_HPP
