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
#ifndef PSP_SPLIT_LOW_RANK_MATRIX_HPP
#define PSP_SPLIT_LOW_RANK_MATRIX_HPP 1

#include "psp/classes/dense_matrix.hpp"

namespace psp {

// For parallelizing the application of U V* (where V* = V^T or V^H) when 
// two processes are involved. One process owns U and the other owns V. Then
// the process owning V can form z := V* x, which is only r entries, then 
// communicate this result to the process owning U so that it may form 
// U y = U (V* x).
template<typename Scalar,bool Conjugated>
struct SplitLowRankMatrix
{
    int height, width, rank;
    bool ownSourceSide;
    int localOffset;
    int partner;

    DenseMatrix<Scalar> D;

    // Storage for V^[T/H] x. This should be computed by the process owning
    // the source side and then communicated to the process owning the target 
    // side.
    mutable Vector<Scalar> z;
};

} // namespace psp

#endif // PSP_SPLIT_LOW_RANK_MATRIX_HPP
