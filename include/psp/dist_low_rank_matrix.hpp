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
#ifndef PSP_DIST_LOW_RANK_MATRIX_HPP
#define PSP_DIST_LOW_RANK_MATRIX_HPP 1

#include "psp/dense_matrix.hpp"

namespace psp {

// For parallelizing the application of U V* (where V* = V^T or V^H) when 
// there are two teams involved. The target team divides U into contiguous
// sets of rows and the source team divides V into contiguous sets of rows.
// U V* can be applied to a vector x by forming V* x, which is only r entries,
// communicating the small set of entries to the target team, and then having
// each member of the target team locally update their portion of z := U V* x.
template<typename Scalar,bool Conjugated>
struct DistLowRankMatrix
{
    int height, width, rank;

    MPI_Comm myTeam;
    MPI_Comm otherTeam;

    bool onSourceTeam;
    DenseMatrix<Scalar> D;

    // Storage for V^[T/H] x. This should be computed by the process owning
    // the source side and then communicated to the process owning the target 
    // side.
    mutable Vector<Scalar> y;
};

} // namespace psp

#endif // PSP_DIST_LOW_RANK_MATRIX_HPP
