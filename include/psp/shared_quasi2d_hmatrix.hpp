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
#ifndef PSP_SHARED_QUASI2D_HMATRIX_HPP
#define PSP_SHARED_QUASI2D_HMATRIX_HPP 1

#include "psp/quasi2d_hmatrix.hpp"
#include "psp/shared_low_rank_matrix.hpp"
#include "psp/shared_dense_matrix.hpp"

namespace psp {

template<typename Scalar,bool Conjugated>
class SharedQuasi2dHMatrix
{
private:
    enum ShellType 
    { NODE, NODE_SYMMETRIC, SHARED_LOW_RANK, SHARED_DENSE, DENSE };

public:

    // TODO
};

} // namespace psp

#endif // PSP_SHARED_QUASI2D_HMATRIX_HPP
