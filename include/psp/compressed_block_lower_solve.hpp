/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and
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
#ifndef PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

namespace psp {

template<typename F>
void CompressedBlockLowerSolve
( Orientation orientation, 
  const cliq::symbolic::SymmFact& S,
  const CompressedFrontTree<F>& L,
        Matrix<F>& localX );

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

#include "./compressed_block_lower_solve/local_front.hpp"
#include "./compressed_block_lower_solve/dist_front.hpp"

#include "./compressed_block_lower_solve/local.hpp"
#include "./compressed_block_lower_solve/dist.hpp"

namespace psp {

template<typename F>
inline void CompressedBlockLowerSolve
( Orientation orientation,
  const cliq::symbolic::SymmFact& S,
  const CompressedFrontTree<F>& L,
        Matrix<F>& localX )
{
#ifndef RELEASE
    PushCallStack("CompressedBlockLowerSolve");
#endif
    if( orientation == NORMAL )
    {
        LocalCompressedBlockLowerForwardSolve( S, L, localX );
        DistCompressedBlockLowerForwardSolve( S, L, localX );
    }
    else
    {
        DistCompressedBlockLowerBackwardSolve( orientation, S, L, localX );
        LocalCompressedBlockLowerBackwardSolve( orientation, S, L, localX );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 
