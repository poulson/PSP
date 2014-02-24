/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP

namespace psp {

template<typename F>
void CompressedBlockLowerSolve
( Orientation orientation, 
  const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
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
  const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& localX )
{
    DEBUG_ONLY(CallStackEntry cse("CompressedBlockLowerSolve"))
    if( orientation == NORMAL )
    {
        LocalCompressedBlockLowerForwardSolve( info, L, localX );
        DistCompressedBlockLowerForwardSolve( info, L, localX );
    }
    else
    {
        const bool conjugate = ( orientation == ADJOINT );
        DistCompressedBlockLowerBackwardSolve( info, L, localX, conjugate );
        LocalCompressedBlockLowerBackwardSolve( info, L, localX, conjugate );
    }
}

} // namespace psp

#endif // ifndef PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 
