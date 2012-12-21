/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#ifndef PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

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
#ifndef RELEASE
    PushCallStack("CompressedBlockLowerSolve");
#endif
    if( orientation == NORMAL )
    {
        LocalCompressedBlockLowerForwardSolve( info, L, localX );
        DistCompressedBlockLowerForwardSolve( info, L, localX );
    }
    else
    {
        DistCompressedBlockLowerBackwardSolve( orientation, info, L, localX );
        LocalCompressedBlockLowerBackwardSolve( orientation, info, L, localX );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 
