/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_COMPRESSED_BLOCK_LDL_SOLVE_HPP
#define PSP_COMPRESSED_BLOCK_LDL_SOLVE_HPP

namespace psp {

template<typename F>
void CompressedBlockLDLSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& localX );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline void CompressedBlockLDLSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& localX )
{
    DEBUG_ONLY(CallStackEntry entry("CompressedBlockLDLSolve"))
    // Solve against block diagonal factor, L D
    CompressedBlockLowerSolve( NORMAL, info, L, localX );

    // Solve against the (conjugate-)transpose of the block unit diagonal L
    if( L.isHermitian )
        CompressedBlockLowerSolve( ADJOINT, info, L, localX );
    else
        CompressedBlockLowerSolve( TRANSPOSE, info, L, localX );
}

} // namespace psp

#endif // ifndef PSP_COMPRESSED_BLOCK_LDL_SOLVE_HPP
