/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_COMPRESSED_BLOCK_LDL_HPP
#define PSP_COMPRESSED_BLOCK_LDL_HPP

namespace psp {

// All fronts of L are required to be initialized to the expansions of the 
// original sparse matrix before calling the following factorizations.

// Default to the QR algorithm since I have run into several problems with 
// the Divide and Conquer version, even in LAPACK 3.3.1.
template<typename F>
void CompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR=false, Base<F> tolA=1e-3, Base<F> tolB=1e-3 );

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

#include "./compressed_block_ldl/local_front.hpp"
#include "./compressed_block_ldl/dist_front.hpp"

#include "./compressed_block_ldl/local.hpp"
#include "./compressed_block_ldl/dist.hpp"

namespace psp {

template<typename F>
inline void CompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, Base<F> tolA, Base<F> tolB )
{
    DEBUG_ONLY(CallStackEntry entry("CompressedBlockLDL"))
    LocalCompressedBlockLDL( info, L, depth, useQR, tolA, tolB );
    DistCompressedBlockLDL( info, L, depth, useQR, tolA, tolB );
}

} // namespace psp

#endif // ifndef PSP_COMPRESSED_BLOCK_LDL_HPP
