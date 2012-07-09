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
#ifndef PSP_COMPRESSED_BLOCK_LDL_HPP
#define PSP_COMPRESSED_BLOCK_LDL_HPP 1

namespace psp {

// All fronts of L are required to be initialized to the expansions of the 
// original sparse matrix before calling the following factorizations.

// Default to the QR algorithm since I have run into several problems with 
// the Divide and Conquer version, even in LAPACK 3.3.1.
template<typename F>
void CompressedBlockLDL
( Orientation orientation, 
  cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR=true );

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
( Orientation orientation, 
  cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR )
{
#ifndef RELEASE
    PushCallStack("CompressedBlockLDL");
#endif
    LocalCompressedBlockLDL( orientation, info, L, depth, useQR );
    DistCompressedBlockLDL( orientation, info, L, depth, useQR );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif /* PSP_COMPRESSED_BLOCK_LDL_HPP */
