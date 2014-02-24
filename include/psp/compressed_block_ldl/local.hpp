/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP
#define PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP

namespace psp {

template<typename F> 
void LocalCompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, Base<F> tolA, Base<F> tolB );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void LocalCompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, Base<F> tolA, Base<F> tolB )
{
    DEBUG_ONLY(CallStackEntry entry("LocalCompressedBlockLDL"))
    const int numLocalNodes = info.localNodes.size();
    for( int s=0; s<numLocalNodes; ++s )
    {
        cliq::SymmNodeInfo& node = info.localNodes[s];
        const int updateSize = node.lowerStruct.size();
        CompressedFront<F>& front = L.localFronts[s];
        Matrix<F>& frontL = front.frontL;
        Matrix<F>& frontBR = front.work;
        frontBR.Empty();
        DEBUG_ONLY(
            if( frontL.Height() != node.size+updateSize ||
                frontL.Width() != node.size )
                LogicError("Front was not the proper size");
        )

        // Add updates from children (if they exist)
        elem::Zeros( frontBR, updateSize, updateSize );
        const int numChildren = node.children.size();
        if( numChildren == 2 )
        {
            const int leftIndex = node.children[0];
            const int rightIndex = node.children[1];
            Matrix<F>& leftUpdate = L.localFronts[leftIndex].work;
            Matrix<F>& rightUpdate = L.localFronts[rightIndex].work;

            // Add the left child's update matrix
            const int leftUpdateSize = leftUpdate.Height();
            for( int jChild=0; jChild<leftUpdateSize; ++jChild )
            {
                const int jFront = node.leftRelInds[jChild];
                for( int iChild=0; iChild<leftUpdateSize; ++iChild )
                {
                    const int iFront = node.leftRelInds[iChild];
                    const F value = leftUpdate.Get(iChild,jChild);
                    if( jFront < node.size )
                        frontL.Update( iFront, jFront, value );
                    else if( iFront >= node.size )
                        frontBR.Update
                        ( iFront-node.size, jFront-node.size, value );
                }
            }
            leftUpdate.Empty();

            // Add the right child's update matrix
            const int rightUpdateSize = rightUpdate.Height();
            for( int jChild=0; jChild<rightUpdateSize; ++jChild )
            {
                const int jFront = node.rightRelInds[jChild];
                for( int iChild=0; iChild<rightUpdateSize; ++iChild )
                {
                    const int iFront = node.rightRelInds[iChild];
                    const F value = rightUpdate.Get(iChild,jChild);
                    if( jFront < node.size )
                        frontL.Update( iFront, jFront, value );
                    else if( iFront >= node.size )
                        frontBR.Update
                        ( iFront-node.size, jFront-node.size, value );
                }
            }
            rightUpdate.Empty();
        }

        // Call the custom partial block LDL
        cliq::FrontBlockLDL( frontL, frontBR, L.isHermitian );

        // Separately compress the A and B portions of the front
        CompressFront( front, depth, false, useQR, tolA, tolB );
    }
}

} // namespace psp

#endif // PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP
