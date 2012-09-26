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
#ifndef PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP
#define PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP 1

namespace psp {

template<typename F> 
void LocalCompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, typename Base<F>::type tolA, typename Base<F>::type tolB );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void LocalCompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, typename Base<F>::type tolA, typename Base<F>::type tolB )
{
#ifndef RELEASE
    PushCallStack("LocalCompressedBlockLDL");
#endif
    const int numLocalNodes = info.localNodes.size();
    for( int s=0; s<numLocalNodes; ++s )
    {
        cliq::LocalSymmNodeInfo& node = info.localNodes[s];
        const int updateSize = node.lowerStruct.size();
        LocalCompressedFront<F>& front = L.localFronts[s];
        Matrix<F>& frontL = front.frontL;
        Matrix<F>& frontBR = front.work;
        frontBR.Empty();
#ifndef RELEASE
        if( frontL.Height() != node.size+updateSize ||
            frontL.Width() != node.size )
            throw std::logic_error("Front was not the proper size");
#endif

        // Add updates from children (if they exist)
        elem::Zeros( updateSize, updateSize, frontBR );
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
                const int jFront = node.leftChildRelIndices[jChild];
                for( int iChild=0; iChild<leftUpdateSize; ++iChild )
                {
                    const int iFront = node.leftChildRelIndices[iChild];
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
                const int jFront = node.rightChildRelIndices[jChild];
                for( int iChild=0; iChild<rightUpdateSize; ++iChild )
                {
                    const int iFront = node.rightChildRelIndices[iChild];
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
        if( L.isHermitian )
            cliq::LocalFrontBlockLDL( ADJOINT, frontL, frontBR );
        else
            cliq::LocalFrontBlockLDL( TRANSPOSE, frontL, frontBR );

        // Separately compress the A and B portions of the front
        LocalFrontCompression( front, depth, false, useQR, tolA, tolB );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP
