/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_LOCAL_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_LOCAL_COMPRESSED_BLOCK_LOWER_SOLVE_HPP

namespace psp {

template<typename F> 
void LocalCompressedBlockLowerForwardSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& X );

template<typename F> 
void LocalCompressedBlockLowerBackwardSolve
( const cliq::DistSymmInfo& info, 
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& X, bool conjugate=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void LocalCompressedBlockLowerForwardSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& X )
{
    DEBUG_ONLY(CallStackEntry entry("LocalCompressedBlockLowerForwardSolve"))
    const int numLocalNodes = info.localNodes.size();
    const int width = X.Width();
    for( int s=0; s<numLocalNodes; ++s )
    {
        const cliq::SymmNodeInfo& node = info.localNodes[s];
        const CompressedFront<F>& front = L.localFronts[s];
        const int sT = front.sT;
        const int sB = front.sB;
        const int depth = front.depth;
        const int frontHeight = (sT+sB)*depth;
        Matrix<F>& W = front.work;

        // Set up a workspace
        W.Resize( frontHeight, width );
        Matrix<F> WT, WB;
        elem::PartitionDown( W, WT, WB, node.size );

        // Pull in the relevant information from the RHS
        Matrix<F> XT;
        View( XT, X, node.myOff, 0, node.size, width );
        WT = XT;
        elem::MakeZeros( WB );

        // Update using the children (if they exist)
        const int numChildren = node.children.size();
        if( numChildren == 2 )
        {
            const int leftIndex = node.children[0];
            const int rightIndex = node.children[1];
            Matrix<F>& leftWork = L.localFronts[leftIndex].work;
            Matrix<F>& rightWork = L.localFronts[rightIndex].work;
            const int leftNodeSize = info.localNodes[leftIndex].size;
            const int rightNodeSize = info.localNodes[rightIndex].size;
            const int leftUpdateSize = leftWork.Height()-leftNodeSize;
            const int rightUpdateSize = rightWork.Height()-rightNodeSize;

            // Add the left child's update onto ours
            Matrix<F> leftUpdate;
            LockedView
            ( leftUpdate, leftWork, leftNodeSize, 0, leftUpdateSize, width );
            for( int iChild=0; iChild<leftUpdateSize; ++iChild )
            {
                const int iFront = node.leftRelInds[iChild]; 
                for( int j=0; j<width; ++j )
                    W.Update( iFront, j, leftUpdate.Get(iChild,j) );
            }
            leftWork.Empty();

            // Add the right child's update onto ours
            Matrix<F> rightUpdate;
            LockedView
            ( rightUpdate, 
              rightWork, rightNodeSize, 0, rightUpdateSize, width );
            for( int iChild=0; iChild<rightUpdateSize; ++iChild )
            {
                const int iFront = node.rightRelInds[iChild];
                for( int j=0; j<width; ++j )
                    W.Update( iFront, j, rightUpdate.Get(iChild,j) );
            }
            rightWork.Empty();
        }
        // else numChildren == 0

        // Solve against this front
        FrontCompressedBlockLowerForwardSolve( front, W );

        // Store the node portion of the result
        XT = WT;
    }
}

template<typename F> 
inline void LocalCompressedBlockLowerBackwardSolve
( const cliq::DistSymmInfo& info, 
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry entry("LocalCompressedBlockLowerBackwardSolve"))
    const int numLocalNodes = info.localNodes.size();
    const int width = X.Width();

    for( int s=numLocalNodes-2; s>=0; --s )
    {
        const cliq::SymmNodeInfo& node = info.localNodes[s];
        const CompressedFront<F>& front = L.localFronts[s];
        const int sT = front.sT;
        const int sB = front.sB;
        const int depth = front.depth;
        const int frontHeight = (sT+sB)*depth;
        Matrix<F>& W = front.work;

        // Set up a workspace
        W.Resize( frontHeight, width );
        Matrix<F> WT, WB;
        elem::PartitionDown( W, WT, WB, node.size );

        // Pull in the relevant information from the RHS
        Matrix<F> XT;
        View( XT, X, node.myOff, 0, node.size, width );
        WT = XT;

        // Update using the parent
        const int parent = node.parent;
        Matrix<F>& parentWork = L.localFronts[parent].work;
        const cliq::SymmNodeInfo& parentNode = info.localNodes[parent];
        const int currentUpdateSize = WB.Height();
        const std::vector<int>& parentRelIndices = 
          ( node.onLeft ? parentNode.leftRelInds : parentNode.rightRelInds );
        for( int iCurrent=0; iCurrent<currentUpdateSize; ++iCurrent )
        {
            const int iParent = parentRelIndices[iCurrent];
            for( int j=0; j<width; ++j )
                WB.Set( iCurrent, j, parentWork.Get(iParent,j) );
        }

        // The left child is numbered lower than the right child, so 
        // we can safely free the parent's work if we are the left child
        if( node.onLeft )
        {
            parentWork.Empty();
            if( parent == numLocalNodes-1 )
                L.distFronts[0].work1d.Empty();
        }

        // Solve against this front
        FrontCompressedBlockLowerBackwardSolve( front, W, conjugate );

        // Store the node portion of the result
        XT = WT;
    }

    // Ensure that all of the temporary buffers are freed (this is overkill)
    L.distFronts[0].work1d.Empty();
    for( int s=0; s<numLocalNodes; ++s )
        L.localFronts[s].work.Empty();
}

} // namespace psp

#endif // ifndef PSP_LOCAL_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
