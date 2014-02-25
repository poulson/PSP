/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_DIST_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_DIST_COMPRESSED_BLOCK_LOWER_SOLVE_HPP

namespace psp {

template<typename F> 
void DistCompressedBlockLowerForwardSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L, 
        Matrix<F>& localX );

template<typename F>
void DistCompressedBlockLowerBackwardSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& localX, bool conjugate=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void DistCompressedBlockLowerForwardSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L, 
        Matrix<F>& localX )
{
    DEBUG_ONLY(CallStackEntry entry("DistCompressedBlockLowerForwardSolve"))
    const int numDistNodes = info.distNodes.size();
    const int width = localX.Width();

    // Copy the information from the local portion into the distributed leaf
    const CompressedFront<F>& localRootFront = L.localFronts.back();
    const DistCompressedFront<F>& distLeafFront = L.distFronts[0];
    const Grid& leafGrid = *distLeafFront.grid;
    distLeafFront.work1d.LockedAttach
    ( localRootFront.work.Height(), localRootFront.work.Width(), 0, 0,
      localRootFront.work.LockedBuffer(), localRootFront.work.LDim(), 
      leafGrid );
    
    // Perform the distributed portion of the forward solve
    for( int s=1; s<numDistNodes; ++s )
    {
        const cliq::DistSymmNodeInfo& childNode = info.distNodes[s-1];
        const cliq::DistSymmNodeInfo& node = info.distNodes[s];
        const DistCompressedFront<F>& childFront = L.distFronts[s-1];
        const DistCompressedFront<F>& front = L.distFronts[s];
        const Grid& childGrid = *childFront.grid;
        const Grid& grid = *front.grid;
        mpi::Comm comm = grid.VCComm();
        mpi::Comm childComm = childGrid.VCComm();
        const int commSize = mpi::CommSize( comm );
        const int commRank = mpi::CommRank( comm );
        const int childCommSize = mpi::CommSize( childComm );
        const int sT = front.sT;
        const int sB = front.sB;
        const int depth = front.depth;
        const int frontHeight = (sT+sB)*depth;

        // Set up a workspace
        DistMatrix<F,VC,STAR>& W = front.work1d;
        W.SetGrid( grid );
        W.Resize( frontHeight, width );
        DistMatrix<F,VC,STAR> WT(grid), WB(grid);
        elem::PartitionDown
        ( W, WT,
             WB, node.size );

        // Pull in the relevant information from the RHS
        // TODO: Switch to approach closer to what Clique now does
        Matrix<F> localXT;
        View
        ( localXT, localX, 
          node.multiVecMeta.localOff, 0, node.multiVecMeta.localSize, width );
        WT.Matrix() = localXT;
        elem::MakeZeros( WB );

        // Pack our child's update
        DistMatrix<F,VC,STAR>& childW = childFront.work1d;
        const int updateSize = childW.Height()-childNode.size;
        DistMatrix<F,VC,STAR> childUpdate;
        LockedView( childUpdate, childW, childNode.size, 0, updateSize, width );
        int sendBufferSize = 0;
        std::vector<int> sendCounts(commSize), sendDispls(commSize);
        for( int proc=0; proc<commSize; ++proc )
        {
            const int sendSize = node.multiVecMeta.numChildSendInds[proc]*width;
            sendCounts[proc] = sendSize;
            sendDispls[proc] = sendBufferSize;
            sendBufferSize += sendSize;
        }
        std::vector<F> sendBuffer( sendBufferSize );

        const bool isLeftChild = ( commRank < commSize/2 );
        const std::vector<int>& myChildRelIndices = 
            ( isLeftChild ? node.leftRelInds : node.rightRelInds );
        const int updateColShift = childUpdate.ColShift();
        const int updateLocalHeight = childUpdate.LocalHeight();
        std::vector<int> packOffsets = sendDispls;
        for( int iChildLocal=0; iChildLocal<updateLocalHeight; ++iChildLocal )
        {
            const int iChild = updateColShift + iChildLocal*childCommSize;
            const int destRank = myChildRelIndices[iChild] % commSize;
            F* packBuf = &sendBuffer[packOffsets[destRank]];
            for( int jChild=0; jChild<width; ++jChild )
                packBuf[jChild] = childUpdate.GetLocal(iChildLocal,jChild);
            packOffsets[destRank] += width;
        }
        packOffsets.clear();
        childW.Empty();
        if( s == 1 )
            L.localFronts.back().work.Empty();

        // Set up the receive buffer
        int recvBufferSize = 0;
        std::vector<int> recvCounts(commSize), recvDispls(commSize);
        for( int proc=0; proc<commSize; ++proc )
        {
            const int recvSize = 
                node.multiVecMeta.childRecvInds[proc].size()*width;
            recvCounts[proc] = recvSize;
            recvDispls[proc] = recvBufferSize;
            recvBufferSize += recvSize;
        }
        std::vector<F> recvBuffer( recvBufferSize );
#ifndef RELEASE
        cliq::VerifySendsAndRecvs( sendCounts, recvCounts, comm );
#endif

        // AllToAll to send and receive the child updates
        cliq::SparseAllToAll
        ( sendBuffer, sendCounts, sendDispls,
          recvBuffer, recvCounts, recvDispls, comm );
        sendBuffer.clear();
        sendCounts.clear();
        sendDispls.clear();

        // Unpack the child updates (with an Axpy)
        for( int proc=0; proc<commSize; ++proc )
        {
            const F* recvValues = &recvBuffer[recvDispls[proc]];
            const std::vector<int>& recvIndices = 
                node.multiVecMeta.childRecvInds[proc];
            for( unsigned k=0; k<recvIndices.size(); ++k )
            {
                const int iFrontLocal = recvIndices[k];
                const F* recvRow = &recvValues[k*width];
                F* WRow = W.Buffer( iFrontLocal, 0 );
                const int WLDim = W.LDim();
                for( int j=0; j<width; ++j )
                    WRow[j*WLDim] += recvRow[j];
            }
        }
        recvBuffer.clear();
        recvCounts.clear();
        recvDispls.clear();

        // Now that the RHS is set up, perform this node's solve
        FrontCompressedBlockLowerForwardSolve( front, W );

        // Store this node's portion of the result
        localXT = WT.Matrix();
    }
    L.localFronts.back().work.Empty();
    L.distFronts.back().work1d.Empty();
}

template<typename F>
inline void DistCompressedBlockLowerBackwardSolve
( const cliq::DistSymmInfo& info,
  const DistCompressedFrontTree<F>& L,
        Matrix<F>& localX, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry entry("DistCompressedBlockLowerBackwardSolve"))
    const int numDistNodes = info.distNodes.size();
    const int width = localX.Width();

    // Directly operate on the root separator's portion of the right-hand sides
    const cliq::DistSymmNodeInfo& rootNode = info.distNodes.back();
    const CompressedFront<F>& localRootFront = L.localFronts.back();
    if( numDistNodes == 1 )
    {
        localRootFront.work.Attach
        ( rootNode.size, width,
          localX.Buffer(rootNode.multiVecMeta.localOff,0), localX.LDim() );
        FrontCompressedBlockLowerBackwardSolve
        ( localRootFront, localRootFront.work, conjugate );
    }
    else
    {
        const DistCompressedFront<F>& rootFront = L.distFronts.back();
        const Grid& rootGrid = *rootFront.grid;
        rootFront.work1d.Attach
        ( rootNode.size, width, 0, 0,
          localX.Buffer(rootNode.multiVecMeta.localOff,0), localX.LDim(), 
          rootGrid );
        FrontCompressedBlockLowerBackwardSolve
        ( rootFront, rootFront.work1d, conjugate );
    }

    std::vector<int>::const_iterator it;
    for( int s=numDistNodes-2; s>=0; --s )
    {
        const cliq::DistSymmNodeInfo& parentNode = info.distNodes[s+1];
        const cliq::DistSymmNodeInfo& node = info.distNodes[s];
        const DistCompressedFront<F>& parentFront = L.distFronts[s+1];
        const DistCompressedFront<F>& front = L.distFronts[s];
        const Grid& grid = *front.grid;
        const Grid& parentGrid = *parentFront.grid;
        mpi::Comm comm = grid.VCComm(); 
        mpi::Comm parentComm = parentGrid.VCComm();
        const int commSize = mpi::CommSize( comm );
        const int parentCommSize = mpi::CommSize( parentComm );
        const int parentCommRank = mpi::CommRank( parentComm );
        const int sT = front.sT;
        const int sB = front.sB;
        const int depth = front.depth;
        const int frontHeight = (sT+sB)*depth;

        // Set up a workspace
        DistMatrix<F,VC,STAR>& W = front.work1d;
        W.SetGrid( grid );
        W.Resize( frontHeight, width );
        DistMatrix<F,VC,STAR> WT(grid), WB(grid);
        elem::PartitionDown
        ( W, WT,
             WB, node.size );

        // Pull in the relevant information from the RHS
        Matrix<F> localXT;
        View
        ( localXT, localX, 
          node.multiVecMeta.localOff, 0, node.multiVecMeta.localSize, width );
        WT.Matrix() = localXT;

        //
        // Set the bottom from the parent
        //

        // Pack the updates using the recv approach from the forward solve
        int sendBufferSize = 0;
        std::vector<int> sendCounts(parentCommSize), sendDispls(parentCommSize);
        for( int proc=0; proc<parentCommSize; ++proc )
        {
            const int sendSize = 
                parentNode.multiVecMeta.childRecvInds[proc].size()*width;
            sendCounts[proc] = sendSize;
            sendDispls[proc] = sendBufferSize;
            sendBufferSize += sendSize;
        }
        std::vector<F> sendBuffer( sendBufferSize );

        DistMatrix<F,VC,STAR>& parentWork = parentFront.work1d;
        for( int proc=0; proc<parentCommSize; ++proc )
        {
            F* sendValues = &sendBuffer[sendDispls[proc]];
            const std::vector<int>& recvIndices = 
                parentNode.multiVecMeta.childRecvInds[proc];
            for( unsigned k=0; k<recvIndices.size(); ++k )
            {
                const int iFrontLocal = recvIndices[k];
                F* sendRow = &sendValues[k*width];
                const F* workRow = parentWork.LockedBuffer( iFrontLocal, 0 );
                const int workLDim = parentWork.LDim();
                for( int j=0; j<width; ++j )
                    sendRow[j] = workRow[j*workLDim];
            }
        }
        parentWork.Empty();

        // Set up the receive buffer
        int recvBufferSize = 0;
        std::vector<int> recvCounts(parentCommSize), recvDispls(parentCommSize);
        for( int proc=0; proc<parentCommSize; ++proc )
        {
            const int recvSize = 
                parentNode.multiVecMeta.numChildSendInds[proc]*width;
            recvCounts[proc] = recvSize;
            recvDispls[proc] = recvBufferSize;
            recvBufferSize += recvSize;
        }
        std::vector<F> recvBuffer( recvBufferSize );
        DEBUG_ONLY(
            cliq::VerifySendsAndRecvs( sendCounts, recvCounts, parentComm )
        )

        // AllToAll to send and recv parent updates
        cliq::SparseAllToAll
        ( sendBuffer, sendCounts, sendDispls,
          recvBuffer, recvCounts, recvDispls, parentComm );
        sendBuffer.clear();
        sendCounts.clear();
        sendDispls.clear();

        // Unpack the updates using the send approach from the forward solve
        const bool isLeftChild = ( parentCommRank < parentCommSize/2 );
        const std::vector<int>& myRelIndices = 
            ( isLeftChild ? parentNode.leftRelInds : parentNode.rightRelInds );
        const int updateColShift = WB.ColShift();
        const int updateLocalHeight = WB.LocalHeight();
        for( int iUpdateLocal=0; 
                 iUpdateLocal<updateLocalHeight; ++iUpdateLocal )
        {
            const int iUpdate = updateColShift + iUpdateLocal*commSize;
            const int startRank = myRelIndices[iUpdate] % parentCommSize;
            const F* recvBuf = &recvBuffer[recvDispls[startRank]];
            for( int j=0; j<width; ++j )
                WB.SetLocal(iUpdateLocal,j,recvBuf[j]);
            recvDispls[startRank] += width;
        }
        recvBuffer.clear();
        recvCounts.clear();
        recvDispls.clear();

        // Use the distributed compressed data
        if( s > 0 )
            FrontCompressedBlockLowerBackwardSolve( front, W, conjugate );
        else
        {
            View( localRootFront.work, W.Matrix() );
            FrontCompressedBlockLowerBackwardSolve
            ( localRootFront, localRootFront.work, conjugate );
        }

        // Store this node's portion of the result
        localXT = WT.Matrix();
    }
}

} // namespace psp

#endif // PSP_DIST_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
