/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_DIST_COMPRESSED_BLOCK_LDL_HPP 
#define PSP_DIST_COMPRESSED_BLOCK_LDL_HPP

namespace psp {

template<typename F> 
void DistCompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, Base<F> tolA, Base<F> tolB );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void DistCompressedBlockLDL
( cliq::DistSymmInfo& info, DistCompressedFrontTree<F>& L, int depth,
  bool useQR, Base<F> tolA, Base<F> tolB )
{
    DEBUG_ONLY(CallStackEntry entry("DistCompressedBlockLDL"))
    // The bottom front is already compressed, so just view the relevant data
    CompressedFront<F>& topLocalFront = L.localFronts.back();
    DistCompressedFront<F>& bottomDistFront = L.distFronts[0];
    const Grid& bottomGrid = *info.distNodes[0].grid;
    bottomDistFront.grid = &bottomGrid;
    bottomDistFront.depth = topLocalFront.depth;
    bottomDistFront.sT = topLocalFront.sT;
    bottomDistFront.sB = topLocalFront.sB;
    bottomDistFront.frontL.Empty();
    bottomDistFront.AGreens.clear();
    bottomDistFront.BGreens.clear();
    bottomDistFront.ACoefficients.clear();
    bottomDistFront.BCoefficients.clear();
    bottomDistFront.work2d.Empty();
    bottomDistFront.work2d.LockedAttach
    ( topLocalFront.work.Height(), topLocalFront.work.Width(), 0, 0,
      topLocalFront.work.LockedBuffer(), topLocalFront.work.LDim(),
      bottomGrid );

    // Perform the distributed portion of the factorization
    const unsigned numDistNodes = info.distNodes.size();
    for( unsigned s=1; s<numDistNodes; ++s )
    {
        const cliq::DistSymmNodeInfo& childNode = info.distNodes[s-1];
        const cliq::DistSymmNodeInfo& node = info.distNodes[s];
        const int updateSize = node.lowerStruct.size();
        DistCompressedFront<F>& childFront = L.distFronts[s-1];
        DistCompressedFront<F>& front = L.distFronts[s];
        front.work2d.Empty();

        const bool computeFactRecvIndices = 
            ( node.factorMeta.childRecvInds.size() == 0 );

        // Grab this front's grid information
        const Grid& grid = front.frontL.Grid();
        mpi::Comm comm = grid.VCComm();
        const unsigned commRank = mpi::CommRank( comm );
        const unsigned commSize = mpi::CommSize( comm );
        const unsigned gridHeight = grid.Height();
        const unsigned gridWidth = grid.Width();

        // Grab the child's grid information
        const DistMatrix<F>& childUpdate = childFront.work2d;
        const Grid& childGrid = childUpdate.Grid();
        const unsigned childGridHeight = childGrid.Height();
        const unsigned childGridWidth = childGrid.Width();

        // Pack our child's update
        const bool isLeftChild = ( commRank < commSize/2 );
        std::vector<int> sendCounts(commSize), sendDispls(commSize);
        int sendBufferSize = 0;
        for( unsigned proc=0; proc<commSize; ++proc )
        {
            const int sendSize = node.factorMeta.numChildSendInds[proc];
            sendCounts[proc] = sendSize;
            sendDispls[proc] = sendBufferSize;
            sendBufferSize += sendSize;
        }
        std::vector<F> sendBuffer( sendBufferSize );

        const std::vector<int>& myChildRelIndices = 
            ( isLeftChild ? node.leftRelInds : node.rightRelInds );
        const int updateColShift = childUpdate.ColShift();
        const int updateRowShift = childUpdate.RowShift();
        const int updateLocalHeight = childUpdate.LocalHeight();
        const int updateLocalWidth = childUpdate.LocalWidth();
        std::vector<int> packOffsets = sendDispls;
        for( int jChildLocal=0; jChildLocal<updateLocalWidth; ++jChildLocal )
        {
            const int jChild = updateRowShift + jChildLocal*childGridWidth;
            const int destGridCol = myChildRelIndices[jChild] % gridWidth;

            int localColShift;
            if( updateColShift > jChild )
                localColShift = 0;
            else if( (jChild-updateColShift) % childGridHeight == 0 )
                localColShift = (jChild-updateColShift)/childGridHeight;
            else
                localColShift = (jChild-updateColShift)/childGridHeight + 1;
            for( int iChildLocal=localColShift; 
                     iChildLocal<updateLocalHeight; ++iChildLocal )
            {
                const int iChild = updateColShift + iChildLocal*childGridHeight;
                const int destGridRow = myChildRelIndices[iChild] % gridHeight;

                const int destRank = destGridRow + destGridCol*gridHeight;
                sendBuffer[packOffsets[destRank]++] = 
                    childUpdate.GetLocal(iChildLocal,jChildLocal);
            }
        }
        DEBUG_ONLY(
            for( unsigned proc=0; proc<commSize; ++proc )
            {
                if( packOffsets[proc]-sendDispls[proc] != 
                    node.factorMeta.numChildSendInds[proc] )
                    LogicError("Error in packing stage");
            }
        )
        packOffsets.clear();
        childFront.work2d.Empty();
        if( s == 1 )
            topLocalFront.work.Empty();

        // Set up the recv buffer for the AllToAll
        if( computeFactRecvIndices )
            ComputeFactRecvInds( node, childNode );
        std::vector<int> recvCounts(commSize), recvDispls(commSize);
        int recvBufferSize=0;
        for( unsigned proc=0; proc<commSize; ++proc )
        {
            const int recvSize = node.factorMeta.childRecvInds[proc].size()/2;
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

        // Unpack the child udpates (with an Axpy)
        front.work2d.SetGrid( front.frontL.Grid() );
        front.work2d.Align( node.size % gridHeight, node.size % gridWidth );
        elem::Zeros( front.work2d, updateSize, updateSize );
        const int leftLocalWidth = front.frontL.LocalWidth();
        const int topLocalHeight = 
            Length( node.size, grid.MCRank(), gridHeight );
        for( unsigned proc=0; proc<commSize; ++proc )
        {
            const F* recvValues = &recvBuffer[recvDispls[proc]];
            const std::vector<int>& recvIndices = 
                node.factorMeta.childRecvInds[proc];
            const int numRecvIndexPairs = recvIndices.size()/2;
            for( int k=0; k<numRecvIndexPairs; ++k )
            {
                const int iFrontLocal = recvIndices[2*k+0];
                const int jFrontLocal = recvIndices[2*k+1];
                const F value = recvValues[k];
                if( jFrontLocal < leftLocalWidth )
                    front.frontL.UpdateLocal( iFrontLocal, jFrontLocal, value );
                else
                    front.work2d.UpdateLocal
                    ( iFrontLocal-topLocalHeight, 
                      jFrontLocal-leftLocalWidth, value );
            }
        }
        recvBuffer.clear();
        recvCounts.clear();
        recvDispls.clear();
        if( computeFactRecvIndices )
            node.factorMeta.childRecvInds.clear();

        // Now that the frontal matrix is set up, perform the factorization
        cliq::FrontBlockLDL( front.frontL, front.work2d, L.isHermitian );

        // Separately compress the A and B blocks 
        CompressFront( front, depth, useQR, tolA, tolB );
    }
    L.localFronts.back().work.Empty();
    L.distFronts.back().work2d.Empty();
}

} // namespace psp

#endif // PSP_DIST_COMPRESSED_BLOCK_LDL_HPP
