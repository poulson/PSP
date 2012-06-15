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
#ifndef PSP_DIST_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_DIST_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

namespace psp {

template<typename F> 
void DistCompressedBlockLowerForwardSolve
( const cliq::symbolic::SymmFact& S,
  const CompressedFrontTree<F>& L, 
        Matrix<F>& localX );

template<typename F>
void DistCompressedBlockLowerBackwardSolve
( Orientation orientation, 
  const cliq::symbolic::SymmFact& S,
  const CompressedFrontTree<F>& L,
        Matrix<F>& localX );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void DistCompressedBlockLowerForwardSolve
( const cliq::symbolic::SymmFact& S,
  const CompressedFrontTree<F>& L, 
        Matrix<F>& localX )
{
    using namespace cliq::symbolic;
#ifndef RELEASE
    PushCallStack("DistCompressedBlockLowerForwardSolve");
#endif
    const int numDistSupernodes = S.dist.supernodes.size();
    const int width = localX.Width();
    if( width == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Copy the information from the local portion into the distributed leaf
    const LocalCompressedFront<F>& localRootFront = L.local.fronts.back();
    const DistCompressedFront<F>& distLeafFront = L.dist.fronts[0];
    const Grid& leafGrid = *distLeafFront.grid;
    distLeafFront.work1d.LockedView
    ( localRootFront.work.Height(), localRootFront.work.Width(), 0,
      localRootFront.work.LockedBuffer(), localRootFront.work.LDim(), 
      leafGrid );
    
    // Perform the distributed portion of the forward solve
    for( int s=1; s<numDistSupernodes; ++s )
    {
        const DistSymmFactSupernode& childSN = S.dist.supernodes[s-1];
        const DistSymmFactSupernode& sn = S.dist.supernodes[s];
        const DistCompressedFront<F>& childFront = L.dist.fronts[s-1];
        const DistCompressedFront<F>& front = L.dist.fronts[s];
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
        W.ResizeTo( frontHeight, width );
        DistMatrix<F,VC,STAR> WT(grid), WB(grid);
        elem::PartitionDown
        ( W, WT,
             WB, sn.size );

        // Pull in the relevant information from the RHS
        Matrix<F> localXT;
        localXT.View( localX, sn.localOffset1d, 0, sn.localSize1d, width );
        WT.LocalMatrix() = localXT;
        elem::MakeZeros( WB );

        // Pack our child's update
        DistMatrix<F,VC,STAR>& childW = childFront.work1d;
        const int updateSize = childW.Height()-childSN.size;
        DistMatrix<F,VC,STAR> childUpdate;
        childUpdate.LockedView( childW, childSN.size, 0, updateSize, width );
        int sendBufferSize = 0;
        std::vector<int> sendCounts(commSize), sendDispls(commSize);
        for( int proc=0; proc<commSize; ++proc )
        {
            const int sendSize = sn.numChildSolveSendIndices[proc]*width;
            sendCounts[proc] = sendSize;
            sendDispls[proc] = sendBufferSize;
            sendBufferSize += sendSize;
        }
        std::vector<F> sendBuffer( sendBufferSize );

        const bool isLeftChild = ( commRank < commSize/2 );
        const std::vector<int>& myChildRelIndices = 
            ( isLeftChild ? sn.leftChildRelIndices
                          : sn.rightChildRelIndices );
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
            L.local.fronts.back().work.Empty();

        // Set up the receive buffer
        int recvBufferSize = 0;
        std::vector<int> recvCounts(commSize), recvDispls(commSize);
        for( int proc=0; proc<commSize; ++proc )
        {
            const int recvSize = sn.childSolveRecvIndices[proc].size()*width;
            recvCounts[proc] = recvSize;
            recvDispls[proc] = recvBufferSize;
            recvBufferSize += recvSize;
        }
        std::vector<F> recvBuffer( recvBufferSize );
#ifndef RELEASE
        // Verify the send and recv counts match
        std::vector<int> actualRecvCounts(commSize);
        mpi::AllToAll
        ( &sendCounts[0],       1,
          &actualRecvCounts[0], 1, comm );
        for( int proc=0; proc<commSize; ++proc )
        {
            if( actualRecvCounts[proc] != recvCounts[proc] )
            {
                std::ostringstream msg;
                msg << "Expected recv count of " << recvCounts[proc]
                    << " but recv'd " << actualRecvCounts[proc]
                    << " from process " << proc << " for supernode "
                    << s << "\n";
                throw std::logic_error( msg.str().c_str() );
            }
        }
        actualRecvCounts.clear();
#endif

        // AllToAll to send and receive the child updates
#ifdef USE_CUSTOM_ALLTOALLV_FOR_SOLVE
        int numSends=0,numRecvs=0;
        for( unsigned proc=0; proc<commSize; ++proc )
        {
            if( sendCounts[proc] != 0 )
                ++numSends;
            if( recvCounts[proc] != 0 )
                ++numRecvs;
        }
        std::vector<mpi::Status> statuses(numSends+numRecvs);
        std::vector<mpi::Request> requests(numSends+numRecvs);
        int rCount=0;
        for( unsigned proc=0; proc<commSize; ++proc )
        {
            int count = recvCounts[proc];
            int displ = recvDispls[proc];
            if( count != 0 )
                mpi::IRecv
                ( &recvBuffer[displ], count, proc, 0, comm, requests[rCount++] );
        }
        for( unsigned proc=0; proc<commSize; ++proc )
        {
            int count = sendCounts[proc];
            int displ = sendDispls[proc];
            if( count != 0 )
                mpi::ISend
                ( &sendBuffer[displ], count, proc, 0, comm, requests[rCount++] );
        }
        mpi::WaitAll( numSends+numRecvs, &requests[0], &statuses[0] );
        statuses.clear();
        requests.clear();
#else
        mpi::AllToAll
        ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], comm );
#endif
        sendBuffer.clear();
        sendCounts.clear();
        sendDispls.clear();

        // Unpack the child updates (with an Axpy)
        for( int proc=0; proc<commSize; ++proc )
        {
            const F* recvValues = &recvBuffer[recvDispls[proc]];
            const std::deque<int>& recvIndices = sn.childSolveRecvIndices[proc];
            for( unsigned k=0; k<recvIndices.size(); ++k )
            {
                const int iFrontLocal = recvIndices[k];
                const F* recvRow = &recvValues[k*width];
                F* WRow = W.LocalBuffer( iFrontLocal, 0 );
                const int WLDim = W.LocalLDim();
                for( int j=0; j<width; ++j )
                    WRow[j*WLDim] += recvRow[j];
            }
        }
        recvBuffer.clear();
        recvCounts.clear();
        recvDispls.clear();

        // Now that the RHS is set up, perform this supernode's solve
        DistFrontCompressedBlockLowerForwardSolve( front, W );

        // Store the supernode portion of the result
        localXT = WT.LocalMatrix();
    }
    L.local.fronts.back().work.Empty();
    L.dist.fronts.back().work1d.Empty();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void DistCompressedBlockLowerBackwardSolve
( Orientation orientation, 
  const cliq::symbolic::SymmFact& S,
  const CompressedFrontTree<F>& L,
        Matrix<F>& localX )
{
    using namespace cliq::symbolic;
#ifndef RELEASE
    PushCallStack("DistCompressedBlockLowerBackwardSolve");
#endif
    const int numDistSupernodes = S.dist.supernodes.size();
    const int width = localX.Width();
    if( width == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Directly operate on the root separator's portion of the right-hand sides
    const DistSymmFactSupernode& rootSN = S.dist.supernodes.back();
    const DistCompressedFront<F>& rootFront = L.dist.fronts.back();
    const Grid& rootGrid = *rootFront.grid;
    rootFront.work1d.View
    ( rootSN.size, width, 0,
      localX.Buffer(rootSN.localOffset1d,0), localX.LDim(), rootGrid );
    DistFrontCompressedBlockLowerBackwardSolve
    ( orientation, rootFront, rootFront.work1d );

    std::vector<int>::const_iterator it;
    for( int s=numDistSupernodes-2; s>=0; --s )
    {
        const DistSymmFactSupernode& parentSN = S.dist.supernodes[s+1];
        const DistSymmFactSupernode& sn = S.dist.supernodes[s];
        const DistCompressedFront<F>& parentFront = L.dist.fronts[s+1];
        const DistCompressedFront<F>& front = L.dist.fronts[s];
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
        W.ResizeTo( frontHeight, width );
        DistMatrix<F,VC,STAR> WT(grid), WB(grid);
        elem::PartitionDown
        ( W, WT,
             WB, sn.size );

        // Pull in the relevant information from the RHS
        Matrix<F> localXT;
        localXT.View( localX, sn.localOffset1d, 0, sn.localSize1d, width );
        WT.LocalMatrix() = localXT;

        //
        // Set the bottom from the parent
        //

        // Pack the updates using the recv approach from the forward solve
        int sendBufferSize = 0;
        std::vector<int> sendCounts(parentCommSize), sendDispls(parentCommSize);
        for( int proc=0; proc<parentCommSize; ++proc )
        {
            const int sendSize = 
                parentSN.childSolveRecvIndices[proc].size()*width;
            sendCounts[proc] = sendSize;
            sendDispls[proc] = sendBufferSize;
            sendBufferSize += sendSize;
        }
        std::vector<F> sendBuffer( sendBufferSize );

        DistMatrix<F,VC,STAR>& parentWork = parentFront.work1d;
        for( int proc=0; proc<parentCommSize; ++proc )
        {
            F* sendValues = &sendBuffer[sendDispls[proc]];
            const std::deque<int>& recvIndices = 
                parentSN.childSolveRecvIndices[proc];
            for( unsigned k=0; k<recvIndices.size(); ++k )
            {
                const int iFrontLocal = recvIndices[k];
                F* sendRow = &sendValues[k*width];
                const F* workRow = 
                    parentWork.LockedLocalBuffer( iFrontLocal, 0 );
                const int workLDim = parentWork.LocalLDim();
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
                parentSN.numChildSolveSendIndices[proc]*width;
            recvCounts[proc] = recvSize;
            recvDispls[proc] = recvBufferSize;
            recvBufferSize += recvSize;
        }
        std::vector<F> recvBuffer( recvBufferSize );
#ifndef RELEASE
        // Verify the send and recv counts match
        std::vector<int> actualRecvCounts(parentCommSize);
        mpi::AllToAll
        ( &sendCounts[0],       1,
          &actualRecvCounts[0], 1, parentComm );
        for( int proc=0; proc<parentCommSize; ++proc )
        {
            if( actualRecvCounts[proc] != recvCounts[proc] )
            {
                std::ostringstream msg;
                msg << "Expected recv count of " << recvCounts[proc]
                    << " but recv'd " << actualRecvCounts[proc]
                    << " from process " << proc << " for supernode "
                    << s << "\n";
                throw std::logic_error( msg.str().c_str() );
            }
        }
        actualRecvCounts.clear();
#endif

        // AllToAll to send and recv parent updates
#ifdef USE_CUSTOM_ALLTOALLV_FOR_SOLVE
        int numSends=0,numRecvs=0;
        for( unsigned proc=0; proc<parentCommSize; ++proc )
        {  
            if( sendCounts[proc] != 0 )
                ++numSends;
            if( recvCounts[proc] != 0 )
                ++numRecvs;
        }
        std::vector<mpi::Status> statuses(numSends+numRecvs);
        std::vector<mpi::Request> requests(numSends+numRecvs);
        int rCount=0; 
        for( unsigned proc=0; proc<parentCommSize; ++proc )
        {    
            int count = recvCounts[proc];
            int displ = recvDispls[proc];
            if( count != 0 )
                mpi::IRecv
                ( &recvBuffer[displ], count, proc, 0, parentComm, 
                  requests[rCount++] );
        }
        for( unsigned proc=0; proc<parentCommSize; ++proc )
        {
            int count = sendCounts[proc];
            int displ = sendDispls[proc];
            if( count != 0 )
                mpi::ISend
                ( &sendBuffer[displ], count, proc, 0, parentComm, 
                  requests[rCount++] );
        }
        mpi::WaitAll( numSends+numRecvs, &requests[0], &statuses[0] );
        statuses.clear();
        requests.clear();
#else
        mpi::AllToAll
        ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], parentComm );
#endif
        sendBuffer.clear();
        sendCounts.clear();
        sendDispls.clear();

        // Unpack the updates using the send approach from the forward solve
        const bool isLeftChild = ( parentCommRank < parentCommSize/2 );
        const std::vector<int>& myRelIndices = 
            ( isLeftChild ? parentSN.leftChildRelIndices
                          : parentSN.rightChildRelIndices );
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
        DistFrontCompressedBlockLowerBackwardSolve( orientation, front, W );

        // Store the supernode portion of the result
        localXT = WT.LocalMatrix();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
