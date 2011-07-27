/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
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

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdates()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdates");
#endif
    const unsigned numTeamLevels = _teams->NumLevels();

    // Count the number of QRs we'll need to perform
    std::vector<int> numQRs(numTeamLevels,0);
    MultiplyHMatUpdatesCountQRs( numQRs );

    // Count the ranks of all of the low-rank updates that we will have to 
    // perform a QR on and also make space for their aggregations.
    int numTotalQRs=0;
    std::vector<int> XOffsets( numTeamLevels );
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        XOffsets[teamLevel] = numTotalQRs;        
        numTotalQRs += numQRs[teamLevel];
    }
    std::vector<Dense<Scalar>*> Xs( numTotalQRs );
    {
        std::vector<int> XOffsetsCopy = XOffsets;
        MultiplyHMatUpdatesLowRankCountAndResize( Xs, XOffsetsCopy, 0 );
    }

    // Carry the low-rank updates down from nodes into the low-rank and dense
    // blocks.
    MultiplyHMatUpdatesLowRankImport( 0 );

    // Allocate space for packed storage of the various components in our
    // distributed QR factorizations.
    numTotalQRs = 0;
    int numQRSteps=0, qrTotalSize=0, tauTotalSize=0, maxRank=0;
    std::vector<int> halfHeightOffsets(numTeamLevels+1),
                     qrOffsets(numTeamLevels+1), 
                     tauOffsets(numTeamLevels+1);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = _teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );
        const Dense<Scalar>* const* XLevel = &Xs[XOffsets[teamLevel]];

        halfHeightOffsets[teamLevel] = 2*numQRSteps;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        for( int k=0; k<numQRs[teamLevel]; ++k )
        {
            const int r = XLevel[k]->Width();
            maxRank = std::max(maxRank,r);

            qrTotalSize += log2TeamSize*(r*r+r);
            tauTotalSize += (log2TeamSize+1)*r;
        }
        numTotalQRs += numQRs[teamLevel];
        numQRSteps += log2TeamSize*numQRs[teamLevel];
    }
    qrOffsets[numTeamLevels] = qrTotalSize;
    tauOffsets[numTeamLevels] = tauTotalSize;

    std::vector<int> halfHeights( 2*numQRSteps );
    std::vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize );
    {
        // Make a disposable copy of the offset vectors and perform the local
        // QR portions of the (distributed) QRs
        std::vector<Scalar> qrWork( lapack::QRWorkSize(maxRank) );
        std::vector<int> tauOffsetsCopy = tauOffsets;
        MultiplyHMatUpdatesLocalQR( tauBuffer, tauOffsetsCopy, qrWork );

        // Perform the parallel portion of the TSQR algorithm
        MultiplyHMatParallelQR
        ( numQRs, Xs, XOffsets, halfHeights, halfHeightOffsets,
          qrBuffer, qrOffsets, tauBuffer, tauOffsets, qrWork );
    }

    // Count the number of entries of R and U that we need to exchange.
    // We also need to exchange U, V, and the dense update when performing
    // F += D, where the F is split.
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatUpdatesExchangeCount( sendSizes, recvSizes );

    // Compute the offsets
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
    }

    // Fill the send buffer
    std::vector<byte> sendBuffer( totalSendSize );
    {
        std::map<int,int> sendOffsetsCopy = sendOffsets;
        std::vector<int> halfHeightOffsetsCopy = halfHeightOffsets;
        std::vector<int> qrOffsetsCopy = qrOffsets;

        MultiplyHMatUpdatesExchangePack
        ( sendBuffer, sendOffsetsCopy, 
          halfHeights, halfHeightOffsetsCopy, qrBuffer, qrOffsetsCopy );
    }

    // Start the non-blocking sends
    MPI_Comm comm = _teams->Team( 0 );
    const int numSends = sendSizes.size();
    std::vector<MPI_Request> sendRequests( numSends );
    int offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Start the non-blocking recvs
    const int numRecvs = recvSizes.size();
    std::vector<MPI_Request> recvRequests( numRecvs );
    std::vector<byte> recvBuffer( totalRecvSize );
    offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    for( int i=0; i<numRecvs; ++i )
        mpi::Wait( recvRequests[i] );

    Dense<Scalar> X, Y, Z;
    std::vector<Real> singularValues, realWork;
    std::vector<Scalar> work;
    MultiplyHMatUpdatesExchangeFinalize
    ( recvBuffer, recvOffsets, halfHeights, halfHeightOffsets,
      qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
      X, Y, Z, singularValues, work, realWork );

    // Don't continue until we know the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::
MultiplyHMatUpdatesCountQRs( std::vector<int>& numQRs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesCountQRs");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const unsigned teamLevel = _teams->TeamLevel( _level );
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesCountQRs( numQRs );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
        if( _inTargetTeam && !_haveDenseUpdate )
            ++numQRs[teamLevel];
        if( _inSourceTeam && !_haveDenseUpdate )
            ++numQRs[teamLevel];
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::
MultiplyHMatUpdatesLowRankCountAndResize
( std::vector<Dense<Scalar>*>& Xs, std::vector<int>& XOffsets, int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLowRankCountAndResize");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
                rank += _UMap.CurrentEntry()->Width();
        }
        else if( _inSourceTeam )
        {
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
                rank += _VMap.CurrentEntry()->Width();
        }

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankCountAndResize
                ( Xs, XOffsets, rank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        // Compute the new rank
        if( _inTargetTeam )
        {
            // Add the F+=HH updates
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
                rank += _colXMap.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
                rank += _UMap.CurrentEntry()->Width();

            // Add the rank of the original low-rank matrix
            rank += DF.ULocal.Width();
        }
        else if( _inSourceTeam )
        {
            // Add the F+=HH updates
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
                rank += _rowXMap.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
                rank += _VMap.CurrentEntry()->Width();

            // Add the rank of the original low-rank matrix
            rank += DF.VLocal.Width();
        }

        // Store the rank and create the space
        const unsigned teamLevel = _teams->TeamLevel( _level );
        if( _inTargetTeam )
        {
            const int oldRank = DF.ULocal.Width();
            const int localHeight = DF.ULocal.Height();

            Dense<Scalar> ULocalCopy;
            hmat_tools::Copy( DF.ULocal, ULocalCopy );
            DF.ULocal.Resize( localHeight, rank, localHeight );
            std::memcpy
            ( DF.ULocal.Buffer(0,rank-oldRank), ULocalCopy.LockedBuffer(), 
              localHeight*oldRank*sizeof(Scalar) );

            Xs[XOffsets[teamLevel]++] = &DF.ULocal;
        }
        if( _inSourceTeam )
        {
            const int oldRank = DF.VLocal.Width();
            const int localWidth = DF.VLocal.Height();

            Dense<Scalar> VLocalCopy;
            hmat_tools::Copy( DF.VLocal, VLocalCopy );
            DF.VLocal.Resize( localWidth, rank, localWidth );
            std::memcpy
            ( DF.VLocal.Buffer(0,rank-oldRank), VLocalCopy.LockedBuffer(),
              localWidth*oldRank*sizeof(Scalar) );

            Xs[XOffsets[teamLevel]++] = &DF.VLocal;
        }
        DF.rank = rank;
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        const unsigned teamLevel = _teams->TeamLevel( _level );

        // Compute the new rank
        if( _inTargetTeam )
        {
            // Add the F+=HH updates
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
                rank += _colXMap.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
                rank += _UMap.CurrentEntry()->Width();

            // Add the original rank
            rank += SF.D.Width();
        }
        else 
        {
            // Add the F+=HH updates
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
                rank += _rowXMap.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
                rank += _VMap.CurrentEntry()->Width();

            // Add the original rank
            rank += SF.D.Width();
        }

        // Create the space and store the rank if we'll need to do a QR
        if( _inTargetTeam )
        {
            const int oldRank = SF.D.Width();
            const int height = SF.D.Height();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( SF.D, UCopy );
            SF.D.Resize( height, rank, height );
            SF.rank = rank;
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              height*oldRank*sizeof(Scalar) );

            if( !_haveDenseUpdate )
                Xs[XOffsets[teamLevel]++] = &SF.D;
        }
        else
        {
            const int oldRank = SF.D.Width();
            const int width = SF.D.Height();

            Dense<Scalar> VCopy;
            hmat_tools::Copy( SF.D, VCopy );
            SF.D.Resize( width, rank, width );
            SF.rank = rank;
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              width*oldRank*sizeof(Scalar) );

            if( !_haveDenseUpdate )
                Xs[XOffsets[teamLevel]++] = &SF.D;
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const unsigned teamLevel = _teams->TeamLevel( _level );
        
        // Compute the total new rank
        {
            // Add the F+=HH updates
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
                rank += _colXMap.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
                rank += _UMap.CurrentEntry()->Width();

            // Add the original low-rank matrix
            rank += F.Rank();
        }

        // Create the space and store the updates. If there are no dense 
        // updates, then mark two more matrices for QR factorization.
        {
            const int oldRank = F.Rank();
            const int height = F.Height();
            const int width = F.Width();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( F.U, UCopy );
            F.U.Resize( height, rank, height );
            std::memcpy
            ( F.U.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              height*oldRank*sizeof(Scalar) );

            Dense<Scalar> VCopy;
            hmat_tools::Copy( F.V, VCopy );
            F.V.Resize( width, rank, width );
            std::memcpy
            ( F.V.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              width*oldRank*sizeof(Scalar) );

            if( !_haveDenseUpdate )
            {
                Xs[XOffsets[teamLevel]++] = &F.U;
                Xs[XOffsets[teamLevel]++] = &F.V;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            // Combine all of the U's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int m = Height();
            const int numLowRankUpdates = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numLowRankUpdates; ++i,_UMap.Increment() )
                rank += _UMap.CurrentEntry()->Width();

            if( numLowRankUpdates == 0 )
                _UMap.Set( 0, new Dense<Scalar>( m, rank ) );
            else
            {
                _UMap.ResetIterator();
                Dense<Scalar>& firstU = *_UMap.CurrentEntry();
                _UMap.Increment();

                Dense<Scalar> firstUCopy;
                hmat_tools::Copy( firstU, firstUCopy );

                firstU.Resize( m, rank, m );
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstUCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), 
                          firstUCopy.LockedBuffer(0,j), m*sizeof(Scalar) );
                }
                // Push the rest of the updates in and then erase them
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& U = *_UMap.CurrentEntry();
                    const int r = U.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), U.LockedBuffer(0,j),
                          m*sizeof(Scalar) );
                    _UMap.EraseCurrentEntry();
                }
            }
        }
        else
        {
            // Combine all of the V's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int n = Width();
            const int numLowRankUpdates = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numLowRankUpdates; ++i,_VMap.Increment() )
                rank += _VMap.CurrentEntry()->Width();

            if( numLowRankUpdates == 0 )
                _VMap.Set( 0, new Dense<Scalar>( n, rank ) );
            else
            {
                _VMap.ResetIterator();
                Dense<Scalar>& firstV = *_VMap.CurrentEntry();
                _VMap.Increment();

                Dense<Scalar> firstVCopy;
                hmat_tools::Copy( firstV, firstVCopy );

                firstV.Resize( n, rank, n );
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstVCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), 
                          firstVCopy.LockedBuffer(0,j), n*sizeof(Scalar) );
                }
                // Push the rest of the updates in and erase them
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& V = *_VMap.CurrentEntry();
                    const int r = V.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), V.LockedBuffer(0,j),
                          n*sizeof(Scalar) );
                    _VMap.EraseCurrentEntry();
                }
            }
        }
        break;
    }
    case DENSE:
    {
        // Condense all of the U's and V's onto the dense matrix
        Dense<Scalar>& D = *_block.data.D;
        const int m = Height();
        const int n = Width();
        const int numLowRankUpdates = _UMap.Size();
        
        _UMap.ResetIterator();
        _VMap.ResetIterator();
        for( int update=0; update<numLowRankUpdates; ++update )
        {
            const Dense<Scalar>& U = *_UMap.CurrentEntry();
            const Dense<Scalar>& V = *_VMap.CurrentEntry();
            const int r = U.Width();
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( 'N', option, m, n, r,
              (Scalar)1, U.LockedBuffer(), U.LDim(),
                         V.LockedBuffer(), V.LDim(),
              (Scalar)1, D.Buffer(),       D.LDim() );
            _UMap.EraseCurrentEntry();
            _VMap.EraseCurrentEntry();
        }

        // Create space for storing the parent updates
        _UMap.Set( 0, new Dense<Scalar>(m,rank) );
        _VMap.Set( 0, new Dense<Scalar>(n,rank) );
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesLowRankImport
( int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLowRankImport");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        std::cout << "DIST_NODE, level=" << _level << ", rank=" << rank << ": ";

        int newRank = rank;
        if( teamSize == 2 )
        {
            if( _inTargetTeam )
            {
                const int tStart = (teamRank==0 ? 0 : 2);
                const int tStop = (teamRank==0 ? 2 : 4);
                const int numEntries = _UMap.Size();
                std::cout << "_UMap.Size()=" << numEntries << " ";
                _UMap.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& ULocal = *_UMap.CurrentEntry();
                    const int r = ULocal.Width();
                    Dense<Scalar> ULocalSub;
                    for( int t=tStart,tOffset=0; t<tStop; 
                         tOffset+=node.targetSizes[t],++t )
                    {
                        ULocalSub.LockedView
                        ( ULocal, tOffset, 0, node.targetSizes[t], r );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).MultiplyHMatUpdatesImportU
                            ( newRank, ULocalSub );
                    }
                    newRank += r;
                    _UMap.EraseCurrentEntry();
                }
            }
            if( _inSourceTeam )
            {
                newRank = rank;
                const int sStart = (teamRank==0 ? 0 : 2);
                const int sStop = (teamRank==0 ? 2 : 4);
                const int numEntries = _VMap.Size();
                std::cout << "_VMap.Size()=" << numEntries << " ";
                _VMap.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& VLocal = *_VMap.CurrentEntry();
                    const int r = VLocal.Width();
                    Dense<Scalar> VLocalSub;
                    for( int s=sStart,sOffset=0; s<sStop; 
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        VLocalSub.LockedView
                        ( VLocal, sOffset, 0, node.sourceSizes[s], r );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).MultiplyHMatUpdatesImportV
                            ( newRank, VLocalSub );
                    }
                    newRank += r;
                    _VMap.EraseCurrentEntry();
                }
            }
        }
        else // teamSize >= 4
        {
            if( _inTargetTeam )
            {
                const int numEntries = _UMap.Size();
                std::cout << "_UMap.Size()=" << numEntries << " ";
                _UMap.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& U = *_UMap.CurrentEntry();
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).MultiplyHMatUpdatesImportU
                            ( newRank, U );
                    newRank += U.Width();
                    _UMap.EraseCurrentEntry();
                }
            }
            if( _inSourceTeam )
            {
                newRank = rank;
                const int numEntries = _VMap.Size();
                std::cout << "_VMap.Size()=" << numEntries << " ";
                _VMap.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& V = *_VMap.CurrentEntry();
                    for( int s=0; s<4; ++s )
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).MultiplyHMatUpdatesImportV
                            ( newRank, V );
                    newRank += V.Width();
                    _VMap.EraseCurrentEntry();
                }
            }
        }
        std::cout << " newRank=" << newRank << std::endl;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankImport( newRank );
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        std::cout << "(SPLIT_)NODE, level=" << _level 
                  << ", rank=" << rank << ": ";
        Node& node = *_block.data.N;
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            std::cout << "_UMap.Size()=" << numEntries << " ";
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& U = *_UMap.CurrentEntry();
                Dense<Scalar> ULocal; 

                for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                {
                    ULocal.LockedView
                    ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).MultiplyHMatUpdatesImportU
                        ( newRank, ULocal );
                }
                newRank += U.Width();
                _UMap.EraseCurrentEntry();
            }
        }
        if( _inSourceTeam )
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            std::cout << "_VMap.Size()=" << numEntries << " ";
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& V = *_VMap.CurrentEntry();
                Dense<Scalar> VLocal;

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    VLocal.LockedView
                    ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                    for( int t=0; t<4; ++t )
                        node.Child(t,s).MultiplyHMatUpdatesImportV
                        ( newRank, VLocal );
                }
                newRank += V.Width();
                _VMap.EraseCurrentEntry();
            }
        }
        std::cout << " newRank=" << newRank << std::endl;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankImport( newRank );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        std::cout << "(DIST_/SPLIT_)LOW_RANK, level=" << _level 
                  << ", rank=" << rank << ": ";
        std::cout.flush();
        int newRank = rank;
        if( _inTargetTeam )
        {
            Dense<Scalar>* mainU;
            if( _block.type == DIST_LOW_RANK )
                mainU = &_block.data.DF->ULocal;
            else if( _block.type == SPLIT_LOW_RANK )
                mainU = &_block.data.SF->D;
            else
                mainU = &_block.data.F->U;

            int numEntries = _colXMap.Size();
            std::cout << "_colXMap.Size()=" << numEntries << " ";
            std::cout.flush();
            _colXMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                const Dense<Scalar>& X = *_colXMap.CurrentEntry();
                const int m = X.Height();
                const int r = X.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( mainU->Buffer(0,newRank+j), X.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                _colXMap.EraseCurrentEntry();
                newRank += r;
            }

            numEntries = _UMap.Size();
            std::cout << "_UMap.Size()=" << numEntries << " ";
            std::cout.flush();
            _UMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                const Dense<Scalar>& U = *_UMap.CurrentEntry();
                const int m = U.Height();
                const int r = U.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( mainU->Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                _UMap.EraseCurrentEntry();
                newRank += r;
            }
        }
        if( _inSourceTeam )
        {
            newRank = rank;

            Dense<Scalar>* mainV;
            if( _block.type == DIST_LOW_RANK )
                mainV = &_block.data.DF->VLocal;
            else if( _block.type == SPLIT_LOW_RANK )
                mainV = &_block.data.SF->D;
            else
                mainV = &_block.data.F->V;
            
            int numEntries = _rowXMap.Size();
            std::cout << "_rowXMap.Size()=" << numEntries << " ";
            std::cout.flush();
            _rowXMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                const Dense<Scalar>& X = *_rowXMap.CurrentEntry();
                const int n = X.Height();
                const int r = X.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( mainV->Buffer(0,newRank+j), X.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                _rowXMap.EraseCurrentEntry();
                newRank += r;
            }

            numEntries = _VMap.Size();
            std::cout << "_VMap.Size()=" << numEntries << " ";
            std::cout.flush();
            _VMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                const int n = V.Height();
                const int r = V.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( mainV->Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                _VMap.EraseCurrentEntry();
                newRank += r;
            }
        }
        std::cout << " newRank=" << newRank << std::endl;
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesImportU
( int rank, const Dense<Scalar>& U )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesImportU");
#endif
    if( !_inTargetTeam || Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamSize == 2 )
        {
            const int tStart = (teamRank==0 ? 0 : 2);            
            const int tStop = (teamRank==0 ? 2 : 4);
            Dense<Scalar> USub;
            for( int t=tStart,tOffset=0; t<tStop; 
                 tOffset+=node.targetSizes[t],++t )
            {
                USub.LockedView
                ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatUpdatesImportU( rank, USub );
            }
        }
        else  // teamSize >= 4
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatUpdatesImportU( rank, U );
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> USub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            USub.LockedView( U, tOffset, 0, node.targetSizes[t], U.Width() );
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesImportU( rank, USub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( DF.ULocal.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( SF.D.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( F.U.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
        break;
    }
    case SPLIT_DENSE:
    case DENSE:
    {
        _UMap.ResetIterator();
        Dense<Scalar>& mainU = *_UMap.CurrentEntry();
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( mainU.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesImportV
( int rank, const Dense<Scalar>& V )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesImportV");
#endif
    if( !_inSourceTeam || Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamSize == 2 )
        {
            const int sStart = (teamRank==0 ? 0 : 2);            
            const int sStop = (teamRank==0 ? 2 : 4);
            Dense<Scalar> VSub;
            for( int s=sStart,sOffset=0; s<sStop; 
                 sOffset+=node.sourceSizes[s],++s )
            {
                VSub.LockedView
                ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                for( int t=0; t<4; ++t )
                    node.Child(t,s).MultiplyHMatUpdatesImportV( rank, VSub );
            }
        }
        else  // teamSize >= 4
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatUpdatesImportV( rank, V );
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> VSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            VSub.LockedView( V, sOffset, 0, node.sourceSizes[s], V.Width() );
            for( int t=0; t<4; ++t )
                node.Child(t,s).MultiplyHMatUpdatesImportV( rank, VSub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( DF.VLocal.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( SF.D.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( F.V.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    case SPLIT_DENSE:
    case DENSE:
    {
        _VMap.ResetIterator();
        Dense<Scalar>& mainV = *_VMap.CurrentEntry();
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( mainV.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesLocalQR
( std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& qrWork )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLocalQR");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLocalQR
                ( tauBuffer, tauOffsets, qrWork );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        const int log2TeamSize = Log2( teamSize );
        const unsigned teamLevel = _teams->TeamLevel(_level);

        if( _inTargetTeam )
        {
            Dense<Scalar>& ULocal = DF.ULocal;
            const int m = ULocal.Height();
            const int r = ULocal.Width();
            if( r <= MaxRank() )
                break;

            lapack::QR
            ( m, r, ULocal.Buffer(), ULocal.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &qrWork[0], qrWork.size() );
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;
        }
        if( _inSourceTeam )
        {
            Dense<Scalar>& VLocal = DF.VLocal;
            const int n = VLocal.Height();
            const int r = VLocal.Width();
            if( r <= MaxRank() )
                break;

            lapack::QR
            ( n, r, VLocal.Buffer(), VLocal.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &qrWork[0], qrWork.size() );
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        if( !_haveDenseUpdate )
        {
            if( SF.rank <= MaxRank() )
                break;

            const unsigned teamLevel = _teams->TeamLevel(_level);
            lapack::QR
            ( SF.D.Height(), SF.rank, SF.D.Buffer(), SF.D.LDim(),
              &tauBuffer[tauOffsets[teamLevel]], &qrWork[0], qrWork.size() );
            tauOffsets[teamLevel] += SF.rank;
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        if( !_haveDenseUpdate )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            Dense<Scalar>& U = F.U;
            Dense<Scalar>& V = F.V;
            const int m = U.Height();
            const int n = V.Height();
            const int r = U.Width();
            if( r <= MaxRank() )
                break;

            lapack::QR
            ( m, r, U.Buffer(), U.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &qrWork[0], qrWork.size() );
            tauOffsets[teamLevel] += r;

            lapack::QR
            ( n, r, V.Buffer(), V.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &qrWork[0], qrWork.size() );
            tauOffsets[teamLevel] += r;
        }
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatParallelQR
( const std::vector<int>& numQRs, 
  const std::vector<Dense<Scalar>*>& Xs,
  const std::vector<int>& XOffsets,
        std::vector<int>& halfHeights,
  const std::vector<int>& halfHeightOffsets,
        std::vector<Scalar>& qrBuffer,
  const std::vector<int>& qrOffsets,
        std::vector<Scalar>& tauBuffer,
  const std::vector<int>& tauOffsets,
        std::vector<Scalar>& qrWork ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatParallelQR");
#endif
    const int globalRank = mpi::CommRank( _teams->Team(0) );
    const int numTeamLevels = _teams->NumLevels();
    const int numSteps = numTeamLevels-1;

    int passes = 0;
    for( int step=0; step<numSteps; ++step )
    {
        MPI_Comm team = _teams->Team( (numSteps-1)-step );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        // Flip the first bit of our rank in this team to get our partner,
        // and then check if our bit is 0 to see if we're the root
        const unsigned firstPartner = teamRank ^ (1u << passes);
        const unsigned firstRoot = !(teamRank & (1u << passes));

        // Compute the total message size for this step:
        //   The # of bytes for an int and (r*r+r)/2 scalars.
        int msgSize = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
            for( int k=0; k<numQRs[l]; ++k )
            {
                const int r = XLevel[k]->Width();
                msgSize += sizeof(int) + (r*r+r)/2*sizeof(Scalar);
            }
        }
        std::vector<byte> sendBuffer( msgSize ), recvBuffer( msgSize );

        // Pack the messages for the firstPartner
        byte* sendHead = &sendBuffer[0];
        for( int l=0; l<numSteps-step; ++l )
        {
            MPI_Comm parentTeam = _teams->Team(l);
            const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));
            const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
            const Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
            int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

            int qrPieceOffset = 0;
            for( int k=0; k<numQRs[l]; ++k )
            {
                const Dense<Scalar>& X = *XLevel[k];
                const int r = X.Width();
                const int halfHeightOffset = (k*log2ParentTeamSize+passes)*2;

                int halfHeight = 0; // initialize so compiler's don't complain
                if( step == 0 )
                {
                    // The first iteration bootstraps the local height from X
                    halfHeight = std::min(X.Height(),r);
                    if( firstRoot )
                        halfHeightLevel[halfHeightOffset] = halfHeight;
                    else
                        halfHeightLevel[halfHeightOffset+1] = halfHeight;
                    Write( sendHead, halfHeight );

                    // Copy our R out of X
                    for( int j=0; j<r; ++j )
                    {
                        const int P = std::min(j+1,halfHeight);
                        Write( sendHead, X.LockedBuffer(0,j), P );
                    }
                }
                else
                {
                    // All but the first iteration read from halfHeights
                    const int sPrev = halfHeightLevel[halfHeightOffset-2];
                    const int tPrev = halfHeightLevel[halfHeightOffset-1];
                    const int halfHeight = std::min(sPrev+tPrev,r);
                    Write( sendHead, halfHeight );

                    // Copy our R out of the last qrBuffer step
                    int qrOffset = qrPieceOffset + (passes-1)*(r*r+r);
                    for( int j=0; j<r; ++j )
                    {
                        const int SPrev = std::min(j+1,sPrev);
                        const int TPrev = std::min(j+1,tPrev);
                        const int P = std::min(j+1,halfHeight);
                        Write( sendHead, &qrLevel[qrOffset], P );
                        qrOffset += SPrev + TPrev;
                    }
                }

                // Move past the padding for this send piece
                if( halfHeight < r )
                {
                    const int sendTrunc = r - halfHeight;
                    sendHead += 
                        (sendTrunc*sendTrunc+sendTrunc)/2*sizeof(Scalar);
                }
                
                // Advance our index to the next send piece
                qrPieceOffset += log2ParentTeamSize*(r*r+r);
            }
        }

        // Exchange with our first partner
        mpi::SendRecv
        ( &sendBuffer[0], msgSize, firstPartner, 0,
          &recvBuffer[0], msgSize, firstPartner, 0, team );

        if( teamSize == 2 )
        {
            // Unpack the recv messages, perform the QR factorizations, and
            // store the local height for the next step if there is one
            const byte* recvHead = &recvBuffer[0];
            for( int l=0; l<numSteps-step; ++l )
            {
                MPI_Comm parentTeam = _teams->Team(l);
                const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));

                bool rootOfNextStep = false;
                const bool haveAnotherComm = ( l+1 < numSteps-step );
                if( haveAnotherComm )
                    rootOfNextStep = !(globalRank & (1u<<(passes+1)));

                const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
                Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
                Scalar* tauLevel = &tauBuffer[tauOffsets[l]];
                int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

                int qrPieceOffset=0, tauPieceOffset=0;
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const Dense<Scalar>& X = *XLevel[k];
                    const int r = X.Width();
                    const int halfHeightOffset = 
                        (k*log2ParentTeamSize+passes)*2;

                    int s, t;
                    int qrOffset = qrPieceOffset + passes*(r*r+r);
                    if( step == 0 )
                    {
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from X    
                                std::memcpy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j),
                                  S*sizeof(Scalar) );
                                qrOffset += S;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from X
                                std::memcpy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j),
                                  T*sizeof(Scalar) );
                                qrOffset += T;
                            }
                        }
                    }
                    else
                    {
                        int qrPrevOffset = qrPieceOffset + (passes-1)*(r*r+r);
                        const int sPrev = halfHeightLevel[halfHeightOffset-2];
                        const int tPrev = halfHeightLevel[halfHeightOffset-1];
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from the previous qrBuffer
                                std::memcpy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  S*sizeof(Scalar) );
                                qrOffset += S;
                                qrPrevOffset += SPrev + TPrev;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from the previous qrBuffer
                                std::memcpy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  T*sizeof(Scalar) );
                                qrOffset += T;
                                qrPrevOffset += SPrev + TPrev;
                            }
                        }
                    }
                    // Move past the padding for this recv piece
                    const int recvTrunc = 
                        ( firstRoot ? r-std::min(t,r) : r-std::min(s,r) );
                    recvHead += 
                        (recvTrunc*recvTrunc+recvTrunc)/2*sizeof(Scalar);

                    hmat_tools::PackedQR
                    ( r, s, t, &qrLevel[qrPieceOffset+passes*(r*r+r)],
                      &tauLevel[tauPieceOffset+(passes+1)*r], &qrWork[0] );
                    
                    if( haveAnotherComm )
                    {
                        const int minDim = std::min(s+t,r);
                        if( rootOfNextStep )
                            halfHeightLevel[halfHeightOffset+2] = minDim;
                        else
                            halfHeightLevel[halfHeightOffset+3] = minDim;
                    }

                    qrPieceOffset += log2ParentTeamSize*(r*r+r);
                    tauPieceOffset += (log2ParentTeamSize+1)*r;
                }
            }
            ++passes;
        }
        else // teamSize >= 4, so this level splits 4 ways
        {
            const unsigned secondPartner = teamRank ^ (1u<<(passes+1));
            const bool secondRoot = !(teamRank & (1u<<(passes+1)));

            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step and into the next 
            // send buffer in a single sweep.
            sendHead = &sendBuffer[0];
            const byte* recvHead = &recvBuffer[0];
            for( int l=0; l<numSteps-step; ++l )
            {
                MPI_Comm parentTeam = _teams->Team(l);
                const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));

                const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
                Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
                Scalar* tauLevel = &tauBuffer[tauOffsets[l]];
                int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

                int qrPieceOffset=0, tauPieceOffset=0;
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const Dense<Scalar>& X = *XLevel[k];
                    const int r = X.Width();
                    const int halfHeightOffset = 
                        (k*log2ParentTeamSize+passes)*2;

                    int s, t;
                    int qrOffset = qrPieceOffset + passes*(r*r+r);
                    if( step == 0 )
                    {
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from X    
                                std::memcpy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j),
                                  S*sizeof(Scalar) );
                                qrOffset += S;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from X
                                std::memcpy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j),
                                  T*sizeof(Scalar) );
                                qrOffset += T;
                            }
                        }
                    }
                    else
                    {
                        int qrPrevOffset = qrPieceOffset + (passes-1)*(r*r+r);
                        const int sPrev = halfHeightLevel[halfHeightOffset-2];
                        const int tPrev = halfHeightLevel[halfHeightOffset-1];
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from last qrBuffer
                                std::memcpy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  S*sizeof(Scalar) );
                                qrOffset += S;
                                qrPrevOffset += SPrev + TPrev;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from last qrBuffer
                                std::memcpy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  T*sizeof(Scalar) );
                                qrOffset += T;
                                qrPrevOffset += SPrev + TPrev;
                            }
                        }
                    }
                    // Move past the padding for this recv piece
                    const int recvTrunc = 
                        ( firstRoot ? r-std::min(t,r) : r-std::min(s,r) );
                    recvHead += 
                        (recvTrunc*recvTrunc+recvTrunc)/2*sizeof(Scalar);

                    hmat_tools::PackedQR
                    ( r, s, t, 
                      &qrLevel[qrPieceOffset+passes*(r*r+r)],
                      &tauLevel[tauPieceOffset+(passes+1)*r], &qrWork[0] );

                    const int minDim = std::min(s+t,r);
                    Write( sendHead, minDim );
                    if( secondRoot )
                        halfHeightLevel[halfHeightOffset+2] = minDim;
                    else
                        halfHeightLevel[halfHeightOffset+3] = minDim;
                    qrOffset = qrPieceOffset + passes*(r*r+r);
                    for( int j=0; j<r; ++j )
                    {
                        const int S = std::min(j+1,s);
                        const int T = std::min(j+1,t);
                        const int P = std::min(j+1,s+t);
                        Write( sendHead, &qrLevel[qrOffset], P );
                        qrOffset += S + T;
                    }
                    // Move past the send padding
                    const int sendTrunc = r-std::min(minDim,r);
                    sendHead += 
                        (sendTrunc*sendTrunc+sendTrunc)/2*sizeof(Scalar);

                    qrPieceOffset += log2ParentTeamSize*(r*r+r);
                    tauPieceOffset += (log2ParentTeamSize+1)*r;
                }
            }
            
            // Exchange with our second partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, secondPartner, 0,
              &recvBuffer[0], msgSize, secondPartner, 0, team );
            
            // Unpack the recv messages and perform the QR factorizations, then
            // fill the local heights for the next step if there is one.
            recvHead = &recvBuffer[0];
            for( int l=0; l<numSteps-step; ++l )
            {
                MPI_Comm parentTeam = _teams->Team(l);
                const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));
                
                bool rootOfNextStep = false;
                const bool haveAnotherComm = ( l+1 < numSteps-step );
                if( haveAnotherComm )
                    rootOfNextStep = !(globalRank & (1u<<(passes+2)));

                const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
                Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
                Scalar* tauLevel = &tauBuffer[tauOffsets[l]];
                int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

                int qrPieceOffset=0, tauPieceOffset=0;
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const Dense<Scalar>& X = *XLevel[k];
                    const int r = X.Width();
                    const int halfHeightOffset 
                        = (k*log2ParentTeamSize+(passes+1))*2;

                    int s, t;
                    int qrOffset = qrPieceOffset + (passes+1)*(r*r+r);
                    int qrPrevOffset = qrPieceOffset + passes*(r*r+r);
                    const int sPrev = halfHeightLevel[halfHeightOffset-2];
                    const int tPrev = halfHeightLevel[halfHeightOffset-1];
                    if( secondRoot )
                    {
                        s = halfHeightLevel[halfHeightOffset];
                        t = Read<int>( recvHead );
                        halfHeightLevel[halfHeightOffset+1] = t;
                        for( int j=0; j<r; ++j )
                        {
                            const int S = std::min(j+1,s);
                            const int T = std::min(j+1,t);
                            const int SPrev = std::min(j+1,sPrev);
                            const int TPrev = std::min(j+1,tPrev);

                            // Read column strip from last qrBuffer
                            std::memcpy
                            ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                              S*sizeof(Scalar) );
                            qrOffset += S;
                            qrPrevOffset += SPrev + TPrev;

                            // Read column strip from recvBuffer
                            Read( &qrBuffer[qrOffset], recvHead, T );
                            qrOffset += T;
                        }
                    }
                    else
                    {
                        s = Read<int>( recvHead );
                        t = halfHeightLevel[halfHeightOffset+1];
                        halfHeightLevel[halfHeightOffset] = s;
                        for( int j=0; j<r; ++j )
                        {
                            const int S = std::min(j+1,s);
                            const int T = std::min(j+1,t);
                            const int SPrev = std::min(j+1,sPrev);
                            const int TPrev = std::min(j+1,tPrev);

                            // Read column strip from recvBuffer
                            Read( &qrLevel[qrOffset], recvHead, S );
                            qrOffset += S;

                            // Read column strip from last qrBuffer
                            std::memcpy
                            ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                              T*sizeof(Scalar) );
                            qrOffset += T;
                            qrPrevOffset += SPrev + TPrev;
                        }
                    }
                    // Move past the padding for this recv piece
                    const int recvTrunc = 
                        ( secondRoot ? r-std::min(t,r) : r-std::min(s,r) );
                    recvHead += 
                        (recvTrunc*recvTrunc+recvTrunc)/2*sizeof(Scalar);

                    hmat_tools::PackedQR
                    ( r, s, t, &qrLevel[qrPieceOffset+(passes+1)*(r*r+r)], 
                      &tauLevel[tauPieceOffset+(passes+2)*r], &qrWork[0] );

                    if( haveAnotherComm )
                    {
                        const int minDim = std::min(s+t,r);
                        if( rootOfNextStep )
                            halfHeightLevel[halfHeightOffset+2] = minDim;
                        else
                            halfHeightLevel[halfHeightOffset+3] = minDim;
                    }

                    qrPieceOffset += log2ParentTeamSize*(r*r+r);
                    tauPieceOffset += (log2ParentTeamSize+1)*r;
                }
            }
            passes += 2;
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesExchangeCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesExchangeCount");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesExchangeCount
                ( sendSizes, recvSizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *_block.data.DF;
        const int r = DF.rank;
        if( (_inTargetTeam && _inSourceTeam) || r <= MaxRank() )
            break;

        const int teamRank = mpi::CommRank( _teams->Team(_level) );
        const int exchangeSize = sizeof(int) + (r*r+r)/2*sizeof(Scalar);
        if( _inTargetTeam )
        {
            AddToMap( sendSizes, _sourceRoot+teamRank, exchangeSize );
            AddToMap( recvSizes, _sourceRoot+teamRank, exchangeSize );
        }
        else
        {
            AddToMap( sendSizes, _targetRoot+teamRank, exchangeSize );
            AddToMap( recvSizes, _targetRoot+teamRank, exchangeSize );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        const int r = SF.rank;
        if( !_haveDenseUpdate )
        {
            if( r <= MaxRank() )
                break;

            const int minUDim = std::min( Height(), r );
            const int minVDim = std::min( Width(), r );
            int packedRUSize = 
                (minUDim*minUDim+minUDim)/2 + (r-minUDim)*minUDim;
            int packedRVSize = 
                (minVDim*minVDim+minVDim)/2 + (r-minVDim)*minVDim;

            // Count the exchange R sizes
            if( _inTargetTeam )
            {
                AddToMap( sendSizes, _sourceRoot, packedRUSize*sizeof(Scalar) );
                AddToMap( recvSizes, _sourceRoot, packedRVSize*sizeof(Scalar) );
            }
            else
            {
                AddToMap( sendSizes, _targetRoot, packedRVSize*sizeof(Scalar) );
                AddToMap( recvSizes, _targetRoot, packedRUSize*sizeof(Scalar) );
            }
        }
        else
        {
            // Factor in exchanging all info needed for both processes to
            // locally finish their updates. 
            const int m = Height();
            const int n = Width();
            if( _inTargetTeam )
            {
                AddToMap
                ( sendSizes, _sourceRoot, m*SF.rank*sizeof(Scalar) );
                AddToMap
                ( recvSizes, _sourceRoot, (n*SF.rank+m*n)*sizeof(Scalar) );
            }
            else
            {
                AddToMap
                ( sendSizes, _targetRoot, (n*SF.rank+m*n)*sizeof(Scalar) );
                AddToMap
                ( recvSizes, _targetRoot, m*SF.rank*sizeof(Scalar) );
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        // Count the send/recv sizes of the U's from the low-rank updates
        if( _inTargetTeam )
        {
            _UMap.ResetIterator();
            const Dense<Scalar>& U = *_UMap.CurrentEntry();
            AddToMap
            ( sendSizes, _sourceRoot, Height()*U.Width()*sizeof(Scalar) );
        }
        else
        {
            _VMap.ResetIterator();
            const Dense<Scalar>& V = *_VMap.CurrentEntry();
            AddToMap
            ( recvSizes, _targetRoot, Height()*V.Width()*sizeof(Scalar) );
        }
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesExchangePack
( std::vector<byte>& sendBuffer, std::map<int,int>& sendOffsets,
  const std::vector<int>& halfHeights, std::vector<int>& halfHeightOffsets,
  const std::vector<Scalar>& qrBuffer, std::vector<int>& qrOffsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesExchangePack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesExchangePack
                ( sendBuffer, sendOffsets, 
                  halfHeights, halfHeightOffsets, qrBuffer, qrOffsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *_block.data.DF;
        const int r = DF.rank;
        if( r <= MaxRank() )
            break;

        MPI_Comm team = _teams->Team( _level );
        const unsigned teamLevel = _teams->TeamLevel( _level );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned teamRank = mpi::CommRank( team );
        const unsigned log2TeamSize = Log2( teamSize );
        if( _inTargetTeam && _inSourceTeam )
        {
            halfHeightOffsets[teamLevel] += 2*log2TeamSize*2;
            qrOffsets[teamLevel] += 2*log2TeamSize*(r*r+r);
            break;
        }

        const int* lastHalfHeightsStage = 
            &halfHeights[halfHeightOffsets[teamLevel]+(log2TeamSize-1)*2];
        const Scalar* lastQRStage = 
            &qrBuffer[qrOffsets[teamLevel]+(log2TeamSize-1)*(r*r+r)];
        halfHeightOffsets[teamLevel] += log2TeamSize*2;
        qrOffsets[teamLevel] += log2TeamSize*(r*r+r);

        const int partner = 
            ( _inTargetTeam ? _sourceRoot : _targetRoot ) + teamRank;
        int sendOffset = sendOffsets[partner];

        const int s = lastHalfHeightsStage[0];
        const int t = lastHalfHeightsStage[1];
        const int minDim = std::min(s+t,r);
        std::memcpy( &sendBuffer[sendOffset], &minDim, sizeof(int) );
        sendOffset += sizeof(int);

        int offset = 0;
        for( int j=0; j<r; ++j )
        {
            const int S = std::min(j+1,s);
            const int T = std::min(j+1,t);
            const int P = std::min(j+1,s+t);
            std::memcpy
            ( &sendBuffer[sendOffset], &lastQRStage[offset], P*sizeof(Scalar) );
            sendOffset += P*sizeof(Scalar);
            offset += S + T;
        }
        sendOffsets[partner] += sizeof(int) + (r*r+r)/2*sizeof(Scalar);
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        const int r = SF.rank;
        if( !_haveDenseUpdate )
        {
            if( r <= MaxRank() )
                break;

            // Pack R
            const int partner = ( _inTargetTeam ? _sourceRoot : _targetRoot );
            const int minDim = std::min( SF.D.Height(), r );
            int sendOffset = sendOffsets[partner];
            for( int j=0; j<minDim; ++j )
            {
                std::memcpy
                ( &sendBuffer[sendOffset],
                  SF.D.LockedBuffer(0,j), (j+1)*sizeof(Scalar) );
                sendOffset += (j+1)*sizeof(Scalar);
            }
            for( int j=minDim; j<r; ++j )
            {
                std::memcpy
                ( &sendBuffer[sendOffset],
                  SF.D.LockedBuffer(0,j), minDim*sizeof(Scalar) );
                sendOffset += minDim*sizeof(Scalar);
            }
            sendOffsets[partner] += 
                ((minDim*minDim+minDim)/2 + (r-minDim)*minDim)*sizeof(Scalar);
        }
        else
        {
            if( _inTargetTeam )
            {
                const int m = Height();
                const int sendOffset = sendOffsets[_sourceRoot];

                // Copy U into the send buffer
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &sendBuffer[sendOffset+j*m*sizeof(Scalar)],
                      SF.D.LockedBuffer(0,j), m*sizeof(Scalar) );

                sendOffsets[_sourceRoot] += m*r*sizeof(Scalar);
            }
            else
            {
                const int m = Height();
                const int n = Width();

                // Copy V into the send buffer
                int sendOffset = sendOffsets[_targetRoot];
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &sendBuffer[sendOffset+j*n*sizeof(Scalar)],
                      SF.D.LockedBuffer(0,j), n*sizeof(Scalar) );
                sendOffset += n*r*sizeof(Scalar);

                // Copy the condensed dense update into the send buffer
                for( int j=0; j<n; ++j )
                    std::memcpy
                    ( &sendBuffer[sendOffset+j*m*sizeof(Scalar)], 
                      _D.LockedBuffer(0,j), m*sizeof(Scalar) );
                sendOffsets[_targetRoot] += (n*r + m*n)*sizeof(Scalar);
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            _UMap.ResetIterator();
            const Dense<Scalar>& U = *_UMap.CurrentEntry();
            const int height = Height();
            const int r = U.Width();
            const int sendOffset = sendOffsets[_sourceRoot];
            for( int j=0; j<r; ++j )
                std::memcpy
                ( &sendBuffer[sendOffset+j*height*sizeof(Scalar)],
                  U.LockedBuffer(0,j), height*sizeof(Scalar) );
            sendOffsets[_sourceRoot] += height*r*sizeof(Scalar);
            _UMap.Clear();
        }
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesExchangeFinalize
( const std::vector<byte>& recvBuffer, std::map<int,int>& recvOffsets,
  const std::vector<int>& halfHeights, std::vector<int>& halfHeightOffsets,
  const std::vector<Scalar>& qrBuffer, std::vector<int>& qrOffsets,
  const std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  Dense<Scalar>& X, Dense<Scalar>& Y, Dense<Scalar>& Z,
  std::vector<Real>& singularValues, 
  std::vector<Scalar>& work, std::vector<Real>& realWork )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesExchangeFinalize");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesExchangeFinalize
                ( recvBuffer, recvOffsets, halfHeights, halfHeightOffsets,
                  qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
                  X, Y, Z, singularValues, work, realWork );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;
        const int r = DF.rank;
        if( r <= MaxRank() )
            break;

        MPI_Comm team = _teams->Team( _level );
        const unsigned teamLevel = _teams->TeamLevel( _level );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned teamRank = mpi::CommRank( team );
        const unsigned log2TeamSize = Log2( teamSize );

        const int*    UHalfHeightsPiece;
        const int*    VHalfHeightsPiece;
        const Scalar* UQRPiece;
        const Scalar* VQRPiece;
        const Scalar* UTauPiece;
        const Scalar* VTauPiece;
        const int*    lastUHalfHeightsStage;
        const int*    lastVHalfHeightsStage;
        const Scalar* lastUQRStage;
        const Scalar* lastVQRStage;
        const Scalar* lastUTauStage;
        const Scalar* lastVTauStage;

        // Set up our pointers and form R_U and R_V in X and Y
        int minDimX, minDimY;
        if( _inTargetTeam && _inSourceTeam )
        {
            UHalfHeightsPiece = &halfHeights[halfHeightOffsets[teamLevel]];
            UQRPiece = &qrBuffer[qrOffsets[teamLevel]];
            UTauPiece = &tauBuffer[tauOffsets[teamLevel]];

            lastUHalfHeightsStage = &UHalfHeightsPiece[(log2TeamSize-1)*2];
            lastUQRStage = &UQRPiece[(log2TeamSize-1)*(r*r+r)];
            lastUTauStage = &UTauPiece[log2TeamSize*r];

            halfHeightOffsets[teamLevel] += log2TeamSize*2;
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;

            VHalfHeightsPiece = &halfHeights[halfHeightOffsets[teamLevel]];
            VQRPiece = &qrBuffer[qrOffsets[teamLevel]];
            VTauPiece = &tauBuffer[tauOffsets[teamLevel]];

            lastVHalfHeightsStage = &VHalfHeightsPiece[(log2TeamSize-1)*2];
            lastVQRStage = &VQRPiece[(log2TeamSize-1)*(r*r+r)];
            lastVTauStage = &VTauPiece[log2TeamSize*r];

            halfHeightOffsets[teamLevel] += log2TeamSize*2;
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;

            const int sX = lastUHalfHeightsStage[0];
            const int tX = lastUHalfHeightsStage[1];
            const int sY = lastVHalfHeightsStage[0];
            const int tY = lastVHalfHeightsStage[1];
            minDimX = std::min(sX+tX,r);
            minDimY = std::min(sY+tY,r);

            X.Resize( minDimX, r );
            hmat_tools::Scale( (Scalar)0, X );
            int offset = 0;
            for( int j=0; j<r; ++j )
            {
                const int S = std::min(j+1,sX);
                const int T = std::min(j+1,tX);
                const int P = std::min(j+1,sX+tX);
                std::memcpy
                ( X.Buffer(0,j), &lastUQRStage[offset], P*sizeof(Scalar) );
                offset += S+T;
            }

            Y.Resize( minDimY, r );
            hmat_tools::Scale( (Scalar)0, Y );
            offset = 0;
            for( int j=0; j<r; ++j )
            {
                const int S = std::min(j+1,sY);
                const int T = std::min(j+1,tY);
                const int P = std::min(j+1,sY+tY);
                std::memcpy
                ( Y.Buffer(0,j), &lastVQRStage[offset], P*sizeof(Scalar) );
                offset += S+T;
            }
        }
        else if( _inTargetTeam )
        {
            UHalfHeightsPiece = &halfHeights[halfHeightOffsets[teamLevel]];
            UQRPiece = &qrBuffer[qrOffsets[teamLevel]];
            UTauPiece = &tauBuffer[tauOffsets[teamLevel]];

            lastUHalfHeightsStage = &UHalfHeightsPiece[(log2TeamSize-1)*2];
            lastUQRStage = &UQRPiece[(log2TeamSize-1)*(r*r+r)];
            lastUTauStage = &UTauPiece[log2TeamSize*r];

            halfHeightOffsets[teamLevel] += log2TeamSize*2;
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;

            VHalfHeightsPiece = 0;
            VQRPiece = 0;
            VTauPiece = 0;
            lastVHalfHeightsStage = 0;
            lastVQRStage = 0;
            lastVTauStage = 0;

            const int sX = lastUHalfHeightsStage[0];
            const int tX = lastUHalfHeightsStage[1];
            minDimX = std::min(sX+tX,r);
            
            X.Resize( minDimX, r );
            hmat_tools::Scale( (Scalar)0, X );
            int offset = 0;
            for( int j=0; j<r; ++j )
            {
                const int S = std::min(j+1,sX);
                const int T = std::min(j+1,tX);
                const int P = std::min(j+1,sX+tX);
                std::memcpy
                ( X.Buffer(0,j), &lastUQRStage[offset], 
                  P*sizeof(Scalar) );
                offset += S+T;
            }

            // We need to determine minimum dimension of Y. 
            const int partner = _sourceRoot + teamRank;
            int recvOffset = recvOffsets[partner];
            std::memcpy( &minDimY, &recvBuffer[recvOffset], sizeof(int) );
            recvOffset += sizeof(int);

            Y.Resize( minDimY, r );
            hmat_tools::Scale( (Scalar)0, Y );
            for( int j=0; j<r; ++j )
            {
                const int P = std::min(j+1,minDimY);
                std::memcpy
                ( Y.Buffer(0,j), &recvBuffer[recvOffset], P*sizeof(Scalar) );
                recvOffset += P*sizeof(Scalar);
            }
            recvOffsets[partner] += sizeof(int) + (r*r+r)/2*sizeof(Scalar);
        }
        else // _inSourceTeam
        {
            UHalfHeightsPiece = 0;
            UQRPiece = 0;
            UTauPiece = 0;
            lastUHalfHeightsStage = 0;
            lastUQRStage = 0;
            lastUTauStage = 0;

            VHalfHeightsPiece = &halfHeights[halfHeightOffsets[teamLevel]];
            VQRPiece = &qrBuffer[qrOffsets[teamLevel]];
            VTauPiece = &tauBuffer[tauOffsets[teamLevel]];

            lastVHalfHeightsStage = &VHalfHeightsPiece[(log2TeamSize-1)*2];
            lastVQRStage = &VQRPiece[(log2TeamSize-1)*(r*r+r)];
            lastVTauStage = &VTauPiece[log2TeamSize*r];

            halfHeightOffsets[teamLevel] += log2TeamSize*2;
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;

            const int sY = lastVHalfHeightsStage[0];
            const int tY = lastVHalfHeightsStage[1];
            minDimY = std::min(sY+tY,r);

            Y.Resize( minDimY, r );
            hmat_tools::Scale( (Scalar)0, Y );
            int offset = 0;
            for( int j=0; j<r; ++j )
            {
                const int S = std::min(j+1,sY);
                const int T = std::min(j+1,tY);
                const int P = std::min(j+1,sY+tY);
                std::memcpy
                ( Y.Buffer(0,j), &lastVQRStage[offset], P*sizeof(Scalar) );
                offset += S+T;
            }

            // Unpack the minimum dimension of X
            const int partner = _targetRoot + teamRank;
            int recvOffset = recvOffsets[partner];
            std::memcpy( &minDimX, &recvBuffer[recvOffset], sizeof(int) );
            recvOffset += sizeof(int);

            // Unpack X
            X.Resize( minDimX, r );
            hmat_tools::Scale( (Scalar)0, X );
            for( int j=0; j<r; ++j )
            {
                const int P = std::min(j+1,minDimX);
                std::memcpy
                ( X.Buffer(0,j), &recvBuffer[recvOffset], P*sizeof(Scalar) );
                recvOffset += P*sizeof(Scalar);
            }
            recvOffsets[partner] += sizeof(int) + (r*r+r)/2*sizeof(Scalar);
        }

        // Overwrite Z with R_U R_V^[T/H]
        Z.Resize( minDimX, minDimY );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, minDimX, minDimY, r,
          (Scalar)1, X.LockedBuffer(), X.LDim(),
                     Y.LockedBuffer(), Y.LDim(),
          (Scalar)0, Z.Buffer(),       Z.LDim() );
        const int minDim = std::min( minDimX, minDimY );

        // Perform an SVD on Z, overwriting Z with the left singular vectors
        // and Y with the adjoint of the right singular vectors. 
        //
        // NOTE: Even though we potentially only need either the left or right
        //       singular vectors, we must be careful to ensure that the signs
        //       are chosen consistently with our partner process. Though we 
        //       could adopt an implicit convention, it is likely easier to 
        //       just compute both for now.
        singularValues.resize( minDim );
        work.resize( lapack::SVDWorkSize(minDimX,minDimY) );
        realWork.resize( lapack::SVDRealWorkSize(minDimX,minDimY) );
        Y.Resize( minDim, minDimY );
        lapack::SVD
        ( 'O', 'S', minDimX, minDimY, Z.Buffer(), Z.LDim(), 
          &singularValues[0], 0, 1, Y.Buffer(), Y.LDim(), 
          &work[0], work.size(), &realWork[0] );

        const int newRank = std::min(minDim,MaxRank());
        X.Resize( 2*r, newRank );
        DF.rank = newRank;
        if( _inTargetTeam )
        {
            // Form the compressed local portion of U.

            // Copy the first newRank singular vectors, scaled by the singular 
            // values, into the top of the X buffer
            const int sLast = lastUHalfHeightsStage[0];
            const int tLast = lastUHalfHeightsStage[1];
            X.Resize( sLast+tLast, newRank );
            hmat_tools::Scale( (Scalar)0, X );
            for( int j=0; j<newRank; ++j )
            {
                const Real sigma = singularValues[j];
                const Scalar* ZCol = Z.LockedBuffer(0,j);
                Scalar* XCol = X.Buffer(0,j);
                for( int i=0; i<minDimX; ++i )
                    XCol[i] = sigma*ZCol[i];
            }

            // Backtransform the last stage
            work.resize( newRank );
            hmat_tools::ApplyPackedQFromLeft
            ( r, sLast, tLast, lastUQRStage, lastUTauStage, X, &work[0] );

            // Backtransform using the middle stages.
            // We know that the height cannot increase as we move backwards
            // (except for the original stage)
            int sPrev=sLast, tPrev=tLast;
            for( int commStage=log2TeamSize-2; commStage>=0; --commStage )
            {
                const int sCurr = UHalfHeightsPiece[commStage*2];
                const int tCurr = UHalfHeightsPiece[commStage*2+1];
                X.Resize( sCurr+tCurr, newRank );

                const bool rootOfPrevStage = !(teamRank & (1u<<(commStage+1)));
                if( rootOfPrevStage )
                {
                    // Zero the bottom part of X 
                    for( int j=0; j<newRank; ++j )
                        std::memset
                        ( X.Buffer(sCurr,j), 0, tCurr*sizeof(Scalar) );
                }
                else
                {
                    // Move the bottom part to the top part and zero the bottom
                    for( int j=0; j<newRank; ++j )
                    {
                        std::memcpy
                        ( X.Buffer(0,j), X.LockedBuffer(sCurr,j), 
                          tCurr*sizeof(Scalar) );
                        std::memset
                        ( X.Buffer(sCurr,j), 0, tCurr*sizeof(Scalar) );
                    }
                }
                hmat_tools::ApplyPackedQFromLeft
                ( r, sCurr, tCurr, &UQRPiece[commStage*(r*r+r)],
                  &UTauPiece[(commStage+1)*r], X, &work[0] );

                sPrev = sCurr;
                tPrev = tCurr;
            }

            // Backtransform using the original stage
            const int m = DF.ULocal.Height();
            hmat_tools::Copy( DF.ULocal, Z );
            DF.ULocal.Resize( m, newRank );
            hmat_tools::Scale( (Scalar)0, DF.ULocal );
            const bool rootOfPrevStage = !(teamRank & 0x1);
            if( rootOfPrevStage )
            {
                // Copy the first sPrev rows of the top part of X into the 
                // top of ULocal
                for( int j=0; j<newRank; ++j )
                    std::memcpy
                    ( DF.ULocal.Buffer(0,j), X.LockedBuffer(0,j),
                      sPrev*sizeof(Scalar) );
            }
            else
            {
                // Copy the first tPrev rows of the bottom part of X into
                // the top of ULocal
                for( int j=0; j<newRank; ++j )
                    std::memcpy
                    ( DF.ULocal.Buffer(0,j), X.LockedBuffer(sPrev,j),
                      tPrev*sizeof(Scalar) );
            }
            work.resize( lapack::ApplyQWorkSize('L',m,newRank) );
            lapack::ApplyQ
            ( 'L', 'N', m, newRank, std::min(m,r), 
              Z.LockedBuffer(), Z.LDim(), &UTauPiece[0],
              DF.ULocal.Buffer(), DF.ULocal.LDim(),  
              &work[0], work.size() );
        }
        if( _inSourceTeam )
        {
            // Form the compressed local portion of V.

            // Copy the first newRank right singular vectors into the top of
            // the X buffer
            const int sLast = lastVHalfHeightsStage[0];
            const int tLast = lastVHalfHeightsStage[1];
            X.Resize( sLast+tLast, newRank );
            hmat_tools::Scale( (Scalar)0, X );
            for( int j=0; j<newRank; ++j )
            {
                const Scalar* YRow = Y.LockedBuffer(j,0);
                const int YLDim = Y.LDim();
                Scalar* XCol = X.Buffer(0,j);
                if( Conjugated )
                    for( int i=0; i<minDimY; ++i )
                        XCol[i] = Conj(YRow[i*YLDim]);
                else
                    for( int i=0; i<minDimY; ++i )
                        XCol[i] = YRow[i*YLDim];
            }

            // Backtransform the last stage
            work.resize( newRank );
            hmat_tools::ApplyPackedQFromLeft
            ( r, sLast, tLast, lastVQRStage, lastVTauStage, X, &work[0] );

            // Backtransform using the middle stages.
            // We know that the height cannot increase as we move backwards
            // (except for the original stage)
            int sPrev=sLast, tPrev=tLast;
            for( int commStage=log2TeamSize-2; commStage>=0; --commStage )
            {
                const int sCurr = VHalfHeightsPiece[commStage*2];
                const int tCurr = VHalfHeightsPiece[commStage*2+1];
                X.Resize( sCurr+tCurr, newRank );

                const bool rootOfPrevStage = !(teamRank & (1u<<(commStage+1)));
                if( rootOfPrevStage )
                {
                    // Zero the bottom part of X
                    for( int j=0; j<newRank; ++j )
                        std::memset
                        ( X.Buffer(sCurr,j), 0, tCurr*sizeof(Scalar) );
                }
                else
                {
                    // Move the bottom part to the top part and zero the bottom
                    for( int j=0; j<newRank; ++j )
                    {
                        std::memcpy
                        ( X.Buffer(0,j), X.LockedBuffer(sCurr,j), 
                          tCurr*sizeof(Scalar) );
                        std::memset
                        ( X.Buffer(sCurr,j), 0, tCurr*sizeof(Scalar) );
                    }
                }
                hmat_tools::ApplyPackedQFromLeft
                ( r, sCurr, tCurr, &VQRPiece[commStage*(r*r+r)],
                  &VTauPiece[(commStage+1)*r], X, &work[0] );

                sPrev = sCurr;
                tPrev = tCurr;
            }

            // Backtransform using the original stage
            const int n = DF.VLocal.Height();
            hmat_tools::Copy( DF.VLocal, Z );
            DF.VLocal.Resize( n, newRank );
            hmat_tools::Scale( (Scalar)0, DF.VLocal );
            const bool rootOfPrevStage = !(teamRank & 0x1);
            if( rootOfPrevStage )
            {
                // Copy the first sPrev rows of the top part of X into the 
                // top of VLocal
                for( int j=0; j<newRank; ++j )
                    std::memcpy
                    ( DF.VLocal.Buffer(0,j), X.LockedBuffer(0,j),
                      sPrev*sizeof(Scalar) );
            }
            else
            {
                // Copy the first tPrev rows of the bottom part of X into
                // the top of VLocal
                for( int j=0; j<newRank; ++j )
                    std::memcpy
                    ( DF.VLocal.Buffer(0,j), X.LockedBuffer(sPrev,j),
                      tPrev*sizeof(Scalar) );
            }

            work.resize( lapack::ApplyQWorkSize('L',n,newRank) );
            lapack::ApplyQ
            ( 'L', 'N', n, newRank, std::min(n,r),
              Z.LockedBuffer(), Z.LDim(), &VTauPiece[0],
              DF.VLocal.Buffer(), DF.VLocal.LDim(),  
              &work[0], work.size() );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        const int r = SF.rank;
        const int minDimU = std::min( Height(), r );
        const int minDimV = std::min( Width(), r );
        const int minDim = std::min( minDimU, minDimV );
        const unsigned teamLevel = _teams->TeamLevel( _level );

        if( !_haveDenseUpdate )
        {
            if( r <= MaxRank() )
                break;

            // Form R_U and R_V
            X.Resize( minDimU, r );
            Y.Resize( minDimV, r );
            if( _inTargetTeam )
            {
                hmat_tools::Scale( (Scalar)0, X );
                for( int j=0; j<minDimU; ++j )
                    std::memcpy
                    ( X.Buffer(0,j), SF.D.LockedBuffer(0,j), 
                      (j+1)*sizeof(Scalar) );
                for( int j=minDimU; j<r; ++j )
                    std::memcpy
                    ( X.Buffer(0,j), SF.D.LockedBuffer(0,j),
                      minDimU*sizeof(Scalar) );

                int recvOffset = recvOffsets[_sourceRoot];
                hmat_tools::Scale( (Scalar)0, Y );
                for( int j=0; j<minDimV; ++j )
                {
                    std::memcpy
                    ( Y.Buffer(0,j), &recvBuffer[recvOffset], 
                      (j+1)*sizeof(Scalar) );
                    recvOffset += (j+1)*sizeof(Scalar);
                }
                for( int j=minDimV; j<r; ++j )
                {
                    std::memcpy
                    ( Y.Buffer(0,j), &recvBuffer[recvOffset],
                      minDimV*sizeof(Scalar) );
                    recvOffset += minDimV*sizeof(Scalar);
                }

                const int packedVSize = 
                    (minDimV*minDimV+minDimV)/2 + (r-minDimV)*minDimV;
                recvOffsets[_sourceRoot] += packedVSize*sizeof(Scalar);
            }
            else // _inSourceTeam
            {
                hmat_tools::Scale( (Scalar)0, X );
                int recvOffset = recvOffsets[_targetRoot];
                for( int j=0; j<minDimU; ++j )
                {
                    std::memcpy
                    ( X.Buffer(0,j), &recvBuffer[recvOffset], 
                      (j+1)*sizeof(Scalar) );
                    recvOffset += (j+1)*sizeof(Scalar);
                }
                for( int j=minDimU; j<r; ++j )
                {
                    std::memcpy
                    ( X.Buffer(0,j), &recvBuffer[recvOffset],
                      minDimU*sizeof(Scalar) );
                    recvOffset += minDimU*sizeof(Scalar);
                }

                const int packedUSize = 
                    (minDimU*minDimU+minDimU)/2 + (r-minDimU)*minDimU;
                recvOffsets[_targetRoot] += packedUSize*sizeof(Scalar);

                hmat_tools::Scale( (Scalar)0, Y );
                for( int j=0; j<minDimV; ++j )
                    std::memcpy
                    ( Y.Buffer(0,j), SF.D.LockedBuffer(0,j), 
                      (j+1)*sizeof(Scalar) );
                for( int j=minDimV; j<r; ++j )
                    std::memcpy
                    ( Y.Buffer(0,j), SF.D.LockedBuffer(0,j),
                      minDimV*sizeof(Scalar) );
            }

            // Overwrite Z with R_U R_V^[T/H]
            Z.Resize( minDimU, minDimV );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( 'N', option, minDimU, minDimV, r,
              (Scalar)1, X.LockedBuffer(), X.LDim(),
                         Y.LockedBuffer(), Y.LDim(),
              (Scalar)0, Z.Buffer(),       Z.LDim() );

            singularValues.resize( minDim );
            work.resize( lapack::SVDWorkSize(minDimU,minDimV) );
            realWork.resize( lapack::SVDRealWorkSize(minDimU,minDimV) );
            const int newRank = std::min( minDim, MaxRank() );
            SF.rank = newRank;
            if( _inTargetTeam )
            {
                // Perform an SVD on Z, overwriting Z with the left singular 
                // vectors.
                lapack::SVD
                ( 'O', 'N', minDimU, minDimV, Z.Buffer(), Z.LDim(), 
                  &singularValues[0], 0, 1, 0, 1, 
                  &work[0], work.size(), &realWork[0] );

                // Backtransform U
                const int m = SF.D.Height();
                hmat_tools::Copy( SF.D, X );
                SF.D.Resize( m, newRank );
                hmat_tools::Scale( (Scalar)0, SF.D );
                // Copy the first newRank singular vectors, scaled by the 
                // singular values, into the top of the SF.D buffer
                for( int j=0; j<newRank; ++j )
                {
                    const Real sigma = singularValues[j];
                    const Scalar* ZCol = Z.LockedBuffer(0,j);
                    Scalar* UCol = SF.D.Buffer(0,j);
                    for( int i=0; i<minDimU; ++i )
                        UCol[i] = sigma*ZCol[i];
                }

                work.resize( lapack::ApplyQWorkSize('L',m,newRank) );
                lapack::ApplyQ
                ( 'L', 'N', m, newRank, minDimU,
                  X.LockedBuffer(), X.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]],
                  SF.D.Buffer(),    SF.D.LDim(),  
                  &work[0], work.size() );
                tauOffsets[teamLevel] += r; // this is an upper bound
            }
            else // _inSourceTeam
            {
                // Perform an SVD on Z, overwriting Z with the adjoint of the
                // right singular vectors.
                lapack::SVD
                ( 'N', 'O', minDimU, minDimV, Z.Buffer(), Z.LDim(), 
                  &singularValues[0], 0, 1, 0, 1, 
                  &work[0], work.size(), &realWork[0] );

                // Backtransform V
                const int n = SF.D.Height();
                hmat_tools::Copy( SF.D, Y );
                SF.D.Resize( n, newRank );
                hmat_tools::Scale( (Scalar)0, SF.D );
                // Copy the first newRank right singular vectors into the 
                // top of the SF.D buffer
                for( int j=0; j<newRank; ++j )
                {
                    const Scalar* ZRow = Z.LockedBuffer(j,0);
                    const int ZLDim = Z.LDim();
                    Scalar* VCol = SF.D.Buffer(0,j);
                    if( Conjugated )
                        for( int i=0; i<minDimV; ++i )
                            VCol[i] = Conj(ZRow[i*ZLDim]);
                    else
                        for( int i=0; i<minDimV; ++i )
                            VCol[i] = ZRow[i*ZLDim];
                }

                work.resize( lapack::ApplyQWorkSize('L',n,newRank) );
                lapack::ApplyQ
                ( 'L', 'N', n, newRank, minDimV,
                  Y.LockedBuffer(), Y.LDim(), &tauBuffer[tauOffsets[teamLevel]],
                  SF.D.Buffer(),    SF.D.LDim(),  
                  &work[0], work.size() );
                tauOffsets[teamLevel] += r; // this is an upper bound
            }
        }
        else
        {
            const int m = Height();
            const int n = Width();
            const int minDim = std::min( m, n );
            const int maxRank = MaxRank();
            if( _inTargetTeam )
            {
                // Unpack V into X
                int recvOffset = recvOffsets[_sourceRoot];
                X.Resize( n, r );
                for( int j=0; j<r; ++j )
                {
                    std::memcpy
                    ( X.Buffer(0,j), &recvBuffer[recvOffset], 
                      n*sizeof(Scalar) );
                    recvOffset += n*sizeof(Scalar);
                }

                // Unpack the dense update
                Z.Resize( m, n );
                for( int j=0; j<n; ++j )
                {
                    std::memcpy
                    ( Z.Buffer(0,j), &recvBuffer[recvOffset],
                      m*sizeof(Scalar) );
                    recvOffset += m*sizeof(Scalar);
                }
                recvOffsets[_sourceRoot] += (n*r + m*n)*sizeof(Scalar);

                // Add U V^[T/H] onto the dense update
                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( 'N', option, m, n, r, 
                  (Scalar)1, SF.D.LockedBuffer(), SF.D.LDim(),
                             X.LockedBuffer(),    X.LDim(),
                  (Scalar)1, Z.Buffer(),          Z.LDim() );

                if( minDim <= maxRank )
                {
                    SF.rank = minDim;
                    if( m == minDim )
                    {
                        // Make U := I (where V := Z^[T/H])
                        SF.D.Resize( m, m );
                        hmat_tools::Scale( (Scalar)0, SF.D );
                        for( int j=0; j<m; ++j )
                            SF.D.Set(j,j,(Scalar)1);
                    }
                    else
                    {
                        // Make U := Z (where V := I)
                        hmat_tools::Copy( Z, SF.D );
                    }
                }
                else // minDim > maxRank
                {
                    SF.rank = maxRank;

                    // Perform an SVD on the dense matrix, overwriting it with
                    // the left singular vectors 
                    singularValues.resize( minDim );
                    work.resize( lapack::SVDWorkSize(m,n) );
                    realWork.resize( lapack::SVDRealWorkSize(m,n) );
                    lapack::SVD
                    ( 'O', 'N', m, n, Z.Buffer(), Z.LDim(), 
                      &singularValues[0], 0, 1, 0, 1, 
                      &work[0], work.size(), &realWork[0] );

                    // Form U with the truncated left singular vectors scaled
                    // by the corresponding singular values
                    SF.D.Resize( m, maxRank );
                    for( int j=0; j<maxRank; ++j )
                    {
                        Scalar* UCol = SF.D.Buffer(0,j);
                        const Scalar* ZCol = Z.Buffer(0,j);
                        const Real sigma = singularValues[j];
                        for( int i=0; i<m; ++i )
                            UCol[i] = sigma*ZCol[i];
                    }
                }
            }
            else // _inSourceTeam
            {
                // Add U V^[T/H] onto the dense update
                const char option = ( Conjugated ? 'C' : 'T' );
                int recvOffset = recvOffsets[_targetRoot];
                const Scalar* UBuffer = (const Scalar*)&recvBuffer[recvOffset];
                blas::Gemm
                ( 'N', option, m, n, r, 
                  (Scalar)1, UBuffer,             m,
                             SF.D.LockedBuffer(), SF.D.LDim(),
                  (Scalar)1, _D.Buffer(),         _D.LDim() );
                recvOffsets[_targetRoot] += m*r*sizeof(Scalar);

                if( minDim <= maxRank )
                {
                    SF.rank = minDim;
                    if( m == minDim )
                    {
                        // Make V := _D^[T/H] (where U := I)
                        if( Conjugated )
                            hmat_tools::Adjoint( _D, SF.D );
                        else
                            hmat_tools::Transpose( _D, SF.D );
                    }
                    else // n == minDim
                    {
                        // Make V := I (where U := _D)
                        SF.D.Resize( n, n );
                        hmat_tools::Scale( (Scalar)0, SF.D );
                        for( int j=0; j<n; ++j )
                            SF.D.Set(j,j,(Scalar)1);
                    }
                }
                else // minDim > maxRank
                {
                    SF.rank = maxRank;

                    // Perform an SVD on the dense matrix, overwriting it with
                    // adjoint of the right singular vectors
                    singularValues.resize( minDim );
                    work.resize( lapack::SVDWorkSize(m,n) );
                    realWork.resize( lapack::SVDRealWorkSize(m,n) );
                    lapack::SVD
                    ( 'N', 'O', m, n, _D.Buffer(), _D.LDim(), 
                      &singularValues[0], 0, 1, 0, 1, 
                      &work[0], work.size(), &realWork[0] );

                    // Form V with the truncated right singular vectors
                    SF.D.Resize( n, maxRank );
                    for( int j=0; j<maxRank; ++j )
                    {
                        Scalar* VCol = SF.D.Buffer(0,j);
                        const Scalar* DRow = _D.Buffer(j,0);
                        const int DLDim = _D.LDim();
                        if( Conjugated )
                            for( int i=0; i<n; ++i )
                                VCol[i] = Conj(DRow[i*DLDim]);
                        else
                            for( int i=0; i<n; ++i )
                                VCol[i] = DRow[i*DLDim];
                    }
                }
            }
            _D.Clear();
            _haveDenseUpdate = false;
            _storedDenseUpdate = false;
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int r = F.Rank();
        const int minDimU = std::min( Height(), r );
        const int minDimV = std::min( Width(), r );
        const int minDim = std::min( minDimU, minDimV );
        const unsigned teamLevel = _teams->TeamLevel( _level );

        if( !_haveDenseUpdate )
        {
            if( r <= MaxRank() )
                break;

            // Form R_U and R_V
            X.Resize( minDimU, r );
            Y.Resize( minDimV, r );

            hmat_tools::Scale( (Scalar)0, X );
            for( int j=0; j<r; ++j )
                std::memcpy
                ( X.Buffer(0,j), F.U.LockedBuffer(0,j), 
                  std::min(minDimU,j+1)*sizeof(Scalar) );
            hmat_tools::Scale( (Scalar)0, Y );
            for( int j=0; j<r; ++j )
                std::memcpy
                ( Y.Buffer(0,j), F.V.LockedBuffer(0,j),
                  std::min(minDimV,j+1)*sizeof(Scalar) );

            // Z := R_U R_V^[T/H]
            Z.Resize( minDimU, minDimV );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( 'N', option, minDimU, minDimV, r,
              (Scalar)1, X.LockedBuffer(), X.LDim(),
                         Y.LockedBuffer(), Y.LDim(),
              (Scalar)0, Z.Buffer(),       Z.LDim() );

            const int maxRank = MaxRank();
            singularValues.resize( minDim );
            work.resize( lapack::SVDWorkSize(minDimU,minDimV) );
            realWork.resize( lapack::SVDRealWorkSize(minDimU,minDimV) );

            // Perform an SVD on Z, overwriting Z with the left singular 
            // vectors, and Y with the adjoint of the right singular vectors.
            lapack::SVD
            ( 'O', 'S', minDimU, minDimV, Z.Buffer(), Z.LDim(), 
              &singularValues[0], 0, 1, Y.Buffer(), Y.LDim(), 
              &work[0], work.size(), &realWork[0] );
            const int newRank = std::min(minDim,maxRank);

            // Backtransform U
            const int m = F.Height();
            hmat_tools::Copy( F.U, X );
            F.U.Resize( m, newRank );
            hmat_tools::Scale( (Scalar)0, F.U );
            // Copy the first few singular vectors, scaled by the 
            // singular values, into the top of the F.U buffer
            for( int j=0; j<newRank; ++j )
            {
                const Real sigma = singularValues[j];
                const Scalar* ZCol = Z.LockedBuffer(0,j);
                Scalar* UCol = F.U.Buffer(0,j);
                for( int i=0; i<minDimU; ++i )
                    UCol[i] = sigma*ZCol[i];
            }

            work.resize( lapack::ApplyQWorkSize('L',m,newRank) );
            lapack::ApplyQ
            ( 'L', 'N', m, newRank, minDimU,
              X.LockedBuffer(), X.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]],
              F.U.Buffer(), F.U.LDim(),  
              &work[0], work.size() );
            tauOffsets[teamLevel] += r;

            // Backtransform V
            const int n = F.Width();
            hmat_tools::Copy( F.V, X );
            F.V.Resize( n, newRank );
            hmat_tools::Scale( (Scalar)0, F.V );
            // Copy the first newRank right singular vectors into the 
            // top of the F.V buffer
            for( int j=0; j<newRank; ++j )
            {
                const Scalar* YRow = Y.LockedBuffer(j,0);
                const int YLDim = Y.LDim();
                Scalar* VCol = F.V.Buffer(0,j);
                if( Conjugated )
                    for( int i=0; i<minDimV; ++i )
                        VCol[i] = Conj(YRow[i*YLDim]);
                else
                    for( int i=0; i<minDimV; ++i )
                        VCol[i] = YRow[i*YLDim];
            }

            work.resize( lapack::ApplyQWorkSize('L',n,maxRank) );
            lapack::ApplyQ
            ( 'L', 'N', n, newRank, minDimV,
              X.LockedBuffer(), X.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]],
              F.V.Buffer(), F.V.LDim(),  
              &work[0], work.size() );
            tauOffsets[teamLevel] += r;
        }
        else
        {
            const int m = F.Height();
            const int n = F.Width();
            const int minDim = std::min( m, n );
            const int maxRank = MaxRank();

            // Add U V^[T/H] onto the dense update
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( 'N', option, m, n, r, 
              (Scalar)1, F.U.LockedBuffer(), F.U.LDim(),
                         F.V.LockedBuffer(), F.V.LDim(),
              (Scalar)1, _D.Buffer(),        _D.LDim() );

            if( minDim <= maxRank )
            {
                if( m == minDim )
                {
                    // Make U := I and V := _D^[T/H]
                    F.U.Resize( minDim, minDim );
                    hmat_tools::Scale( (Scalar)0, F.U );
                    for( int j=0; j<minDim; ++j )
                        F.U.Set(j,j,(Scalar)1);
                    if( Conjugated )
                        hmat_tools::Adjoint( _D, F.V );
                    else
                        hmat_tools::Transpose( _D, F.V );
                }
                else
                {
                    // Make U := _D and V := I
                    hmat_tools::Copy( _D, F.U );
                    F.V.Resize( minDim, minDim );
                    hmat_tools::Scale( (Scalar)0, F.V );
                    for( int j=0; j<minDim; ++j )
                        F.V.Set(j,j,(Scalar)1);
                }
            }
            else // minDim > maxRank
            {
                // Perform an SVD on the dense matrix, overwriting it with
                // the left singular vectors and Y with the adjoint of the 
                // right singular vecs
                Y.Resize( std::min(m,n), n );
                singularValues.resize( minDim );
                work.resize( lapack::SVDWorkSize(m,n) );
                realWork.resize( lapack::SVDRealWorkSize(m,n) );
                lapack::SVD
                ( 'O', 'S', m, n, _D.Buffer(), _D.LDim(), 
                  &singularValues[0], 0, 1, Y.Buffer(), Y.LDim(), 
                  &work[0], work.size(), &realWork[0] );

                // Form U with the truncated left singular vectors scaled
                // by the corresponding singular values
                F.U.Resize( m, maxRank );
                for( int j=0; j<maxRank; ++j )
                {
                    Scalar* UCol = F.U.Buffer(0,j);
                    const Scalar* DCol = _D.Buffer(0,j);
                    const Real sigma = singularValues[j];
                    for( int i=0; i<m; ++i )
                        UCol[i] = sigma*DCol[i];
                }

                // Form V with the truncated right singular vectors
                F.V.Resize( n, maxRank );
                for( int j=0; j<maxRank; ++j )
                {
                    Scalar* VCol = F.V.Buffer(0,j);
                    const Scalar* YRow = Y.Buffer(j,0);
                    const int YLDim = Y.LDim();
                    if( Conjugated )
                        for( int i=0; i<n; ++i )
                            VCol[i] = Conj(YRow[i*YLDim]);
                    else
                        for( int i=0; i<n; ++i )
                            VCol[i] = YRow[i*YLDim];
                }
            }
            _D.Clear();
            _haveDenseUpdate = false;
            _storedDenseUpdate = false;
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
        {
            SplitDense& SD = *_block.data.SD;
            const int m = SD.D.Height();
            const int n = SD.D.Width();

            _VMap.ResetIterator();
            const Dense<Scalar>& V = *_VMap.CurrentEntry();
            const int r = V.Width();

            // Add U V^[T/H] onto our dense matrix
            const char option = ( Conjugated ? 'C' : 'T' );
            const Scalar* UBuffer = 
                (const Scalar*)&recvBuffer[recvOffsets[_targetRoot]];
            blas::Gemm
            ( 'N', option, m, n, r,
              (Scalar)1, UBuffer,          m,
                         V.LockedBuffer(), V.LDim(),
              (Scalar)1, SD.D.Buffer(),    SD.D.LDim() );
            recvOffsets[_targetRoot] += m*r*sizeof(Scalar);

            _VMap.Clear();
        }
        break;
    }
    case DENSE:
    {
        Dense<Scalar>& D = *_block.data.D;      
        const int m = D.Height();
        const int n = D.Width();

        _UMap.ResetIterator();
        _VMap.ResetIterator();
        const Dense<Scalar>& U = *_UMap.CurrentEntry();
        const Dense<Scalar>& V = *_VMap.CurrentEntry();
        const int r = U.Width();

        // Add U V^[T/H] onto our dense matrix
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, m, n, r,
          (Scalar)1, U.LockedBuffer(), U.LDim(),
                     V.LockedBuffer(), V.LDim(),
          (Scalar)1, D.Buffer(),       D.LDim() );

        _UMap.Clear();
        _VMap.Clear();
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

