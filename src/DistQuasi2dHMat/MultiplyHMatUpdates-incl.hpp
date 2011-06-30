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
    std::vector<int> rankOffsets(numTeamLevels);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        rankOffsets[teamLevel] = numTotalQRs;        
        numTotalQRs += numQRs[teamLevel];
    }
    std::vector<int> ranks(numTotalQRs);
    MultiplyHMatUpdatesLowRankCountAndResize( ranks, rankOffsets, 0 );

    // Carry the low-rank updates down from nodes into the low-rank and dense
    // blocks.
    MultiplyHMatUpdatesLowRankImport( 0 );

    // Allocate space for packed storage of the various components in our
    // distributed QR factorizations.
    numTotalQRs = 0;
    int qrTotalSize=0, tauTotalSize=0, maxRank=0;
    std::vector<int> qrOffsets(numTeamLevels+1), 
                     tauOffsets(numTeamLevels+1);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = _teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        for( int i=0; i<numQRs[teamLevel]; ++i )
        {
            const int r = ranks[rankOffsets[teamLevel]+i];
            maxRank = std::max(maxRank,r);

            qrTotalSize += log2TeamSize*(r*r+r);
            tauTotalSize += (log2TeamSize+1)*r;
        }
        numTotalQRs += numQRs[teamLevel];
    }
    qrOffsets[numTeamLevels] = qrTotalSize;
    tauOffsets[numTeamLevels] = tauTotalSize;

    std::vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize );
    {
        std::vector<Scalar> qrWork( lapack::QRWorkSize(maxRank) );
        std::vector<int> qrOffsetsCopy, tauOffsetsCopy;

        // Make a disposable copy of the offset vectors and perform the local
        // QR portions of the (distributed) QRs
        qrOffsetsCopy = qrOffsets;
        tauOffsetsCopy = tauOffsets;
        MultiplyHMatUpdatesLocalQR
        ( qrBuffer, qrOffsetsCopy, tauBuffer, tauOffsetsCopy, qrWork );

        // Perform the parallel portion of the TSQR algorithm
        qrOffsetsCopy = qrOffsets;
        tauOffsetsCopy = tauOffsets;
        MultiplyHMatUpdatesParallelQR
        ( numQRs, ranks, rankOffsets,
          qrBuffer, qrOffsetsCopy, tauBuffer, tauOffsetsCopy, qrWork );
    }

    // Count the number of entries of R and U that we need to exchange
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
    std::vector<Scalar> sendBuffer( totalSendSize );
    {
        std::map<int,int> sendOffsetsCopy = sendOffsets;
        std::vector<int> qrOffsetsCopy = qrOffsets;

        MultiplyHMatUpdatesExchangePack
        ( sendBuffer, sendOffsetsCopy, qrBuffer, qrOffsetsCopy );
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
    std::vector<Scalar> recvBuffer( totalRecvSize );
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

    const int maxLocalDim = std::max( LocalHeight(), LocalWidth() );
    const int maxQHeight = std::max( maxLocalDim, 2*maxRank );
    Dense<Scalar> RU( maxRank, maxRank ), 
                  RV( maxRank, maxRank ), 
                  W( 2*maxRank, maxRank );
    std::vector<Real> singularValues( maxRank );
    std::vector<Scalar>  
        applyQWork( lapack::ApplyQWorkSize('L',maxQHeight,maxRank) );
    std::vector<Scalar> svdWork( lapack::SVDWorkSize(maxRank,maxRank) );
    std::vector<Real> svdRealWork( lapack::SVDRealWorkSize(maxRank,maxRank) );
    MultiplyHMatUpdatesExchangeFinalize
    ( recvBuffer, recvOffsets, qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
      RU, RV, singularValues, W, svdWork, svdRealWork, applyQWork );

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
        if( _inTargetTeam && _DMap.Size() == 0 )
            ++numQRs[teamLevel];
        if( _inSourceTeam && _DMap.Size() == 0 )
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
( std::vector<int>& ranks, std::vector<int>& rankOffsets, int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLowRankCountAndResize");
#endif
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
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }
        else if( _inSourceTeam )
        {
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _VMap.NextEntry()->Width();
        }

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankCountAndResize
                ( ranks, rankOffsets, rank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        // Compute the total update rank
        if( _inTargetTeam )
        {
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _colXMap.NextEntry()->Width();

            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }
        else if( _inSourceTeam )
        {
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _rowXMap.NextEntry()->Width();

            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _VMap.NextEntry()->Width();
        }

        // Store the rank and create the space
        const unsigned teamLevel = _teams->TeamLevel( _level );
        if( _inTargetTeam )
        {
            ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = DF.ULocal.Width();
            Dense<Scalar> ULocalCopy;
            hmat_tools::Copy( DF.ULocal, ULocalCopy );
            DF.ULocal.Resize( LocalHeight(), rank );
            std::memcpy
            ( DF.ULocal.Buffer(0,rank-oldRank), ULocalCopy.LockedBuffer(), 
              LocalHeight()*rank*sizeof(Scalar) );
        }
        if( _inSourceTeam )
        {
            ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = DF.VLocal.Width();
            Dense<Scalar> VLocalCopy;
            hmat_tools::Copy( DF.VLocal, VLocalCopy );
            DF.VLocal.Resize( LocalWidth(), rank );
            std::memcpy
            ( DF.VLocal.Buffer(0,rank-oldRank), VLocalCopy.LockedBuffer(),
              LocalWidth()*rank*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;

        const unsigned numDenseUpdates = _DMap.Size();
        const unsigned teamLevel = _teams->TeamLevel( _level );

        // Compute the total update rank
        if( _inTargetTeam )
        {
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _colXMap.NextEntry()->Width();

            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }
        else 
        {
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _rowXMap.NextEntry()->Width();

            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _VMap.NextEntry()->Width();
        }

        // Create the space and store the rank if we'll need to do a QR
        if( _inTargetTeam )
        {
            const int numDenseUpdates = _DMap.Size();
            if( numDenseUpdates == 0 )
                ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = SF.D.Width();
            Dense<Scalar> UCopy;
            hmat_tools::Copy( SF.D, UCopy );
            SF.D.Resize( Height(), rank );
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              Height()*rank*sizeof(Scalar) );
        }
        else
        {
            if( numDenseUpdates == 0 )
                ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = SF.D.Width();
            Dense<Scalar> VCopy;
            hmat_tools::Copy( SF.D, VCopy );
            SF.D.Resize( Width(), rank );
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              Width()*rank*sizeof(Scalar) );
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;

        const unsigned numDenseUpdates = _DMap.Size();
        const unsigned teamLevel = _teams->TeamLevel( _level );
        // Compute the total update rank
        {
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _colXMap.NextEntry()->Width();

            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }

        // Create the space and store the updates if there is no dense update
        {
            if( numDenseUpdates == 0 )
            {
                ranks[rankOffsets[teamLevel]++] = rank;
                ranks[rankOffsets[teamLevel]++] = rank;
            }
            const int oldRank = F.Rank();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( F.U, UCopy );
            F.U.Resize( Height(), rank );
            std::memcpy
            ( F.U.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              Height()*rank*sizeof(Scalar) );

            Dense<Scalar> VCopy;
            hmat_tools::Copy( F.V, VCopy );
            F.V.Resize( Width(), rank );
            std::memcpy
            ( F.V.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              Width()*rank*sizeof(Scalar) );
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
            for( int update=0; update<numLowRankUpdates; ++update )
                rank += _UMap.NextEntry()->Width();

            if( numLowRankUpdates == 0 )
            {
                _UMap[0] = new Dense<Scalar>( m, rank );
            }
            else
            {
                _UMap.ResetIterator();
                Dense<Scalar> firstUCopy;
                Dense<Scalar> firstU = *_UMap.NextEntry();
                firstUCopy = firstU;
                firstU.Resize( m, rank );
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
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& U = *_UMap.NextEntry();
                    const int r = U.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), U.LockedBuffer(0,j),
                          m*sizeof(Scalar) );
                    _UMap.EraseLastEntry();
                }
            }
        }
        else
        {
            // Combine all of the U's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int n = Width();
            const int numLowRankUpdates = _VMap.Size();
            _VMap.ResetIterator();
            for( int update=0; update<numLowRankUpdates; ++update )
                rank += _VMap.NextEntry()->Width();

            if( numLowRankUpdates == 0 )
            {
                _VMap[0] = new Dense<Scalar>( n, rank );
            }
            else
            {
                _VMap.ResetIterator();
                Dense<Scalar> firstVCopy;
                Dense<Scalar> firstV = *_VMap.NextEntry();
                firstVCopy = firstV;
                firstV.Resize( n, rank );
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
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& V = *_VMap.NextEntry();
                    const int r = V.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), V.LockedBuffer(0,j),
                          n*sizeof(Scalar) );
                    _VMap.EraseLastEntry();
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
            const Dense<Scalar>& U = *_UMap.NextEntry();
            const Dense<Scalar>& V = *_VMap.NextEntry();
            const int r = U.Width();
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( 'N', option, m, n, r,
              (Scalar)1, U.LockedBuffer(), U.LDim(),
                         V.LockedBuffer(), V.LDim(),
              (Scalar)1, D.Buffer(),       D.LDim() );
            _UMap.EraseLastEntry();
            _VMap.EraseLastEntry();
        }

        const int numDenseUpdates = _DMap.Size();
        _DMap.ResetIterator();
        for( int update=0; update<numDenseUpdates; ++update )
        {
            const Dense<Scalar>& DUpdate = *_DMap.NextEntry();
            for( int j=0; j<n; ++j )
            {
                const Scalar* DUpdateCol = DUpdate.LockedBuffer(0,j);
                Scalar* DCol = D.Buffer(0,j);
                for( int i=0; i<m; ++i )
                    DCol[i] += DUpdateCol[i];
            }
            _DMap.EraseLastEntry();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesLowRankImport
( int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLowRankImport");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& U = *_UMap.NextEntry();
                Dense<Scalar> ULocal; 

                for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                {
                    for( int s=0; s<4; ++s )
                    {
                        ULocal.LockedView
                        ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                        node.Child(t,s).MultiplyHMatUpdatesImportU
                        ( newRank, ULocal );
                    }
                }
                newRank += U.Width();
                _UMap.EraseLastEntry();
            }
        }
        if( _inSourceTeam )
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& V = *_VMap.NextEntry();
                Dense<Scalar> VLocal;

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    for( int t=0; t<4; ++t )
                    {
                        VLocal.LockedView
                        ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                        node.Child(t,s).MultiplyHMatUpdatesImportV
                        ( newRank, VLocal );
                    }
                }
                newRank += V.Width();
                _VMap.EraseLastEntry();
            }
        }

        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankImport( newRank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF; 
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& U = *_UMap.NextEntry();
                const int m = U.Height();
                const int r = U.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( DF.ULocal.Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                _UMap.EraseLastEntry();
                newRank += r;
            }
        }
        if( _inSourceTeam )
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& V = *_VMap.NextEntry();
                const int n = V.Height();
                const int r = V.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( DF.VLocal.Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                _VMap.EraseLastEntry();
                newRank += r;
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF; 
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& U = *_UMap.NextEntry();
                const int m = U.Height();
                const int r = U.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( SF.D.Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                _UMap.EraseLastEntry();
                newRank += r;
            }
        }
        else
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& V = *_VMap.NextEntry();
                const int n = V.Height();
                const int r = V.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( SF.D.Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                _VMap.EraseLastEntry();
                newRank += r;
            }
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        int newRank = rank;
        const int numEntries = _UMap.Size();
        _UMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            Dense<Scalar>& U = *_UMap.NextEntry();
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.U.Buffer(0,newRank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
            _UMap.EraseLastEntry();
            newRank += r;
        }
        newRank = rank;
        _VMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            Dense<Scalar>& V = *_VMap.NextEntry();
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.V.Buffer(0,newRank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
            _VMap.EraseLastEntry();
            newRank += r;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesImportU
( int rank, const Dense<Scalar>& U )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesImportU");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> ULocal;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0; s<4; ++s )
            {
                ULocal.LockedView
                ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                node.Child(t,s).MultiplyHMatUpdatesImportU( rank, ULocal );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inTargetTeam )
        {
            DistLowRank& DF = *_block.data.DF;
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( DF.ULocal.Buffer(0,rank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 && _inTargetTeam )
        {
            SplitLowRank& SF = *_block.data.SF;
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( SF.D.Buffer(0,rank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
        }
        break;
    }
    case LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            LowRank<Scalar,Conjugated>& F = *_block.data.F;
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.U.Buffer(0,rank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesImportV
( int rank, const Dense<Scalar>& V )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesImportV");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> VLocal;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            for( int t=0; t<4; ++t )
            {
                VLocal.LockedView
                ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                node.Child(t,s).MultiplyHMatUpdatesImportV( rank, VLocal );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam )
        {
            DistLowRank& DF = *_block.data.DF;
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( DF.VLocal.Buffer(0,rank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 && _inSourceTeam )
        {
            SplitLowRank& SF = *_block.data.SF;
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( SF.D.Buffer(0,rank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
        }
        break;
    }
    case LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            LowRank<Scalar,Conjugated>& F = *_block.data.F;
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.V.Buffer(0,rank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesLocalQR
( std::vector<Scalar>& qrBuffer,  std::vector<int>& qrOffsets,
  std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& qrWork )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLocalQR");
#endif
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
                ( qrBuffer, qrOffsets, tauBuffer, tauOffsets, qrWork );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
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

            std::memset
            ( &qrBuffer[qrOffsets[teamLevel]], 0, (r*r+r)*sizeof(Scalar) );
            const bool root = !(teamRank & 0x1);
            if( root )
            {
                // Copy our R into the upper triangle of the next
                // matrix to factor (which is 2r x r)
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                      ULocal.LockedBuffer(0,j), 
                      std::min(m,j+1)*sizeof(Scalar) );
            }
            else
            {
                // Copy our R into the lower triangle of the next
                // matrix to factor (which is 2r x r)
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                      ULocal.LockedBuffer(0,j), 
                      std::min(m,j+1)*sizeof(Scalar) );
            }
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
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

            std::memset
            ( &qrBuffer[qrOffsets[teamLevel]], 0, (r*r+r)*sizeof(Scalar) );
            const bool root = !(teamRank & 0x1);
            if( root )
            {
                // Copy our R into the upper triangle of the next
                // matrix to factor (which is 2r x r)
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                      VLocal.LockedBuffer(0,j), 
                      std::min(n,j+1)*sizeof(Scalar) );
            }
            else
            {
                // Copy our R into the lower triangle of the next
                // matrix to factor (which is 2r x r)
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                      VLocal.LockedBuffer(0,j), 
                      std::min(n,j+1)*sizeof(Scalar) );
            }
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;

        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            if( _inTargetTeam )
            {
                Dense<Scalar>& U = SF.D;
                const int m = U.Height();
                const int r = U.Width();
                if( r <= MaxRank() )
                    break;

                lapack::QR
                ( m, r, U.Buffer(), U.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], 
                  &qrWork[0], qrWork.size() );
                tauOffsets[teamLevel] += r;
            }
            else
            {
                Dense<Scalar>& V = SF.D;
                const int n = V.Height();
                const int r = V.Width();
                if( r <= MaxRank() )
                    break;

                lapack::QR
                ( n, r, V.Buffer(), V.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], 
                  &qrWork[0], qrWork.size() );
                tauOffsets[teamLevel] += r;
            }
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesParallelQR
( std::vector<int>& numQRs, 
  std::vector<int>& ranks, std::vector<int>& rankOffsets, 
  std::vector<Scalar>& qrBuffer, std::vector<int>& qrOffsets,
  std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& qrWork ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesParallelQR");
#endif
    // Perform the combined distributed TSQR factorizations.
    // This could almost certainly be simplified...
    const int numTeamLevels = _teams->NumLevels();
    const int numSteps = numTeamLevels-1;
    for( int step=0; step<numSteps; ++step )
    {
        MPI_Comm team = _teams->Team( step );
        const int teamSize = mpi::CommSize( team );
        const unsigned teamRank = mpi::CommRank( team );
        const bool haveAnotherComm = ( step < numSteps-1 );
        // only valid result if we have a next step...
        const bool rootOfNextStep = !(teamRank & 0x100);
        const int passes = 2*step;

        // Compute the total message size for this step
        int msgSize = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            for( int i=0; i<numQRs[l]; ++i )
            {
                const int r = ranks[rankOffsets[l]+i];     
                msgSize += (r*r+r)/2;
            }
        }
        std::vector<Scalar> sendBuffer( msgSize ), recvBuffer( msgSize );

        // Flip the first bit of our rank in this team to get our partner,
        // and then check if our bit is 0 to see if we're the root
        const unsigned firstPartner = teamRank ^ 0x1;
        const unsigned firstRoot = !(teamRank & 0x1);

        // Pack the messages for the firstPartner
        int sendOffset = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            int qrOffset = qrOffsets[l];
            MPI_Comm thisTeam = _teams->Team(l);
            const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

            for( int k=0; k<numQRs[l]; ++k )
            {
                const int r = ranks[rankOffsets[l]+k];
                if( firstRoot )
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
                else
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)+(j+1)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
                qrOffset += log2ThisTeamSize*(r*r+r);
            }
        }

        // Exchange with our first partner
        mpi::SendRecv
        ( &sendBuffer[0], msgSize, firstPartner, 0,
          &recvBuffer[0], msgSize, firstPartner, 0, team );

        if( teamSize == 4 )
        {
            // Flip the second bit of our rank in this team to get our partner,
            // and then check if our bit is 0 to see if we're the root
            const unsigned secondPartner = teamRank ^ 0x10;
            const bool secondRoot = !(teamRank & 0x10);

            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step and into the next 
            // send buffer in a single sweep.
            sendOffset = 0;
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                int qrOffset = qrOffsets[l];
                int tauOffset = tauOffsets[l];
                MPI_Comm thisTeam = _teams->Team(l);
                const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

                for( int k=0; k<numQRs[l]; ++k )
                {
                    const int r = ranks[rankOffsets[l]+k];

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[qrOffset+passes*(r*r+r)], 
                      &tauBuffer[tauOffset+(passes+1)*r], &qrWork[0] );
                    if( secondRoot )
                    {
                        // Copy into the upper triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
                    else
                    {
                        // Copy into the lower triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)+
                                        (j+1)],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
                    qrOffset += log2ThisTeamSize*(r*r+r);
                    tauOffset += (log2ThisTeamSize+1)*r;
                }
            }
            
            // Exchange with our second partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, secondPartner, 0,
              &recvBuffer[0], msgSize, secondPartner, 0, team );
            
            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step when necessary.
            recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                int qrOffset = qrOffsets[l];
                int tauOffset = tauOffsets[l];
                MPI_Comm thisTeam = _teams->Team(l);
                const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

                for( int k=0; k<numQRs[l]; ++k )
                {
                    const int r = ranks[rankOffsets[l]+k];

                    if( secondRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)+
                                        (j+1)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[qrOffset+(passes+1)*(r*r+r)], 
                      &tauBuffer[tauOffset+(passes+2)*r], &qrWork[0] );
                    if( haveAnotherComm )
                    {
                        if( rootOfNextStep )
                        {
                            // Copy into the upper triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[qrOffset+(passes+2)*(r*r+r)+
                                            (j*j+j)],
                                  &qrBuffer[qrOffset+(passes+1)*(r*r+r)+
                                            (j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                        else
                        {
                            // Copy into the lower triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[qrOffset+(passes+2)*(r*r+r)+
                                            (j*j+j)+(j+1)],
                                  &qrBuffer[qrOffset+(passes+1)*(r*r+r)+
                                            (j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                    }
                    qrOffset += log2ThisTeamSize*(r*r+r);
                    tauOffset += (log2ThisTeamSize+1)*r;
                }
            }
        }
        else // teamSize == 2
        {
            // Unpack the recv messages and perform the QR factorizations
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                int qrOffset = qrOffsets[l];
                int tauOffset = tauOffsets[l];
                MPI_Comm thisTeam = _teams->Team(l);
                const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

                for( int k=0; k<numQRs[l]; ++k )
                {
                    const int r = ranks[rankOffsets[l]+k];

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[qrOffset+passes*(r*r+r)], 
                      &tauBuffer[tauOffset+(passes+1)*r], &qrWork[0] );

                    qrOffset += log2ThisTeamSize*(r*r+r);
                    tauOffset += (log2ThisTeamSize+1)*r;
                }
            }
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
        if( _inTargetTeam && _inSourceTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        const int r = DF.rank;
        if( r <= MaxRank() )
            break;

        if( _inTargetTeam )
        {
            AddToMap( sendSizes, _sourceRoot, (r*r+r)/2 );
            AddToMap( recvSizes, _sourceRoot, (r*r+r)/2 );
        }
        else
        {
            AddToMap( sendSizes, _targetRoot, (r*r+r)/2 );
            AddToMap( recvSizes, _targetRoot, (r*r+r)/2 );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        const int r = SF.rank;
        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
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
                AddToMap( sendSizes, _sourceRoot, packedRUSize );
                AddToMap( recvSizes, _sourceRoot, packedRVSize );
            }
            else
            {
                AddToMap( sendSizes, _targetRoot, packedRVSize );
                AddToMap( recvSizes, _targetRoot, packedRUSize );
            }
        }
        else
        {
            // Count the send/recv U sizes
            if( _inTargetTeam )
                AddToMap( sendSizes, _sourceRoot, Height()*SF.rank );
            else
                AddToMap( recvSizes, _targetRoot, Height()*SF.rank );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        // Count the send/recv sizes of the U's from the low-rank updates
        if( _inTargetTeam )
            AddToMap( sendSizes, _sourceRoot, Height()*_UMap[0]->Width() );
        else
            AddToMap( recvSizes, _targetRoot, Height()*_VMap[0]->Width() );
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
( std::vector<Scalar>& sendBuffer, std::map<int,int>& sendOffsets,
  const std::vector<Scalar>& qrBuffer, std::vector<int>& qrOffsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesExchangePack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesExchangePack
                ( sendBuffer, sendOffsets, qrBuffer, qrOffsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inTargetTeam && _inSourceTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        const int r = DF.rank;
        if( r <= MaxRank() )
            break;

        MPI_Comm team = _teams->Team( _level );
        const unsigned teamLevel = _teams->TeamLevel( _level );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        const Scalar* lastQRChunk = 
            &qrBuffer[qrOffsets[teamLevel]+(log2TeamSize-1)*(r*r+r)];

        const int partner = ( _inTargetTeam ? _sourceRoot : _targetRoot );
        const int sendOffset = sendOffsets[partner];
        for( int j=0; j<r; ++j )
            std::memcpy
            ( &sendBuffer[sendOffset+(j*j+j)/2], 
              &lastQRChunk[j*j+j], (j+1)*sizeof(Scalar) );
        sendOffsets[partner] += (r*r+r)/2;
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        const int r = SF.rank;
        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            if( r <= MaxRank() )
                break;

            // Pack R
            const int minDim = std::min( SF.D.Height(), r );

            const int partner = ( _inTargetTeam ? _sourceRoot : _targetRoot );
            const int sendOffset = sendOffsets[partner];
            for( int j=0; j<minDim; ++j )
                std::memcpy
                ( &sendBuffer[sendOffset+(j*j+j)/2],
                  SF.D.LockedBuffer(0,j), (j+1)*sizeof(Scalar) );
            for( int j=minDim; j<r; ++j )
                std::memcpy
                ( &sendBuffer[sendOffset+
                              (minDim*minDim+minDim)/2+(j-minDim)*minDim],
                  SF.D.LockedBuffer(0,j), minDim*sizeof(Scalar) );
            sendOffsets[partner] += 
                (minDim*minDim+minDim)/2 + (r-minDim)*minDim;
        }
        else
        {
            if( _inTargetTeam )
            {
                const int height = Height();
                const int sendOffset = sendOffsets[_sourceRoot];
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( &sendBuffer[sendOffset+j*height],
                      SF.D.LockedBuffer(0,j), height*sizeof(Scalar) );
                sendOffsets[_sourceRoot] += height*r;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            const Dense<Scalar>& U = *_UMap[0];
            const int height = Height();
            const int r = U.Width();
            const int sendOffset = sendOffsets[_sourceRoot];
            for( int j=0; j<r; ++j )
                std::memcpy
                ( &sendBuffer[sendOffset+j*height],
                  U.LockedBuffer(0,j), height*sizeof(Scalar) );
            sendOffsets[_sourceRoot] += height*r;
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
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& recvOffsets,
  const std::vector<Scalar>& qrBuffer, std::vector<int>& qrOffsets,
  const std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  Dense<Scalar>& RU, Dense<Scalar>& RV, std::vector<Real>& singularValues, 
  Dense<Scalar>& W, 
  std::vector<Scalar>& svdWork, std::vector<Real>& svdRealWork,
  std::vector<Scalar>& applyQWork )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesExchangeFinalize");
#endif
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
                ( recvBuffer, recvOffsets, 
                  qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
                  RU, RV, singularValues, W, svdWork, svdRealWork, applyQWork );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inTargetTeam && _inSourceTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        const int r = DF.rank;
        if( r <= MaxRank() )
            break;

        MPI_Comm team = _teams->Team( _level );
        const unsigned teamLevel = _teams->TeamLevel( _level );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        const int partner = ( _inTargetTeam ? _sourceRoot : _targetRoot );

        //const Scalar* qrSection = &qrBuffer[qrOffsets[teamLevel]];
        const Scalar* lastQRChunk = 
            &qrBuffer[qrOffsets[teamLevel]+(log2TeamSize-1)*(r*r+r)];

        const int recvOffset = recvOffsets[partner];

        // TODO: Form RU and RV, then RU RV^[T/H], compute its SVD, and
        //       then backtransform either the left or right singular 
        //       vectors that we keep
        hmat_tools::Scale( (Scalar)0, RU );
        for( int j=0; j<r; ++j )
            std::memcpy
            ( RU.Buffer(0,j), &lastQRChunk[j*j+j], (j+1)*sizeof(Scalar) );
        for( int j=0; j<r; ++j )
            std::memcpy
            ( RV.Buffer(0,j), &recvBuffer[recvOffset+(j*j+j)/2], 
              (j+1)*sizeof(Scalar) );
        const char option = ( Conjugated ? 'C' : 'T' );
        // Overwrite RU with RU RV^[T/H]
        blas::Trmm
        ( 'R', 'U', option, 'N', r, r, 
          1, RV.LockedBuffer(), RV.LDim(), RU.Buffer(), RU.LDim() );
        // Perform an SVD on RU, overwriting RU with the left singular vectors 
        // and RV with the right singular vectors.
        lapack::SVD
        ( 'O', 'S', r, r, RU.Buffer(), RU.LDim(), 
          &singularValues[0], 0, 1, RV.Buffer(), RV.LDim(), 
          &svdWork[0], svdWork.size(), &svdRealWork[0] );
        // Form the compressed U and V
        // HERE

        recvOffsets[partner] += (r*r+r)/2;
        break;
    }
    case SPLIT_LOW_RANK:
    {
        /*
        const SplitLowRank& SF = *_block.data.SF;
        const int r = SF.rank;
        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            if( r <= MaxRank() )
                break;
            // TODO
        }
        else
        {
            if( _inSourceTeam )
            {
                // TODO
            }
        }
        */
        break;
    }
    case LOW_RANK:
    {
        // TODO
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
        {
            // TODO
        }
    }
    case DENSE:
    {
        // TODO
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

