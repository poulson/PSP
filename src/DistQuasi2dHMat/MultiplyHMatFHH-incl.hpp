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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPrecompute
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPrecompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if(( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    C._colFHHContextMap.Set( key, new MultiplyDenseContext );
                    MultiplyDenseContext& colContext = 
                        C._colFHHContextMap.Get( key );
                    colContext.numRhs = sampleRank;
                    C._colXMap.Set
                    ( key, new Dense<Scalar>( A.LocalHeight(), sampleRank ) );
                    Dense<Scalar>& colX = C._colXMap.Get( key );
                    hmat_tools::Scale( (Scalar)0, colX );
                    A.MultiplyDenseInitialize( colContext, sampleRank );
                    A.MultiplyDensePrecompute
                    ( colContext, alpha, B._colT, colX );

                    C._rowFHHContextMap.Set( key, new MultiplyDenseContext );
                    MultiplyDenseContext& rowContext = 
                        C._rowFHHContextMap.Get( key );
                    rowContext.numRhs = sampleRank;
                    C._rowXMap.Set
                    ( key, new Dense<Scalar>( B.LocalWidth(), sampleRank ) );
                    Dense<Scalar>& rowX = C._rowXMap.Get( key );
                    hmat_tools::Scale( (Scalar)0, rowX );
                    B.AdjointMultiplyDenseInitialize( rowContext, sampleRank );
                    B.AdjointMultiplyDensePrecompute
                    ( rowContext, Conj(alpha), A._rowT, rowX );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPrecompute
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel );
            }
            break;
        }
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSums
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSums");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each reduce
    const unsigned numTeamLevels = _teams->NumLevels();
    const unsigned numReduces = numTeamLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    A.MultiplyHMatFHHSumsCount( B, C, sizes, startLevel, endLevel );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatFHHSumsPack
    ( B, C, buffer, offsetsCopy, startLevel, endLevel );

    // Perform the reduces with log2(p) messages
    A._teams->TreeSumToRoots( buffer, sizes );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatFHHSumsUnpack( B, C, buffer, offsets, startLevel, endLevel );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<int>& sizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDenseSumsCount( sizes, sampleRank );
                    B.TransposeMultiplyDenseSumsCount( sizes, sampleRank );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel );
            }
            break;
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsPack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsPack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDenseSumsPack
                    ( C._colFHHContextMap.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseSumsPack
                    ( C._rowFHHContextMap.Get( key ), buffer, offsets );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel );
            }
            break;
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDenseSumsUnpack
                    ( C._colFHHContextMap.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseSumsUnpack
                    ( C._rowFHHContextMap.Get( key ), buffer, offsets );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel );
            }
            break;
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassData
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassData");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFHHPassDataCount
    ( B, C, sendSizes, recvSizes, startLevel, endLevel );

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
    std::vector<Scalar> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatFHHPassDataPack
    ( B, C, sendBuffer, offsets, startLevel, endLevel );

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
    A.MultiplyHMatFHHPassDataUnpack
    ( B, C, recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void 
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassDataCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
                    B.TransposeMultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendSizes, recvSizes, startLevel, endLevel );
            }
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassDataPack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataPack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDensePassDataPack
                    ( C._colFHHContextMap.Get( key ), sendBuffer, offsets );
                    B.TransposeMultiplyDensePassDataPack
                    ( C._rowFHHContextMap.Get( key ), 
                      A._rowT, sendBuffer, offsets );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendBuffer, offsets, startLevel, endLevel );
            }
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassDataUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDensePassDataUnpack
                    ( C._colFHHContextMap.Get( key ), recvBuffer, offsets );
                    B.TransposeMultiplyDensePassDataUnpack
                    ( C._rowFHHContextMap.Get( key ), recvBuffer, offsets );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              recvBuffer, offsets, startLevel, endLevel );
            }
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcasts
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcasts");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each broadcast
    const unsigned numTeamLevels = _teams->NumLevels();
    const unsigned numBroadcasts = numTeamLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    A.MultiplyHMatFHHBroadcastsCount( B, C, sizes, startLevel, endLevel );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( unsigned i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatFHHBroadcastsPack
    ( B, C, buffer, offsetsCopy, startLevel, endLevel );

    // Perform the broadcasts with log2(p) messages
    A._teams->TreeBroadcasts( buffer, sizes );

    // Unpack the broadcasted buffers
    A.MultiplyHMatFHHBroadcastsUnpack
    ( B, C, buffer, offsets, startLevel, endLevel );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<int>& sizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDenseBroadcastsCount( sizes, sampleRank );
                    B.TransposeMultiplyDenseBroadcastsCount
                    ( sizes, sampleRank );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel );
            }
            break;
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsPack
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsPack");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDenseBroadcastsPack
                    ( C._colFHHContextMap.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseBroadcastsPack
                    ( C._rowFHHContextMap.Get( key ), buffer, offsets );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel );
            }
            break;
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    A.MultiplyDenseBroadcastsUnpack
                    ( C._colFHHContextMap.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseBroadcastsUnpack
                    ( C._rowFHHContextMap.Get( key ), buffer, offsets );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel );
            }
            break;
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcompute
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A.MultiplyHMatFHHPostcomputeC( alpha, B, C, startLevel, endLevel );
    C.MultiplyHMatFHHPostcomputeCCleanup( startLevel, endLevel );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcomputeC
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcomputeC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._level >= startLevel && A._level < endLevel )
                {
                    // Finish computing A B Omega1
                    A.MultiplyDensePostcompute
                    ( C._colFHHContextMap.Get( key ), 
                      alpha, B._colT, C._colXMap.Get( key ) );

                    // Finish computing B' A' Omega2
                    B.AdjointMultiplyDensePostcompute
                    ( C._rowFHHContextMap.Get( key ), 
                      Conj(alpha), A._rowT, C._rowXMap.Get( key ) );
                }
            }
            else if( A._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPostcomputeC
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel );
            }
            break;
        }
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcomputeCCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcomputeCCleanup");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& C = *this;
    C._colFHHContextMap.Clear();
    C._rowFHHContextMap.Clear();

    switch( C._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        if( C._level >= startLevel && C._level < endLevel - 1 )
        {
            Node& nodeC = *C._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeC.Child(t,s).MultiplyHMatFHHPostcomputeCCleanup
                    ( startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalize
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalize");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    const int r = SampleRank( C.MaxRank() );
    const unsigned numTeamLevels = C._teams->NumLevels();
    std::vector<int> numQRs(numTeamLevels,0), 
                     numTargetFHH(numTeamLevels,0), 
                     numSourceFHH(numTeamLevels,0);
    C.MultiplyHMatFHHFinalizeCounts
    ( numQRs, numTargetFHH, numSourceFHH, startLevel, endLevel );

    // Set up the space for the packed 2r x r matrices and taus.
    int numTotalQRs=0, numQRSteps=0, qrTotalSize=0, tauTotalSize=0;
    std::vector<int> XOffsets(numTeamLevels), halfHeightOffsets(numTeamLevels),
                     qrOffsets(numTeamLevels), tauOffsets(numTeamLevels);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = C._teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        XOffsets[teamLevel] = numTotalQRs;
        halfHeightOffsets[teamLevel] = 2*numQRSteps;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        numTotalQRs += numQRs[teamLevel];
        numQRSteps += numQRs[teamLevel]*log2TeamSize;
        qrTotalSize += numQRs[teamLevel]*log2TeamSize*(r*r+r);
        tauTotalSize += numQRs[teamLevel]*(log2TeamSize+1)*r;
    }

    std::vector<Dense<Scalar>*> Xs( numTotalQRs );
    std::vector<int> halfHeights( 2*numQRSteps );
    std::vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize ),
                        qrWork( lapack::QRWorkSize( r ) );

    // Form our contributions to Omega2' (alpha A B Omega1) updates here, 
    // before we overwrite the _colXMap and _rowXMap results.
    // The distributed summations will not occur until after the parallel
    // QR factorizations.
    std::vector<int> leftOffsets(numTeamLevels), middleOffsets(numTeamLevels), 
                     rightOffsets(numTeamLevels);
    int totalAllReduceSize = 0;
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        leftOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numTargetFHH[teamLevel]*r*r;

        middleOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numTargetFHH[teamLevel]*r*r;

        rightOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numSourceFHH[teamLevel]*r*r;
    }

    // Compute the local contributions to the middle updates, 
    // Omega2' (alpha A B Omega1)
    std::vector<Scalar> allReduceBuffer( totalAllReduceSize );
    {
        std::vector<int> middleOffsetsCopy = middleOffsets;

        A.MultiplyHMatFHHFinalizeMiddleUpdates
        ( B, C, allReduceBuffer, middleOffsetsCopy, startLevel, endLevel );
    }

    // Perform the large local QR's and pack into the QR buffer as appropriate
    {
        std::vector<int> XOffsetsCopy, tauOffsetsCopy;
        XOffsetsCopy = XOffsets;
        tauOffsetsCopy = tauOffsets;

        C.MultiplyHMatFHHFinalizeLocalQR
        ( Xs, XOffsetsCopy, tauBuffer, tauOffsetsCopy, qrWork, 
          startLevel, endLevel );
    }

    C.MultiplyHMatParallelQR
    ( numQRs, Xs, XOffsets, halfHeights, halfHeightOffsets, 
      qrBuffer, qrOffsets, tauBuffer, tauOffsets, qrWork );

    // Explicitly form the Q's
    Dense<Scalar> Z( 2*r, r );
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = C._teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );
        const unsigned teamRank = mpi::CommRank( team );

        Dense<Scalar>** XLevel = &Xs[XOffsets[teamLevel]];
        const int* halfHeightsLevel = 
            &halfHeights[halfHeightOffsets[teamLevel]];
        const Scalar* qrLevel = &qrBuffer[qrOffsets[teamLevel]];
        const Scalar* tauLevel = &tauBuffer[tauOffsets[teamLevel]];

        for( int k=0; k<numQRs[teamLevel]; ++k )
        {
            Dense<Scalar>& X = *XLevel[k];
            const int* halfHeightsPiece = &halfHeightsLevel[k*log2TeamSize*2];
            const Scalar* qrPiece = &qrLevel[k*log2TeamSize*(r*r+r)];
            const Scalar* tauPiece = &tauLevel[k*(log2TeamSize+1)*r];

           if( log2TeamSize > 0 )
           {
                const int* lastHalfHeightsStage = 
                    &halfHeightsPiece[(log2TeamSize-1)*2];
                const Scalar* lastQRStage = &qrPiece[(log2TeamSize-1)*(r*r+r)];
                const Scalar* lastTauStage = &tauPiece[log2TeamSize*r];
 
                const int sLast = lastHalfHeightsStage[0];
                const int tLast = lastHalfHeightsStage[1];

                // Form the identity matrix in the top r x r submatrix
                // of a zeros (sLast+tLast) x r matrix.
                Z.Resize( sLast+tLast, r );
                hmat_tools::Scale( (Scalar)0, Z );
                for( int j=0; j<std::min(sLast+tLast,r); ++j )
                    Z.Set(j,j,(Scalar)1 );

                // Backtransform the last stage
                qrWork.resize( r );
                hmat_tools::ApplyPackedQFromLeft
                ( r, sLast, tLast, lastQRStage, lastTauStage, Z, &qrWork[0] );

                // Take care of the middle stages before handling the large 
                // original stage.
                int sPrev=sLast, tPrev=tLast;
                for( int commStage=log2TeamSize-2; commStage>=0; --commStage )
                {
                    const int sCurr = halfHeightsPiece[commStage*2];
                    const int tCurr = halfHeightsPiece[commStage*2+1];
                    Z.Resize( sCurr+tCurr, r );

                    const bool rootOfPrevStage = 
                        !(teamRank & (1u<<(commStage+1)));
                    if( rootOfPrevStage )
                    {
                        // Zero the bottom half of Z
                        for( int j=0; j<r; ++j )
                            std::memset
                            ( Z.Buffer(sCurr,j), 0, tCurr*sizeof(Scalar) );
                    }
                    else
                    {
                        // Move the bottom part to the top part and zero the
                        // bottom 
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( Z.Buffer(0,j), Z.LockedBuffer(sCurr,j), 
                              tCurr*sizeof(Scalar) );
                            std::memset
                            ( Z.Buffer(sCurr,j), 0, tCurr*sizeof(Scalar) );
                        }
                    }
                    hmat_tools::ApplyPackedQFromLeft
                    ( r, sCurr, tCurr, &qrPiece[commStage*(r*r+r)], 
                      &tauPiece[(commStage+1)*r], Z, &qrWork[0] );

                    sPrev = sCurr;
                    tPrev = tCurr;
                }

                // Take care of the original stage. Do so by forming Y := X, 
                // then zeroing X and placing our piece of Z at its top.
                const int m = X.Height();
                Dense<Scalar> Y;
                hmat_tools::Copy( X, Y );
                hmat_tools::Scale( (Scalar)0, X );
                const bool rootOfPrevStage = !(teamRank & 0x1);
                if( rootOfPrevStage )
                {
                    // Copy the first sPrev rows of the top half of Z into
                    // the top of X
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( X.Buffer(0,j), Z.LockedBuffer(0,j), 
                          sPrev*sizeof(Scalar) );
                }
                else
                {
                    // Copy the first tPrev rows of the bottom part of Z into 
                    // the top of X
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( X.Buffer(0,j), Z.LockedBuffer(sPrev,j), 
                          tPrev*sizeof(Scalar) );
                }
                qrWork.resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, std::min(m,r),
                  Y.LockedBuffer(), Y.LDim(), &tauPiece[0],
                  X.Buffer(),       X.LDim(), &qrWork[0], qrWork.size() );
            }
            else // this team only contains one process
            {
                // Make a copy of X and then form the left part of identity.
                const int m = X.Height();
                Dense<Scalar> Y; 
                hmat_tools::Copy( X, Y );
                hmat_tools::Scale( (Scalar)0, X );
                for( int j=0; j<std::min(m,r); ++j )
                    X.Set(j,j,(Scalar)1);
                // Backtransform the last stage
                qrWork.resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, std::min(m,r),
                  Y.LockedBuffer(), Y.LDim(), &tauPiece[0],
                  X.Buffer(),       X.LDim(), &qrWork[0], qrWork.size() );
            }
        }
    }
    XOffsets.clear();
    qrOffsets.clear(); qrBuffer.clear();
    tauOffsets.clear(); tauBuffer.clear();
    qrWork.clear();
    Z.Clear();

    // Form our local contributions to Q1' Omega2 and Q2' Omega1
    {
        std::vector<int> leftOffsetsCopy, rightOffsetsCopy;
        leftOffsetsCopy = leftOffsets;
        rightOffsetsCopy = rightOffsets;

        A.MultiplyHMatFHHFinalizeOuterUpdates
        ( B, C, allReduceBuffer, leftOffsetsCopy, rightOffsetsCopy,
          startLevel, endLevel );
    }

    // Perform a custom AllReduce on the buffers to finish forming
    // Q1' Omega2, Omega2' (alpha A B Omega1), and Q2' Omega1
    {
        // Generate offsets and sizes for each entire level
        const unsigned numAllReduces = numTeamLevels-1;
        std::vector<int> sizes(numAllReduces);
        for( unsigned teamLevel=0; teamLevel<numAllReduces; ++teamLevel )
            sizes[teamLevel] = (2*numTargetFHH[teamLevel]+
                                  numSourceFHH[teamLevel])*r*r;

        A._teams->TreeSums( allReduceBuffer, sizes );
    }

    // Finish forming the low-rank approximation
    std::vector<Scalar> U( r*r ), VH( r*r ),
                        svdWork( lapack::SVDWorkSize(r,r) );
    std::vector<Real> singularValues( r ), 
                      svdRealWork( lapack::SVDRealWorkSize(r,r) ); 
    A.MultiplyHMatFHHFinalizeFormLowRank
    ( B, C, allReduceBuffer, leftOffsets, middleOffsets, rightOffsets,
      singularValues, U, VH, svdWork, svdRealWork, startLevel, endLevel );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeCounts
( std::vector<int>& numQRs, 
  std::vector<int>& numTargetFHH, std::vector<int>& numSourceFHH,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeCounts");
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
        if( _level < endLevel - 1 )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHFinalizeCounts
                    ( numQRs, numTargetFHH, numSourceFHH, 
                      startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
        // TODO: Think about avoiding the expensive F += H H proceduce in the 
        //       case where there is already a dense update. We could simply
        //       add H H onto the dense update.
        if( _level >= startLevel && _level < endLevel )
        {
            if( _inTargetTeam )
            {
                const unsigned teamLevel = _teams->TeamLevel(_level);
                numQRs[teamLevel] += _colXMap.Size();
                numTargetFHH[teamLevel] += _colXMap.Size();
            }
            if( _inSourceTeam )
            {
                const unsigned teamLevel = _teams->TeamLevel(_level);
                numQRs[teamLevel] += _rowXMap.Size();
                numSourceFHH[teamLevel] += _rowXMap.Size();
            }
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeMiddleUpdates
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<Scalar>& allReduceBuffer,
        std::vector<int>& middleOffsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeMiddleUpdates");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    const int rank = SampleRank( C.MaxRank() );
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C._inTargetTeam && 
                    C._level >= startLevel && C._level < endLevel ) 
                {
                    // Handle the middle update, Omega2' (alpha A B Omega1)
                    const int key = A._sourceOffset;
                    const Dense<Scalar>& X = C._colXMap.Get( key );
                    const Dense<Scalar>& Omega2 = A._rowOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* middleUpdate = 
                        &allReduceBuffer[middleOffsets[teamLevel]];
                    blas::Gemm
                    ( 'C', 'N', rank, rank, X.Height(),
                      (Scalar)1, Omega2.LockedBuffer(), Omega2.LDim(),
                                 X.LockedBuffer(),      X.LDim(),
                      (Scalar)0, middleUpdate,          rank );
                    middleOffsets[teamLevel] += rank*rank;
                }
            }
            else if( C._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHFinalizeMiddleUpdates
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              allReduceBuffer, middleOffsets, 
                              startLevel, endLevel );
            }
            break;
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeLocalQR
( std::vector<Dense<Scalar>*>& Xs, std::vector<int>& XOffsets,
  std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& qrWork,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeLocalQR");
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
        if( _level < endLevel - 1 )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHFinalizeLocalQR
                    ( Xs, XOffsets, tauBuffer, tauOffsets, qrWork,
                      startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( _level >= startLevel && _level < endLevel )
        {
            MPI_Comm team = _teams->Team( _level );
            const unsigned teamLevel = _teams->TeamLevel(_level);
            const int log2TeamSize = Log2( mpi::CommSize( team ) );
            const int r = SampleRank( MaxRank() );

            if( _inTargetTeam )
            {
                _colXMap.ResetIterator();
                const unsigned numEntries = _colXMap.Size();
                for( unsigned i=0; i<numEntries; ++i,_colXMap.Increment() )
                {
                    Dense<Scalar>& X = *_colXMap.CurrentEntry();
                    Xs[XOffsets[teamLevel]++] = &X;

                    lapack::QR
                    ( X.Height(), X.Width(), X.Buffer(), X.LDim(), 
                      &tauBuffer[tauOffsets[teamLevel]], 
                      &qrWork[0], qrWork.size() );
                    tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                }
            }
            if( _inSourceTeam )
            {
                _rowXMap.ResetIterator();
                const int numEntries = _rowXMap.Size();
                for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
                {
                    Dense<Scalar>& X = *_rowXMap.CurrentEntry();
                    Xs[XOffsets[teamLevel]++] = &X;
                
                    lapack::QR
                    ( X.Height(), X.Width(), X.Buffer(), X.LDim(), 
                      &tauBuffer[tauOffsets[teamLevel]], 
                      &qrWork[0], qrWork.size() );
                    tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                }
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeOuterUpdates
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<Scalar>& allReduceBuffer,
        std::vector<int>& leftOffsets,
        std::vector<int>& rightOffsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeOuterUpdates");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    const int rank = SampleRank( C.MaxRank() );
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C._level >= startLevel && C._level < endLevel )
                {
                    const int key = A._sourceOffset;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    if( C._inTargetTeam ) 
                    {
                        // Handle the left update, Q1' Omega2
                        const Dense<Scalar>& Q1 = C._colXMap.Get( key );
                        const Dense<Scalar>& Omega2 = A._rowOmega;
                        Scalar* leftUpdate = 
                            &allReduceBuffer[leftOffsets[teamLevel]];
                        blas::Gemm
                        ( 'C', 'N', Q1.Width(), rank, A.LocalHeight(),
                          (Scalar)1, Q1.LockedBuffer(),     Q1.LDim(),
                                     Omega2.LockedBuffer(), Omega2.LDim(),
                          (Scalar)0, leftUpdate,            rank );
                        leftOffsets[teamLevel] += rank*rank;
                    }
                    if( C._inSourceTeam )
                    {
                        // Handle the right update, Q2' Omega1
                        const Dense<Scalar>& Q2 = C._rowXMap.Get( key );
                        const Dense<Scalar>& Omega1 = B._colOmega;
                        Scalar* rightUpdate = 
                            &allReduceBuffer[rightOffsets[teamLevel]];

                        blas::Gemm
                        ( 'C', 'N', Q2.Width(), rank, B.LocalWidth(),
                          (Scalar)1, Q2.LockedBuffer(),     Q2.LDim(),
                                     Omega1.LockedBuffer(), Omega1.LDim(),
                          (Scalar)0, rightUpdate,           rank );
                        rightOffsets[teamLevel] += rank*rank;
                    }
                }
            }
            else if( C._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHFinalizeOuterUpdates
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              allReduceBuffer, leftOffsets, rightOffsets,
                              startLevel, endLevel );
            }
            break;
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeFormLowRank
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<Scalar>& allReduceBuffer,
        std::vector<int>& leftOffsets,
        std::vector<int>& middleOffsets,
        std::vector<int>& rightOffsets,
        std::vector<Real>& singularValues,
        std::vector<Scalar>& U,
        std::vector<Scalar>& VH,
        std::vector<Scalar>& svdWork,
        std::vector<Real>& svdRealWork,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeFormLowRank");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    const int rank = SampleRank( C.MaxRank() );
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C._level >= startLevel && C._level < endLevel )
                {
                    const int key = A._sourceOffset;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    if( C._inTargetTeam ) 
                    {
                        // Form Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                        // in the place of X.
                        Dense<Scalar>& X = C._colXMap.Get( key );
                        Scalar* leftUpdate = 
                            &allReduceBuffer[leftOffsets[teamLevel]];
                        const Scalar* middleUpdate = 
                            &allReduceBuffer[middleOffsets[teamLevel]];

                        lapack::AdjointPseudoInverse
                        ( X.Width(), rank, leftUpdate, rank, &singularValues[0],
                          &U[0], rank, &VH[0], rank, &svdWork[0], 
                          svdWork.size(), &svdRealWork[0] );

                        // We can use the VH space to hold the product 
                        // pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                        blas::Gemm
                        ( 'N', 'N', X.Width(), rank, rank, 
                          (Scalar)1, leftUpdate,   rank, 
                                     middleUpdate, rank, 
                          (Scalar)0, &VH[0],       rank );

                        // Q1 := X.
                        Dense<Scalar> Q1;
                        hmat_tools::Copy( X, Q1 );
                        X.Resize( X.Height(), rank );

                        // Form 
                        // X := Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                        blas::Gemm
                        ( 'N', 'N', Q1.Height(), rank, Q1.Width(),
                          (Scalar)1, Q1.LockedBuffer(), Q1.LDim(),
                                     &VH[0],            rank, 
                          (Scalar)0, X.Buffer(),        X.LDim() );

                        leftOffsets[teamLevel] += rank*rank;
                        middleOffsets[teamLevel] += rank*rank;
                    }
                    if( C._inSourceTeam )
                    {
                        // Form Q2 pinv(Q2' Omega1)' or its conjugate
                        Dense<Scalar>& X = C._rowXMap.Get( key );
                        Scalar* rightUpdate = 
                            &allReduceBuffer[rightOffsets[teamLevel]];

                        lapack::AdjointPseudoInverse
                        ( X.Width(), rank, rightUpdate, rank, 
                          &singularValues[0],
                          &U[0], rank, &VH[0], rank, 
                          &svdWork[0], svdWork.size(), &svdRealWork[0] );

                        // Q2 := X
                        Dense<Scalar> Q2;
                        hmat_tools::Copy( X, Q2 );
                        X.Resize( X.Height(), rank );

                        blas::Gemm
                        ( 'N', 'N', Q2.Height(), rank, Q2.Width(),
                          (Scalar)1, Q2.LockedBuffer(), Q2.LDim(),
                                     rightUpdate,       rank,
                          (Scalar)0, X.Buffer(),        X.LDim() );
                        if( !Conjugated )
                            hmat_tools::Conjugate( X );

                        rightOffsets[teamLevel] += rank*rank;
                    }
                }
            }
            else if( C._level < endLevel - 1 )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHFinalizeFormLowRank
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              allReduceBuffer,
                              leftOffsets, middleOffsets, rightOffsets, 
                              singularValues, U, VH, svdWork, svdRealWork,
                              startLevel, endLevel );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

