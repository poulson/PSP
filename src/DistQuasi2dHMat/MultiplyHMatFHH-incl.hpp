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
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPrecompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inSourceTeam || A._inTargetTeam )
                {
                    C._colFHHContextMap[key] = new MultiplyDenseContext;
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    context.numRhs = sampleRank;
                    if( A._inTargetTeam )
                    {
                        C._colXMap[key] = 
                            new Dense<Scalar>( A.LocalHeight(), sampleRank );
                        Dense<Scalar>& X = *C._colXMap[key];
                        hmat_tools::Scale( (Scalar)0, X );
                    }
                    if( A._inSourceTeam && A._inTargetTeam )
                    {
                        Dense<Scalar>& X = *C._colXMap[key];
                        A.MultiplyDensePrecompute( context, alpha, B._colT, X );
                    }
                    else if( A._inSourceTeam )
                    {
                        Dense<Scalar> dummy( 0, sampleRank );
                        A.MultiplyDensePrecompute
                        ( context, alpha, B._colT, dummy );
                    }
                    else // A._inTargetTeam
                    {
                        Dense<Scalar>& X = *C._colXMap[key];
                        Dense<Scalar> dummy( 0, sampleRank );
                        A.MultiplyDensePrecompute( context, alpha, dummy, X );
                    }
                }
                if( B._inSourceTeam || B._inTargetTeam )
                {
                    C._rowFHHContextMap[key] = new MultiplyDenseContext;
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    context.numRhs = sampleRank;
                    if( B._inSourceTeam )
                    {
                        C._rowXMap[key] = 
                            new Dense<Scalar>( B.LocalWidth(), sampleRank );
                        Dense<Scalar>& X = *C._rowXMap[key];
                        hmat_tools::Scale( (Scalar)0, X );
                    }
                    if( B._inSourceTeam && B._inTargetTeam )
                    {
                        Dense<Scalar>& X = *C._rowXMap[key];
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), A._rowT, X );
                    }
                    else if( B._inTargetTeam )
                    {
                        Dense<Scalar> dummy( 0, sampleRank );
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), A._rowT, dummy );
                    }
                    else // B._inSourceTeam
                    {
                        Dense<Scalar>& X = *C._rowXMap[key];
                        Dense<Scalar> dummy( 0, sampleRank );
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), dummy, X );
                    }
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPrecompute
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
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
                DistQuasi2dHMat<Scalar,Conjugated>& C )
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
    A.MultiplyHMatFHHSumsCount( B, C, sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatFHHSumsPack( B, C, buffer, offsetsCopy );

    // Perform the reduces with log2(p) messages
    A._teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatFHHSumsUnpack( B, C, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inSourceTeam )
                    A.MultiplyDenseSumsCount( sizes, sampleRank );
                if( B._inTargetTeam )
                    B.TransposeMultiplyDenseSumsCount
                    ( sizes, sampleRank );
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
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
  std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsPack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseSumsPack( context, buffer, offsets );
                }
                if( B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseSumsPack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
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
  const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseSumsUnpack( context, buffer, offsets );
                }
                if( B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseSumsUnpack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
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
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassData");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFHHPassDataCount( B, C, sendSizes, recvSizes );

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
    A.MultiplyHMatFHHPassDataPack( B, C, sendBuffer, offsets );

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
    A.MultiplyHMatFHHPassDataUnpack( B, C, recvBuffer, recvOffsets );

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
        std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                    A.MultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
                if( B._inSourceTeam || B._inTargetTeam )
                    B.TransposeMultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendSizes, recvSizes );
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
  std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataPack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

    const int key = A._sourceOffset;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDensePassDataPack
                    ( context, sendBuffer, offsets );
                }

                if( B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDensePassDataPack
                    ( context, A._rowT, sendBuffer, offsets );
                }
                else if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    Dense<Scalar> dummy( 0, sampleRank );
                    B.TransposeMultiplyDensePassDataPack
                    ( context, dummy, sendBuffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendBuffer, offsets );
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
  const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDensePassDataUnpack
                    ( context, recvBuffer, offsets );
                }
                if( B._inSourceTeam || B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDensePassDataUnpack
                    ( context, recvBuffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              recvBuffer, offsets );
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
                DistQuasi2dHMat<Scalar,Conjugated>& C )
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
    A.MultiplyHMatFHHBroadcastsCount( B, C, sizes );

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
    A.MultiplyHMatFHHBroadcastsPack( B, C, buffer, offsetsCopy );

    // Perform the broadcasts with log2(p) messages
    A._teams->TreeBroadcasts( buffer, sizes, offsets );

    // Unpack the broadcasted buffers
    A.MultiplyHMatFHHBroadcastsUnpack( B, C, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inTargetTeam )
                    A.MultiplyDenseBroadcastsCount( sizes, sampleRank );
                if( B._inSourceTeam )
                    B.TransposeMultiplyDenseBroadcastsCount
                    ( sizes, sampleRank );
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
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
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsPack");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseBroadcastsPack( context, buffer, offsets );
                }
                if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseBroadcastsPack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
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
  const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                if( A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseBroadcastsUnpack( context, buffer, offsets );
                }
                if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseBroadcastsUnpack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
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
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A.MultiplyHMatFHHPostcomputeC( alpha, B, C );
    C.MultiplyHMatFHHPostcomputeCCleanup();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcomputeC
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcomputeC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

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
                // Finish computing A B Omega1
                if( A._inSourceTeam && A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    Dense<Scalar>& X = *C._colXMap[key];
                    A.MultiplyDensePostcompute( context, alpha, B._colT, X );
                }
                else if( A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    Dense<Scalar>& X = *C._colXMap[key];
                    Dense<Scalar> dummy( 0, sampleRank );
                    A.MultiplyDensePostcompute( context, alpha, dummy, X );
                }

                // Finish computing B' A' Omega2
                if( B._inSourceTeam && B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    Dense<Scalar>& X = *C._rowXMap[key];
                    B.AdjointMultiplyDensePostcompute
                    ( context, Conj(alpha), A._rowT, X );
                }
                else if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    Dense<Scalar>& X = *C._rowXMap[key];
                    Dense<Scalar> dummy( 0, sampleRank );
                    B.AdjointMultiplyDensePostcompute
                    ( context, Conj(alpha), dummy, X );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPostcomputeC
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcomputeCCleanup()
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
        Node& nodeC = *C._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeC.Child(t,s).MultiplyHMatFHHPostcomputeCCleanup();
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
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
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
    C.MultiplyHMatFHHFinalizeCounts( numQRs, numTargetFHH, numSourceFHH );

    // Set up the space for the packed 2r x r matrices and taus.
    int numTotalQRs=0, qrTotalSize=0, tauTotalSize=0;
    std::vector<int> XOffsets(numTeamLevels),
                     qrOffsets(numTeamLevels), qrPieceSizes(numTeamLevels), 
                     tauOffsets(numTeamLevels), tauPieceSizes(numTeamLevels);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = C._teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        XOffsets[teamLevel] = numTotalQRs;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        qrPieceSizes[teamLevel] = log2TeamSize*(r*r+r);
        tauPieceSizes[teamLevel] = (log2TeamSize+1)*r;

        numTotalQRs += numQRs[teamLevel];
        qrTotalSize += numQRs[teamLevel]*qrPieceSizes[teamLevel];
        tauTotalSize += numQRs[teamLevel]*tauPieceSizes[teamLevel];
    }

    std::vector<Dense<Scalar>*> Xs( numTotalQRs );
    std::vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize ),
                        work( lapack::QRWorkSize( r ) );

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
    std::vector<Scalar> allReduceBuffer( totalAllReduceSize );
    {
        std::vector<int> middleOffsetsCopy = middleOffsets;

        A.MultiplyHMatFHHFinalizeMiddleUpdates
        ( B, C, allReduceBuffer, middleOffsetsCopy );
    }

    // Perform the large local QR's and pack into the QR buffer as appropriate
    {
        std::vector<int> XOffsetsCopy, qrOffsetsCopy, tauOffsetsCopy;
        XOffsetsCopy = XOffsets;
        qrOffsetsCopy = qrOffsets;
        tauOffsetsCopy = tauOffsets;

        C.MultiplyHMatFHHFinalizeLocalQR
        ( Xs, XOffsetsCopy, 
          qrBuffer, qrOffsetsCopy, tauBuffer, tauOffsetsCopy, work );
    }

    // TODO: Push this into a subroutine
    //
    // Perform the combined distributed TSQR factorizations.
    // This could almost certainly be simplified...
    const int numSteps = numTeamLevels-1;
    for( int step=0; step<numSteps; ++step )
    {
        MPI_Comm team = C._teams->Team( step );
        const int teamSize = mpi::CommSize( team );
        const unsigned teamRank = mpi::CommRank( team );
        const bool haveAnotherComm = ( step < numSteps-1 );
        // only valid result if we have a next step...
        const bool rootOfNextStep = !(teamRank & 0x100); 
        const int passes = 2*step;

        // Flip the first bit of our rank in this team to get our partner,
        // and then check if our bit is 0 to see if we're the root
        const unsigned firstPartner = teamRank ^ 0x1;
        const bool firstRoot = !(teamRank & 0x1);

        // Count the messages to send/recv to/from firstPartner
        int msgSize = 0;
        for( int j=0; j<numSteps-step; ++j )
            msgSize += numQRs[j]*(r*r+r)/2;
        std::vector<Scalar> sendBuffer( msgSize ), recvBuffer( msgSize );

        // Pack the messages for the firstPartner
        int sendOffset = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            for( int k=0; k<numQRs[l]; ++k )
            {
                if( firstRoot )
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffsets[l]+k*qrPieceSizes[l]+
                                    passes*(r*r+r)+(j*j+j)],
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
                          &qrBuffer[qrOffsets[l]+k*qrPieceSizes[l]+
                                    passes*(r*r+r)+(j*j+j)+(j+1)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
            }
        }

        // Exchange with our first partner
        mpi::SendRecv
        ( &sendBuffer[0], msgSize, firstPartner, 0,
          &recvBuffer[0], msgSize, firstPartner, 0, team );

        if( teamSize == 4 )
        {
            // Flip the second bit in our team rank to get our partner, and
            // then check if our bit is 0 to see if we're the root
            const unsigned secondPartner = teamRank ^ 0x10;
            const bool secondRoot = !(teamRank & 0x10);

            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step and into the next 
            // send buffer in a single sweep.
            sendOffset = 0;
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const unsigned thisQROffset = 
                        qrOffsets[l]+k*qrPieceSizes[l]+passes*(r*r+r);
                    const unsigned thisTauOffset = 
                        tauOffsets[l]+k*tauPieceSizes[l]+(passes+1)*r;

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)+(j+1)],
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
                            ( &qrBuffer[thisQROffset+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[thisQROffset], &tauBuffer[thisTauOffset],
                      &work[0] );
                    if( secondRoot )
                    {
                        // Copy into the upper triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[thisQROffset+(j*j+j)],
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
                            ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)+
                                        (j+1)],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
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
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const unsigned thisQROffset = 
                        qrOffsets[l]+k*qrPieceSizes[l]+(passes+1)*(r*r+r);
                    const unsigned thisTauOffset = 
                        tauOffsets[l]+k*tauPieceSizes[l]+(passes+2)*r;

                    if( secondRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)+(j+1)],
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
                            ( &qrBuffer[thisQROffset+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }

                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[thisQROffset], &tauBuffer[thisTauOffset],
                      &work[0] );
                    if( haveAnotherComm )
                    {
                        if( rootOfNextStep )
                        {
                            // Copy into the upper triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)],
                                  &qrBuffer[thisQROffset+(j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                        else
                        {
                            // Copy into the lower triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)+(j+1)],
                                  &qrBuffer[thisQROffset+(j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                    }
                }
            }
        }
        else // teamSize == 2
        {
            // Exchange with our partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, firstPartner, 0,
              &recvBuffer[0], msgSize, firstPartner, 0, team );

            // Unpack the recv messages and perform the QR factorizations
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const unsigned thisQROffset = 
                        qrOffsets[l]+k*qrPieceSizes[l]+passes*(r*r+r);
                    const unsigned thisTauOffset = 
                        tauOffsets[l]+k*tauPieceSizes[l]+(passes+1)*r;

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)+(j+1)],
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
                            ( &qrBuffer[thisQROffset+(j*j+j)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[thisQROffset], &tauBuffer[thisTauOffset],
                      &work[0] );
                }
            }
        }
    }

    // Explicitly form the Q's
    Dense<Scalar> Z( 2*r, r );
    std::vector<Scalar> applyQWork( r, 1 );
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = C._teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );
        const unsigned teamRank = mpi::CommRank( team );
        const Scalar* thisQRLevel = &qrBuffer[qrOffsets[teamLevel]];
        const Scalar* thisTauLevel = &tauBuffer[tauOffsets[teamLevel]];

        for( int k=0; k<numQRs[teamLevel]; ++k )
        {
            const Scalar* thisQRPiece = &thisQRLevel[k*qrPieceSizes[teamLevel]];
            const Scalar* thisTauPiece = 
                &thisTauLevel[k*tauPieceSizes[teamLevel]];
            const Scalar* lastQRStage = &thisQRPiece[(log2TeamSize-1)*(r*r+r)];
            const Scalar* lastTauStage = &thisTauPiece[log2TeamSize*r];

            if( log2TeamSize > 0 )
            {
                // Form the identity matrix in the top r x r submatrix
                // of a zeroed 2r x r matrix.
                std::memset( Z.Buffer(), 0, 2*r*r*sizeof(Scalar) );
                for( int j=0; j<r; ++j )
                    Z.Set(j,j,(Scalar)1);
                // Backtransform the last stage
                hmat_tools::ApplyPackedQFromLeft
                ( r, lastQRStage, lastTauStage, Z, &work[0] );
                // Take care of the middle stages before handling the large 
                // original stage.
                for( int commStage=log2TeamSize-2; commStage>=0; --commStage )
                {
                    const bool rootOfLastStage = 
                        !(teamRank & (1u<<(commStage+1)));
                    if( rootOfLastStage )
                    {
                        // Zero the bottom half of Z
                        for( int j=0; j<r; ++j )
                            std::memset( Z.Buffer(r,j), 0, r*sizeof(Scalar) );
                    }
                    else
                    {
                        // Move the bottom half to the top half and zero the
                        // bottom half
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( Z.Buffer(0,j), Z.LockedBuffer(r,j), 
                              r*sizeof(Scalar) );
                            std::memset( Z.Buffer(r,j), 0, r*sizeof(Scalar) );
                        }
                    }
                    hmat_tools::ApplyPackedQFromLeft
                    ( r, &thisQRPiece[commStage*(r*r+r)], 
                      &thisTauPiece[(commStage+1)*r], Z, &work[0] );
                }
                // Take care of the original stage. Do so by forming Y := X, 
                // then zeroing X and placing our piece of Z at its top.
                Dense<Scalar>& X = *Xs[XOffsets[teamLevel]+k]; 
                const int m = X.Height();
                const int minDim = std::min( m, r );
                Dense<Scalar> Y;
                hmat_tools::Copy( X, Y );
                hmat_tools::Scale( (Scalar)0, X );
                const bool rootOfLastStage = !(teamRank & 0x1);
                if( rootOfLastStage )
                {
                    // Copy the first minDim rows of the top half of Z into
                    // the top of X
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( X.Buffer(0,j), Z.LockedBuffer(0,j), 
                          minDim*sizeof(Scalar) );
                }
                else
                {
                    // Copy the first minDim rows of the bottom half of Z into 
                    // the top of X
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( X.Buffer(0,j), Z.LockedBuffer(r,j), 
                          minDim*sizeof(Scalar) );
                }
                work.resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, minDim,
                  Y.LockedBuffer(), Y.LDim(), &thisTauPiece[0],
                  X.Buffer(),       X.LDim(), &work[0], work.size() );
            }
            else // this team only contains one process
            {
                Dense<Scalar>& X = *Xs[XOffsets[teamLevel]+k]; 
                const int m = X.Height();
                const int minDim = std::min(m,r);

                // Make a copy of X and then form the left part of identity.
                Dense<Scalar> Y; 
                Y = X;
                hmat_tools::Scale( (Scalar)0, X );
                for( int j=0; j<minDim; ++j )
                    X.Set(j,j,(Scalar)1);
                // Backtransform the last stage
                work.resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, minDim,
                  Y.LockedBuffer(), Y.LDim(), &thisTauPiece[0],
                  X.Buffer(),       X.LDim(), &work[0], work.size() );
            }
        }
    }
    XOffsets.clear();
    qrOffsets.clear(); qrPieceSizes.clear(); qrBuffer.clear();
    tauOffsets.clear(); tauPieceSizes.clear(); tauBuffer.clear();
    work.clear();
    Z.Clear();
    applyQWork.clear();

    // Form our local contributions to Q1' Omega2 and Q2' Omega1
    {
        std::vector<int> leftOffsetsCopy, rightOffsetsCopy;
        leftOffsetsCopy = leftOffsets;
        rightOffsetsCopy = rightOffsets;

        A.MultiplyHMatFHHFinalizeOuterUpdates
        ( B, C, allReduceBuffer, leftOffsetsCopy, rightOffsetsCopy );
    }

    // Perform a custom AllReduce on the buffers to finish forming
    // Q1' Omega2, Omega2' (alpha A B Omega1), and Q2' Omega1
    {
        // Generate offsets and sizes for each entire level
        std::vector<int> sizes, offsets;
        totalAllReduceSize = 0;
        for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
        {
            offsets[teamLevel] = totalAllReduceSize;
            sizes[teamLevel] = r*r*(2*numTargetFHH[teamLevel]+
                                      numSourceFHH[teamLevel]);

            totalAllReduceSize += 2*numTargetFHH[teamLevel]*r*r;
            totalAllReduceSize += numSourceFHH[teamLevel]*r*r;
        }

        A._teams->TreeSums( allReduceBuffer, sizes, offsets );
    }

    // Finish forming the low-rank approximation
    std::vector<Scalar> U( r*r ), VH( r*r ),
                        svdWork( lapack::SVDWorkSize(r,r) );
    std::vector<Real> singularValues( r ), 
                      svdRealWork( lapack::SVDRealWorkSize(r,r) ); 
    A.MultiplyHMatFHHFinalizeFormLowRank
    ( B, C, allReduceBuffer, leftOffsets, middleOffsets, rightOffsets,
      singularValues, U, VH, svdWork, svdRealWork );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeCounts
( std::vector<int>& numQRs, 
  std::vector<int>& numTargetFHH, std::vector<int>& numSourceFHH )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeCounts");
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
                node.Child(t,s).MultiplyHMatFHHFinalizeCounts
                ( numQRs, numTargetFHH, numSourceFHH );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
        if( _inTargetTeam )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            numQRs[teamLevel] += _colXMap.Size();
            ++numTargetFHH[teamLevel];
        }
        if( _inSourceTeam )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            numQRs[teamLevel] += _rowXMap.Size();
            ++numSourceFHH[teamLevel];
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
        std::vector<int>& middleOffsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeMiddleUpdates");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

    const int r = SampleRank( C.MaxRank() );

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
                if( C._inTargetTeam ) 
                {
                    // Handle the middle update, Omega2' (alpha A B Omega1)
                    const Dense<Scalar>& X = *C._colXMap[A._sourceOffset];
                    const Dense<Scalar>& Omega2 = A._rowOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* middleUpdate = 
                        &allReduceBuffer[middleOffsets[teamLevel]];
                    middleOffsets[teamLevel] += r*r;

                    blas::Gemm
                    ( 'C', 'N', r, r, A.LocalHeight(),
                      (Scalar)1, Omega2.LockedBuffer(), Omega2.LDim(),
                                 X.LockedBuffer(),      X.LDim(),
                      (Scalar)0, middleUpdate,          r );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        nodeA.Child(t,r).
                        MultiplyHMatFHHFinalizeMiddleUpdates
                        ( nodeB.Child(r,s), nodeC.Child(t,s), 
                          allReduceBuffer, middleOffsets );
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
  std::vector<Scalar>& qrBuffer,  std::vector<int>& qrOffsets,
  std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& work )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeLocalQR");
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
                node.Child(t,s).MultiplyHMatFHHFinalizeLocalQR
                ( Xs, XOffsets, qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
                  work );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        const int teamSize = mpi::CommSize( team );
        const int log2TeamSize = Log2( teamSize );
        const int r = SampleRank( MaxRank() );

        if( _inTargetTeam )
        {
            const unsigned numEntries = _colXMap.Size();
            const unsigned teamLevel = _teams->TeamLevel(_level);
            _colXMap.ResetIterator();
            for( unsigned i=0; i<numEntries; ++i )
            {
                Dense<Scalar>& X = *_colXMap.NextEntry();
                Xs[XOffsets[teamLevel]++] = &X;
                lapack::QR
                ( X.Height(), X.Width(), X.Buffer(), X.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
                tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                if( log2TeamSize > 0 )
                {
                    std::memset
                    ( &qrBuffer[qrOffsets[teamLevel]], 0, 
                      (r*r+r)*sizeof(Scalar) );
                    const bool root = !(teamRank & 0x1);
                    if( root )
                    {
                        // Copy our R into the upper triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                    else
                    {
                        // Copy our R into the lower triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                }
                qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            }
        }
        if( _inSourceTeam )
        {
            const int numEntries = _rowXMap.Size();
            const unsigned teamLevel = _teams->TeamLevel(_level);
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                Dense<Scalar>& X = *_rowXMap.NextEntry();
                Xs[XOffsets[teamLevel]++] = &X;
                lapack::QR
                ( X.Height(), X.Width(), X.Buffer(), X.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
                tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                if( log2TeamSize > 0 )
                {
                    std::memset
                    ( &qrBuffer[qrOffsets[teamLevel]], 0, 
                      (r*r+r)*sizeof(Scalar) );
                    const bool root = !(teamRank & 0x1);
                    if( root )
                    {
                        // Copy our R into the upper triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                    else
                    {
                        // Copy our R into the lower triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                }
                qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
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
        std::vector<int>& rightOffsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeOuterUpdates");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

    const int r = SampleRank( C.MaxRank() );

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
                if( C._inTargetTeam ) 
                {
                    // Handle the left update, Q1' Omega2
                    const Dense<Scalar>& Q1 = *C._colXMap[A._sourceOffset];
                    const Dense<Scalar>& Omega2 = A._rowOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* leftUpdate = 
                        &allReduceBuffer[leftOffsets[teamLevel]];
                    leftOffsets[teamLevel] += r*r;

                    blas::Gemm
                    ( 'C', 'N', r, r, A.LocalHeight(),
                      (Scalar)1, Q1.LockedBuffer(),     Q1.LDim(),
                                 Omega2.LockedBuffer(), Omega2.LDim(),
                      (Scalar)0, leftUpdate,            r );
                }
                if( C._inSourceTeam )
                {
                    // Handle the right update, Q2' Omega1
                    const Dense<Scalar>& Q2 = *C._rowXMap[A._sourceOffset];
                    const Dense<Scalar>& Omega1 = B._colOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* rightUpdate = 
                        &allReduceBuffer[rightOffsets[teamLevel]];
                    rightOffsets[teamLevel] += r*r;

                    blas::Gemm
                    ( 'C', 'N', r, r, B.LocalWidth(),
                      (Scalar)1, Q2.LockedBuffer(),     Q2.LDim(),
                                 Omega1.LockedBuffer(), Omega1.LDim(),
                      (Scalar)0, rightUpdate,           r );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        nodeA.Child(t,r).
                        MultiplyHMatFHHFinalizeOuterUpdates
                        ( nodeB.Child(r,s), nodeC.Child(t,s), allReduceBuffer,
                          leftOffsets, rightOffsets );
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
        std::vector<Real>& svdRealWork ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeFormLowRank");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( C._block.type == EMPTY )
        return;

    const int r = SampleRank( C.MaxRank() );

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
                if( C._inTargetTeam ) 
                {
                    // Form Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                    // in the place of X.
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Dense<Scalar>& X = *C._colXMap[A._sourceOffset];

                    Scalar* leftUpdate = 
                        &allReduceBuffer[leftOffsets[teamLevel]];
                    const Scalar* middleUpdate = 
                        &allReduceBuffer[middleOffsets[teamLevel]];
                    leftOffsets[teamLevel] += r*r;
                    middleOffsets[teamLevel] += r*r;

                    lapack::AdjointPseudoInverse
                    ( r, r, leftUpdate, r, &singularValues[0],
                      &U[0], r, &VH[0], r, &svdWork[0], svdWork.size(),
                      &svdRealWork[0] );

                    // We can use the VH space to hold the product 
                    // pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                    blas::Gemm
                    ( 'N', 'N', r, r, r, 
                      (Scalar)1, leftUpdate,   r, 
                                 middleUpdate, r, 
                      (Scalar)0, &VH[0],       r );

                    // Q1 := X.
                    Dense<Scalar> Q1;
                    Q1 = X;

                    // Form X := Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                    blas::Gemm
                    ( 'N', 'N', Q1.Height(), r, r,
                      (Scalar)1, Q1.LockedBuffer(), Q1.LDim(),
                                 &VH[0],            r, 
                      (Scalar)0, X.Buffer(),        X.LDim() );
                }
                if( C._inSourceTeam )
                {
                    // Form Q2 pinv(Q2' Omega1) or its conjugate
                    Dense<Scalar>& X = *C._rowXMap[A._sourceOffset];
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);

                    Scalar* rightUpdate = 
                        &allReduceBuffer[rightOffsets[teamLevel]];
                    rightOffsets[teamLevel] += r*r;

                    lapack::AdjointPseudoInverse
                    ( r, r, rightUpdate, r, &singularValues[0],
                      &U[0], r, &VH[0], r, &svdWork[0], svdWork.size(),
                      &svdRealWork[0] );

                    // Q2 := X
                    Dense<Scalar> Q2;
                    Q2 = X;

                    blas::Gemm
                    ( 'N', 'C', Q2.Height(), r, r,
                      (Scalar)1, Q2.LockedBuffer(), Q2.LDim(),
                                 rightUpdate,       r,
                      (Scalar)0, X.Buffer(),        X.LDim() );
                    if( !Conjugated )
                        hmat_tools::Conjugate( X );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        nodeA.Child(t,r).MultiplyHMatFHHFinalizeFormLowRank
                        ( nodeB.Child(r,s), nodeC.Child(t,s), allReduceBuffer,
                          leftOffsets, middleOffsets, rightOffsets, 
                          singularValues, U, VH, svdWork, svdRealWork );
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

