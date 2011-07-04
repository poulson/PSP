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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanks
( DistQuasi2dHMat<Scalar,Conjugated>& B ) 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanks");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Count the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFormGhostRanksCount( B, sendSizes, recvSizes );

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
    std::vector<int> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatFormGhostRanksPack( B, sendBuffer, offsets );

    // Start the non-blocking sends
    MPI_Comm comm = A._teams->Team( 0 );
#ifndef RELEASE
    const int rank = mpi::CommRank( comm );
    if( rank == 0 )
    {
        std::cerr << "Forming ranks requires process 0 to send to "
                  << sendSizes.size() << " processes and recv from "
                  << recvSizes.size() << std::endl;
    }
#endif
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
    std::vector<int> recvBuffer( totalRecvSize );
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
    A.MultiplyHMatFormGhostRanksUnpack( B, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B ,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanksCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( A._sourceRoot != B._targetRoot )
    {
        std::ostringstream s;
        s << "A._sourceRoot=" << A._sourceRoot << ", B._targetRoot="
          << B._targetRoot << ", level=" << A._level;
        throw std::logic_error( s.str().c_str() );
    }
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    MPI_Comm team = A._teams->Team( A._level );
    const int teamRank = mpi::CommRank( team );
    std::pair<int,int> AOffsets( A._targetOffset, A._sourceOffset ),
                       BOffsets( B._targetOffset, B._sourceOffset );

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
            // Check if C will be admissible
            const bool admissibleC = 
                A.Admissible( B._xSource, A._xTarget, B._ySource, A._yTarget );

            if( !admissibleC )
            {
                // Recurse
                const Node& nodeA = *A._block.data.N;        
                const Node& nodeB = *B._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFormGhostRanksCount
                            ( nodeB.Child(r,s), sendSizes, recvSizes );
            }
            break;
        }
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Test if we need to send to A's target team
            if( B._inSourceTeam &&
                A._targetRoot != B._targetRoot && 
                A._targetRoot != B._sourceRoot )
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            AddToMap( recvSizes, B._sourceRoot+teamRank, 1 );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        // Check if we need to send to B's source team
        if( A._inTargetTeam &&
            B._sourceRoot != A._targetRoot &&
            B._sourceRoot != A._sourceRoot )
            AddToMap( sendSizes, B._sourceRoot+teamRank, 1 );

        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam &&
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot )
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            AddToMap( recvSizes, B._sourceRoot+teamRank, 1 );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    {
        AddToMap( recvSizes, A._targetRoot+teamRank, 1 );

        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot )
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
            break;
        }
        default:
            break;
        }
        break;
    }
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot )
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            AddToMap( recvSizes, B._sourceRoot+teamRank, 1 );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksPack
( const DistQuasi2dHMat<Scalar,Conjugated>& B ,
  std::vector<int>& sendBuffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanksPack");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    MPI_Comm team = A._teams->Team( A._level );
    const int teamRank = mpi::CommRank( team );
    std::pair<int,int> AOffsets( A._targetOffset, A._sourceOffset ),
                       BOffsets( B._targetOffset, B._sourceOffset );

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
            // Check if C will be admissible
            const bool admissibleC = 
                A.Admissible( B._xSource, A._xTarget, B._ySource, A._yTarget );

            if( !admissibleC )
            {
                // Recurse
                const Node& nodeA = *A._block.data.N;        
                const Node& nodeB = *B._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFormGhostRanksPack
                            ( nodeB.Child(r,s), sendBuffer, offsets );
            }
            break;
        }
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Test if we need to send to A's target team
            if( B._inSourceTeam &&
                A._targetRoot != B._targetRoot && 
                A._targetRoot != B._sourceRoot )
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        // Check if we need to send to B's source team
        if( A._inTargetTeam && 
            B._sourceRoot != A._targetRoot &&
            B._sourceRoot != A._sourceRoot )
            sendBuffer[offsets[B._sourceRoot+teamRank]++] = A.Rank();

        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot )
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot )
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
            break;
        }
        default:
            break;
        }
        break;
    }
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot )
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B ,
  const std::vector<int>& recvBuffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanksUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    MPI_Comm team = A._teams->Team( A._level );
    const int teamRank = mpi::CommRank( team );
    std::pair<int,int> AOffsets( A._targetOffset, A._sourceOffset ),
                       BOffsets( B._targetOffset, B._sourceOffset );

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
            // Check if C will be admissible
            const bool admissibleC = 
                A.Admissible( B._xSource, A._xTarget, B._ySource, A._yTarget );

            if( !admissibleC )
            {
                // Recurse
                Node& nodeA = *A._block.data.N;        
                Node& nodeB = *B._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFormGhostRanksUnpack
                            ( nodeB.Child(r,s), recvBuffer, offsets );
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            B.SetGhostRank( recvBuffer[offsets[B._sourceRoot+teamRank]++] );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        switch( B._block.type )
        {
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            B.SetGhostRank( recvBuffer[offsets[B._sourceRoot+teamRank]++] );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    {
        A.SetGhostRank( recvBuffer[offsets[A._targetRoot+teamRank]++] );
        break;
    }
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            B.SetGhostRank( recvBuffer[offsets[B._sourceRoot+teamRank]++] );
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

