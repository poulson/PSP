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
    const int numLevels = A._numLevels;

    MPI_Comm comm = A._teams->Team( 0 );
    const int rank = mpi::CommRank( comm );

    // Count the send/recv sizes
    std::vector<std::set<std::pair<int,int> > > markedANodes(numLevels),
                                                markedBNodes(numLevels);
    std::map<int,int> sendSizes, recvSizes;
    mpi::Barrier( comm );
    if( rank == 0 )
    {
        std::cout << "Count...";
        std::cout.flush();
    }
    A.MultiplyHMatFormGhostRanksCount
    ( B, markedANodes, markedBNodes, sendSizes, recvSizes );
    mpi::Barrier( comm );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;
    for( int i=0; i<numLevels; ++i )
    {
        markedANodes[i].clear();
        markedBNodes[i].clear();
    }

    // Compute the offsets
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
        std::ostringstream s;
        s << rank << " sending " << it->second << " to " << it->first 
          << std::endl;
        std::cout << s.str();
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
        std::ostringstream s;
        s << rank << " recving " << it->second << " from " << it->first 
          << std::endl;
        std::cout << s.str();
    }

    // Fill the send buffer
    std::vector<int> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    mpi::Barrier( comm );
    if( rank == 0 )
    {
        std::cout << "Pack...";
        std::cout.flush();
    }
    A.MultiplyHMatFormGhostRanksPack
    ( B, markedANodes, markedBNodes, sendBuffer, offsets );
    mpi::Barrier( comm );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;
    for( int i=0; i<numLevels; ++i )
    {
        markedANodes[i].clear();
        markedBNodes[i].clear();
    }

    // Start the non-blocking sends
    if( rank == 0 )
        std::cout << "Sends" << std::endl;
    //MPI_Comm comm = A._teams->Team( 0 );
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
    if( rank == 0 )
        std::cout << "Recvs" << std::endl;
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
    if( rank == 0 )
        std::cout << "Waiting for recvs..." << std::endl;
    for( int i=0; i<numRecvs; ++i )
        mpi::Wait( recvRequests[i] );
    if( rank == 0 )
    {
        std::cout << "Unpacking...";
        std::cout.flush();
    }
    A.MultiplyHMatFormGhostRanksUnpack
    ( B, markedANodes, markedBNodes, recvBuffer, recvOffsets );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
    if( rank == 0 )
        std::cout << "Finished sends." << std::endl;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B ,
  std::vector<std::set<std::pair<int,int> > >& markedANodes,
  std::vector<std::set<std::pair<int,int> > >& markedBNodes,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanksCount");
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
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE_GHOST:
        case NODE_GHOST:
        {
            // Recurse
            const Node& nodeA = *A._block.data.N;        
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MultiplyHMatFormGhostRanksCount
                        ( nodeB.Child(r,s), markedANodes, markedBNodes,
                          sendSizes, recvSizes );
            break;
        }
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Test if we need to send to A's target team
            if( B._inSourceTeam &&
                A._targetRoot != B._targetRoot && 
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                 ( markedBNodes[B._level].begin(),
                   markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            // Test if we need to recv from B's source team
            if( !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( recvSizes, B._sourceRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
            }
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
            B._sourceRoot != A._sourceRoot &&
            !std::binary_search
            ( markedANodes[A._level].begin(),
              markedANodes[A._level].end(), AOffsets ) )
        {
            AddToMap( sendSizes, B._sourceRoot+teamRank, 1 );
            markedANodes[A._level].insert( AOffsets );
        }

        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam &&
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            // Check if we need to recv from B's source team
            if( !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( recvSizes, B._sourceRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
            }
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
        // Check if we need to recv from A's target team
        if( !std::binary_search
            ( markedANodes[A._level].begin(),
              markedANodes[A._level].end(), AOffsets ) )
        {
            AddToMap( recvSizes, A._targetRoot+teamRank, 1 );
            markedANodes[A._level].insert( AOffsets );
        }

        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
            }
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
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( sendSizes, A._targetRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            // Check if we need to recv from A's target team
            if( !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                AddToMap( recvSizes, A._targetRoot+teamRank, 1 );
                markedBNodes[B._level].insert( BOffsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksPack
( const DistQuasi2dHMat<Scalar,Conjugated>& B ,
  std::vector<std::set<std::pair<int,int> > >& markedANodes,
  std::vector<std::set<std::pair<int,int> > >& markedBNodes,
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
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE_GHOST:
        case NODE_GHOST:
        {
            // Recurse
            const Node& nodeA = *A._block.data.N;        
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MultiplyHMatFormGhostRanksPack
                        ( nodeB.Child(r,s), markedANodes, markedBNodes,
                          sendBuffer, offsets );
            break;
        }
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Test if we need to send to A's target team
            if( B._inSourceTeam &&
                A._targetRoot != B._targetRoot && 
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                 ( markedBNodes[B._level].begin(),
                   markedBNodes[B._level].end(), BOffsets ) )
            {
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
                markedBNodes[B._level].insert( BOffsets );
            }
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
            B._sourceRoot != A._sourceRoot &&
            !std::binary_search
            ( markedANodes[A._level].begin(),
              markedANodes[A._level].end(), AOffsets ) )
        {
            sendBuffer[offsets[B._sourceRoot+teamRank]++] = A.Rank();
            markedANodes[A._level].insert( AOffsets );
        }

        switch( B._block.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B._inSourceTeam && 
                A._targetRoot != B._targetRoot &&
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
                markedBNodes[B._level].insert( BOffsets );
            }
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
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
                markedBNodes[B._level].insert( BOffsets );
            }
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
                A._targetRoot != B._sourceRoot &&
                !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                sendBuffer[offsets[A._targetRoot+teamRank]++] = B.Rank();
                markedBNodes[B._level].insert( BOffsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B ,
  std::vector<std::set<std::pair<int,int> > >& markedANodes,
  std::vector<std::set<std::pair<int,int> > >& markedBNodes,
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
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE_GHOST:
        case NODE_GHOST:
        {
            // Recurse
            Node& nodeA = *A._block.data.N;        
            Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MultiplyHMatFormGhostRanksUnpack
                        ( nodeB.Child(r,s), markedANodes, markedBNodes,
                          recvBuffer, offsets );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            // Test if we need to recv from B's source team
            if( !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                B.SetGhostRank( recvBuffer[offsets[B._sourceRoot+teamRank]++] );
                markedBNodes[B._level].insert( BOffsets );
            }
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
            // Check if we need to recv from B's source team
            if( !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                B.SetGhostRank( recvBuffer[offsets[B._sourceRoot+teamRank]++] );
                markedBNodes[B._level].insert( BOffsets );
            }
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
        // Check if we need to recv from A's target team
        if( !std::binary_search
            ( markedANodes[A._level].begin(),
              markedANodes[A._level].end(), AOffsets ) )
        {
            A.SetGhostRank( recvBuffer[offsets[A._targetRoot+teamRank]++] );
            markedANodes[A._level].insert( AOffsets );
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
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            // Check if we need to recv from B's source team
            if( !std::binary_search
                ( markedBNodes[B._level].begin(),
                  markedBNodes[B._level].end(), BOffsets ) )
            {
                B.SetGhostRank( recvBuffer[offsets[B._sourceRoot+teamRank]++] );
                markedBNodes[B._level].insert( BOffsets );
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

