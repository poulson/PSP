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
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int numLevels = A._numLevels;

    std::vector<std::set<std::pair<int,int> > > markedANodes(numLevels),
                                                markedBNodes(numLevels);
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFormGhostRanksCount
    ( B, markedANodes, markedBNodes, sendSizes, recvSizes );
    // HERE
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
        return;

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
            // Test if we need to send to A's target root
            MPI_Comm team = B._teams->Team( B._level );
            const int teamRank = mpi::CommRank( team );
            if( B._inSourceTeam && teamRank == 0 &&
                A._targetRoot != B._targetRoot && 
                A._targetRoot != B._sourceRoot )
            {
                const std::pair<int,int> 
                    BOffsets( B._targetOffset, B._sourceOffset );
                if( !std::binary_search
                    ( markedBNodes[B._level].begin(),
                      markedBNodes[B._level].end(), BOffsets ) )
                {
                    ++sendSizes[A._targetRoot];
                    markedBNodes[B._level].insert( BOffsets );
                }
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            // Test if we need to recv from A's target root
            MPI_Comm team = B._teams->Team( B._level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                const std::pair<int,int>
                    BOffsets( B._targetOffset, B._sourceOffset );
                if( !std::binary_search
                    ( markedBNodes[B._level].begin(),
                      markedBNodes[B._level].end(), BOffsets ) )
                {
                    ++recvSizes[A._targetRoot];
                    markedBNodes[B._level].insert( BOffsets );
                }
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        switch( B._block.type )
        {
        // TODO
        default:
            break;
        }
        break;
    }
    // TODO
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

