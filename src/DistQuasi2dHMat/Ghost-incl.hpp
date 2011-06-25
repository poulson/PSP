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
bool
psp::DistQuasi2dHMat<Scalar,Conjugated>::Ghosted() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Ghosted");
#endif
    RequireRoot();

    bool ghosted = true;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        MPI_Comm comm = _teams->Team( 0 );
        const int commSize = mpi::CommSize( comm );
        const int commRank = mpi::CommRank( comm );

        const Node& node = *_block.data.N;
        if( commSize >= 4 )
        {
            const int subteam = commRank / (commSize/4);
            ghosted = !(node.Child(3-subteam,3-subteam)._block.type == EMPTY);
        }
        else // commSize == 2
            ghosted = !(node.Child(2-commRank,2-commRank)._block.type == EMPTY);
        break;
    }

    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return ghosted;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::FormGhostNodes()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::FormGhostNodes");
#endif
    RequireRoot();

    // Each level will have a set of source/target offsets where the structure
    // is known.
    std::vector< std::set<int> > 
        sourceStructure( _numLevels ), targetStructure( _numLevels );
    FillStructureRecursion( sourceStructure, targetStructure );
    
    MPI_Comm comm = _teams->Team( 0 );
    const int commSize = mpi::CommSize( comm );

    // Fill in the local ghosted structure (but without the ghosts' ranks)
    std::vector< std::vector<BlockId> > sendBlockIds( commSize );
    FindGhostNodesRecursion
    ( sendBlockIds, sourceStructure, targetStructure, 0, 0 );

    // Have every process let every other process know how many blockIds
    // they will be sending for the rank requests
    std::vector<int> numSendBlocks( commSize ), numRecvBlocks( commSize );
    for( int i=0; i<commSize; ++i )
        numSendBlocks[i] = sendBlockIds[i].size();
    mpi::AllToAll( &numSendBlocks[0], 1, &numRecvBlocks[0], 1, comm );

    // Set up the send/recv counts/displacement index vectors for the AllToAllV
    std::vector<int> sendCounts( commSize ), sendDispls( commSize ),
                     recvCounts( commSize ), recvDispls( commSize );
    for( int i=0; i<commSize; ++i )
        sendCounts[i] = numSendBlocks[i]*sizeof(BlockId);
    sendDispls[0] = 0;
    for( int i=1; i<commSize; ++i )
        sendDispls[i] = sendDispls[i-1] + sendCounts[i-1];
    const int totalSendCount = sendDispls[commSize-1] + sendCounts[commSize-1];
    for( int i=0; i<commSize; ++i )
        recvCounts[i] = numRecvBlocks[i]*sizeof(BlockId);
    recvDispls[0] = 0;
    for( int i=1; i<commSize; ++i )
        recvDispls[i] = recvDispls[i-1] + recvCounts[i-1];
    const int totalRecvCount = recvDispls[commSize-1] + recvCounts[commSize-1];

    // Pack and perform the AllToAllV, then unpack
    std::vector<byte> sendBuf( totalSendCount ), recvBuf( totalRecvCount );
    for( int i=0; i<commSize; ++i )
        std::memcpy
        ( &sendBuf[sendDispls[i]], &sendBlockIds[i][0], sendCounts[i] );
    mpi::AllToAllV
    ( &sendBuf[0], &sendCounts[0], &sendDispls[0],
      &recvBuf[0], &recvCounts[0], &recvDispls[0], comm );
    std::vector< std::vector<BlockId> > recvBlockIds( commSize );
    for( int i=0; i<commSize; ++i )
        recvBlockIds[i].resize( numRecvBlocks[i] );
    for( int i=0; i<commSize; ++i )
        std::memcpy
        ( &recvBlockIds[i][0], &recvBuf[recvDispls[i]], recvCounts[i] );

    for( int i=0; i<commSize; ++i )
    {
        const int numRecvBlocks = recvCounts[i] / sizeof(BlockId);
        for( int j=0; j<numRecvBlocks; ++j )
            GetRank
            ( recvBlockIds[i][j], 
              *((int*)&recvBuf[recvDispls[i]+j*sizeof(int)]) );
    }

    // Prepare for sending/recving the ranks. Switch the send/recv buffers
    // since we're going in the opposite direction now and they are sufficiently
    // large.
    for( int i=0; i<commSize; ++i )
        sendCounts[i] = numRecvBlocks[i]*sizeof(int);
    sendDispls[0] = 0;
    for( int i=1; i<commSize; ++i )
        sendDispls[i] = sendDispls[i-1] + sendCounts[i-1];
    for( int i=0; i<commSize; ++i )
        recvCounts[i] = numSendBlocks[i]*sizeof(int);
    recvDispls[0] = 0;
    for( int i=1; i<commSize; ++i )
        recvDispls[i] = recvDispls[i-1] + recvCounts[i-1]; 
    mpi::AllToAllV
    ( &recvBuf[0], &sendCounts[0], &sendDispls[0],
      &sendBuf[0], &recvCounts[0], &recvDispls[0], comm );

    for( int i=0; i<commSize; ++i )
    {
        const int numRecvRanks = recvCounts[i] / sizeof(int);
        for( int j=0; j<numRecvRanks; ++j )
            SetGhostRank
            ( sendBlockIds[i][j], 
              *((int*)&sendBuf[recvDispls[i]+j*sizeof(int)]) );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::PruneGhostNodes()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::PruneGhostNodes");
#endif
    switch( _block.type )
    {
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
        _block.Clear();
        break;

    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).PruneGhostNodes();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::FillStructureRecursion
( std::vector< std::set<int> >& sourceStructure, 
  std::vector< std::set<int> >& targetStructure ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::FillSourceStructureRecursion");
#endif

    switch( _block.type )
    {
    case DIST_NODE:
    {
        sourceStructure[_level].insert( _sourceOffset );
        targetStructure[_level].insert( _targetOffset );

        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        const Node& node = *_block.data.N;
        if( teamSize >= 4 )
        {
            const int subteam = teamRank / (teamSize/4);
            for( int t=0; t<4; ++t )
                node.Child( t, subteam ).FillStructureRecursion
                ( sourceStructure, targetStructure );
            for( int s=0; s<4; ++s )
                node.Child( subteam, s ).FillStructureRecursion
                ( sourceStructure, targetStructure );
        }
        else // teamSize == 2
        {
            if( teamRank == 0 )
            {
                // Upper half
                for( int t=0; t<2; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FillStructureRecursion
                        ( sourceStructure, targetStructure );
                // Bottom-left block
                for( int t=2; t<4; ++t )
                    for( int s=0; s<2; ++s )
                        node.Child(t,s).FillStructureRecursion
                        ( sourceStructure, targetStructure );
            }
            else // teamRank == 1
            {
                // Upper-right block
                for( int t=0; t<2; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).FillStructureRecursion
                        ( sourceStructure, targetStructure );
                // Bottom half
                for( int t=2; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FillStructureRecursion
                        ( sourceStructure, targetStructure );
            }
        }
        break;
    }

    case SPLIT_NODE:
    case NODE:
    {
        sourceStructure[_level].insert( _sourceOffset );
        targetStructure[_level].insert( _targetOffset );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).FillStructureRecursion
                ( sourceStructure, targetStructure );
        break;
    }

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    case SPLIT_DENSE:
    case DENSE:
        sourceStructure[_level].insert( _sourceOffset );
        targetStructure[_level].insert( _targetOffset );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::FindGhostNodesRecursion
( std::vector< std::vector<BlockId> >& blockIds,
  const std::vector< std::set<int> >& sourceStructure, 
  const std::vector< std::set<int> >& targetStructure,
  int sourceRoot, int targetRoot )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::FindGhostNodesRecursion");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        if( teamSize >= 4 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindGhostNodesRecursion
                    ( blockIds, sourceStructure, targetStructure, 
                      sourceRoot+s*teamSize/4, targetRoot+t*teamSize/4 );
        }
        else // teamSize == 2
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindGhostNodesRecursion
                    ( blockIds, sourceStructure, targetStructure,
                      sourceRoot+s/2, targetRoot+t/2 );
        }
        break;
    }

    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).FindGhostNodesRecursion
                ( blockIds, sourceStructure, targetStructure, 
                  sourceRoot, targetRoot );
        break;
    }

    case EMPTY:
    {
        if( !std::binary_search
            ( sourceStructure[_level].begin(),
              sourceStructure[_level].end(), _sourceOffset ) &&
            !std::binary_search
            ( targetStructure[_level].begin(),
              targetStructure[_level].end(), _targetOffset ) )
            break;
                               
        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );

        if( Admissible() )
        {
            if( teamSize >= 2 )
            {
                _block.type = DIST_LOW_RANK_GHOST;
                _block.data.DFG = new DistLowRankGhost;
            }
            else // teamSize == 1
            {
                if( sourceRoot == targetRoot )
                {
                    _block.type = LOW_RANK_GHOST;
                    _block.data.FG = new LowRankGhost;
                }
                else
                {
                    _block.type = SPLIT_LOW_RANK_GHOST;
                    _block.data.SFG = new SplitLowRankGhost;
                }
            }
            BlockId id;
            id.level = _level;
            id.sourceOffset = _sourceOffset;
            id.targetOffset = _targetOffset;
            blockIds[sourceRoot].push_back( id );
        }
        else if( _numLevels > 1 )
        {
            _block.data.N = NewNode();
            Node& node = *_block.data.N;

            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+(s/2), 2*_yTarget+(t/2),
                          *_teams, _level+1, false, false, 
                          sourceRoot, targetRoot );

            if( teamSize >= 2 )
            {
                _block.type = DIST_NODE_GHOST;
                if( teamSize >= 4 )
                {
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).FindGhostNodesRecursion
                            ( blockIds, sourceStructure, targetStructure,
                              sourceRoot+s*teamSize/4, 
                              targetRoot+t*teamSize/4 );
                }
                else // teamSize == 2
                {
                    for( int t=0; t<4; ++t ) 
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).FindGhostNodesRecursion
                            ( blockIds, sourceStructure, targetStructure,
                              sourceRoot+s/2, targetRoot+t/2 );
                }
            }
            else // teamSize == 1
            {
                if( sourceRoot == targetRoot )
                    _block.type = NODE_GHOST;
                else
                    _block.type = SPLIT_NODE_GHOST;
                
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FindGhostNodesRecursion
                        ( blockIds, sourceStructure, targetStructure,
                          sourceRoot, targetRoot );
            }
        }
        else
        {
            if( sourceRoot == targetRoot )
                _block.type = DENSE_GHOST;
            else
                _block.type = SPLIT_DENSE_GHOST;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::GetRank
( const BlockId& blockId, int& rank ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::GetRank");
#endif
    switch( _block.type )
    {
    case DIST_NODE: 
    case SPLIT_NODE:
    case NODE:
    {
        const Node& node = *_block.data.N;
        int t=0,tOffset=_targetOffset;
        for(; t<4; tOffset+=node.targetSizes[t],++t )
            if( tOffset+node.targetSizes[t] > blockId.targetOffset )
                break;
        int s=0,sOffset=_sourceOffset;
        for(; s<4; sOffset+=node.sourceSizes[s],++s )
            if( sOffset+node.sourceSizes[s] > blockId.sourceOffset )
                break;
        node.Child(t,s).GetRank( blockId, rank );
        break;
    }
    case DIST_LOW_RANK:
        rank = _block.data.DF->rank;
        break;
    case SPLIT_LOW_RANK:
        rank = _block.data.SF->rank;
        break;
    case LOW_RANK:
        rank = _block.data.F->Rank();
        break;

    default:
#ifndef RELEASE
        throw std::logic_error("Invalid logic in GetRank");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::SetGhostRank
( const BlockId& blockId, const int rank ) 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::SetGhostRank");
#endif
    switch( _block.type )
    {
    case DIST_NODE: 
    case SPLIT_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        Node& node = *_block.data.N;
        int t=0,tOffset=_targetOffset;
        for(; t<4; tOffset+=node.targetSizes[t],++t )
            if( tOffset+node.targetSizes[t] > blockId.targetOffset )
                break;
        int s=0,sOffset=_sourceOffset;
        for(; s<4; sOffset+=node.sourceSizes[s],++s )
            if( sOffset+node.sourceSizes[s] > blockId.sourceOffset )
                break;
        node.Child(t,s).SetGhostRank( blockId, rank );
        break;
    }
    case DIST_LOW_RANK_GHOST:
        _block.data.DFG->rank = rank;
        break;
    case SPLIT_LOW_RANK_GHOST:
        _block.data.SFG->rank = rank;
        break;
    case LOW_RANK_GHOST:
        _block.data.FG->rank = rank;
        break;

    default:
#ifndef RELEASE
        throw std::logic_error("Invalid logic in SetGhostRank");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

