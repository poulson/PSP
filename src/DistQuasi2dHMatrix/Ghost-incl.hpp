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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FormGhostNodes()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FormGhostNodes");
#endif
    // Each level will have a set of source/target offsets where the structure
    // is known.
    std::vector< std::set<int> > 
        sourceStructure( _numLevels ), targetStructure( _numLevels );
    FillStructureRecursion( sourceStructure, targetStructure );
    
    MPI_Comm comm = _subcomms->Subcomm( 0 );
    const int commSize = mpi::CommSize( comm );

    std::vector< std::vector<BlockId> > blockIds( commSize );
    FindGhostNodesRecursion( blockIds, sourceStructure, targetStructure, 0, 0 );

    // TODO: Perform AllToAll to send number of blockIds to each process
    // TODO: Send and receive all blockIds
    // TODO: Pack all ranks according to blockIds
    // TODO: Send and receive all ranks
    // TODO: Unpack all ranks
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::PruneGhostNodes()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::PruneGhostNodes");
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

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    case SPLIT_DENSE:
    case DENSE:
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FillStructureRecursion
( std::vector< std::set<int> >& sourceStructure, 
  std::vector< std::set<int> >& targetStructure ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FillSourceStructureRecursion");
#endif

    switch( _block.type )
    {
    case DIST_NODE:
    {
        sourceStructure[_level].insert( _sourceOffset );
        targetStructure[_level].insert( _targetOffset );

        MPI_Comm team = _subcomms->Subcomm( _level );
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

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FindGhostNodesRecursion
( std::vector< std::vector<BlockId> >& blockIds,
  const std::vector< std::set<int> >& sourceStructure, 
  const std::vector< std::set<int> >& targetStructure,
  int sourceRoot, int targetRoot )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FindGhostNodesRecursion");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        MPI_Comm team = _subcomms->Subcomm( _level );
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

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    case SPLIT_DENSE:
    case DENSE:
        break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
#ifndef RELEASE
        throw std::logic_error("Already filled in some ghost blocks");
#endif
        break;

    case EMPTY:
    {
        if( !std::binary_search
            ( sourceStructure[_level].begin(),
              sourceStructure[_level].end(), _sourceOffset ) &&
            !std::binary_search
            ( targetStructure[_level].begin(),
              targetStructure[_level].end(), _targetOffset ) )
            break;
                               
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );

        if( Admissible() )
        {
            if( teamSize >= 2 )
            {
                _block.type = DIST_LOW_RANK_GHOST;
                _block.data.DFG = new DistLowRankGhost;
                _block.data.DFG->sourceRoot = sourceRoot;
                _block.data.DFG->targetRoot = targetRoot;
            }
            else // teamSize == 1
            {
                if( sourceRoot == targetRoot )
                {
                    _block.type = LOW_RANK_GHOST;
                    _block.data.FG = new LowRankGhost;
                    _block.data.FG->owner = sourceRoot;
                }
                else
                {
                    _block.type = SPLIT_LOW_RANK_GHOST;
                    _block.data.SFG = new SplitLowRankGhost;
                    _block.data.SFG->sourceOwner = sourceRoot;
                    _block.data.SFG->targetOwner = targetRoot;
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
            _block.data.NG = NewNodeGhost( sourceRoot, targetRoot );
            NodeGhost& nodeGhost = *_block.data.NG;
            if( teamSize >= 2 )
            {
                _block.type = DIST_NODE_GHOST;
                if( teamSize >= 4 )
                {
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            nodeGhost.Child(t,s).FindGhostNodesRecursion
                            ( blockIds, sourceStructure, targetStructure,
                              sourceRoot+s*teamSize/4, 
                              targetRoot+t*teamSize/4 );
                }
                else // teamSize == 2
                {
                    for( int t=0; t<4; ++t ) 
                        for( int s=0; s<4; ++s )
                            nodeGhost.Child(t,s).FindGhostNodesRecursion
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
                        nodeGhost.Child(t,s).FindGhostNodesRecursion
                        ( blockIds, sourceStructure, targetStructure,
                          sourceRoot, targetRoot );
            }
        }
        else
        {
            if( sourceRoot == targetRoot )
            {
                _block.type = DENSE_GHOST;
                _block.data.DG = new DenseGhost;
                _block.data.DG->owner = sourceRoot;
            }
            else
            {
                _block.type = SPLIT_DENSE_GHOST;
                _block.data.SDG = new SplitDenseGhost;
                _block.data.SDG->sourceOwner = sourceRoot;
                _block.data.SDG->targetOwner = targetRoot;
            }
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

