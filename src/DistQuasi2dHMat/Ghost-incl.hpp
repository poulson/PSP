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
    
    // Fill in the local ghosted structure (but without the ghosts' ranks)
    FindGhostNodesRecursion( sourceStructure, targetStructure, 0, 0 );
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
( const std::vector< std::set<int> >& sourceStructure, 
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
                    ( sourceStructure, targetStructure, 
                      sourceRoot+s*teamSize/4, targetRoot+t*teamSize/4 );
        }
        else // teamSize == 2
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindGhostNodesRecursion
                    ( sourceStructure, targetStructure,
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
                ( sourceStructure, targetStructure, sourceRoot, targetRoot );
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
                _block.data.DFG->rank = -1;
            }
            else // teamSize == 1
            {
                if( sourceRoot == targetRoot )
                {
                    _block.type = LOW_RANK_GHOST;
                    _block.data.FG = new LowRankGhost;
                    _block.data.FG->rank = -1;
                }
                else
                {
                    _block.type = SPLIT_LOW_RANK_GHOST;
                    _block.data.SFG = new SplitLowRankGhost;
                    _block.data.SFG->rank = -1;
                }
            }
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
                            ( sourceStructure, targetStructure,
                              sourceRoot+s*teamSize/4, 
                              targetRoot+t*teamSize/4 );
                }
                else // teamSize == 2
                {
                    for( int t=0; t<4; ++t ) 
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).FindGhostNodesRecursion
                            ( sourceStructure, targetStructure,
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
                        ( sourceStructure, targetStructure,
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

