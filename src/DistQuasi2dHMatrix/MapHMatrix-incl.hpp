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

// C := alpha A B
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrix");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    const DistQuasi2d& A = *this;
    if( !A.Ghosted() || !B.Ghosted() )
        throw std::logic_error("A and B must have their ghost nodes");
    C.Clear();

    MapHMatrixContext context;
    A.MapHMatrixMainPrecompute( context, alpha, B, C );
    /*
    A.MapHMatrixMainPassData( context, alpha, B, C );
    A.MapHMatrixMainPostcompute( context, alpha, B, C );
    A.MapHMatrixFHHPrecompute( context, alpha, B, C );
    A.MapHMatrixFHHPassData( context, alpha, B, C );
    A.MapHMatrixFHHPostcompute( context, alpha, B, C );
    A.MapHMatrixFHHFinalize( context, alpha, B, C );
    A.MapHMatrixRoundedAddition( context, alpha, B, C );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixFillMemberData
( const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
        DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapHMatrixFillMemberData");
#endif
    const DistQuasi2d& A = *this;

    C._numLevels = A._numLevels;
    C._maxRank = A._maxRank;
    C._sourceOffset = B._sourceOffset;
    C._targetOffset = A._targetOffset;
    C._stronglyAdmissible = ( A._stronglyAdmissible || B._stronglyAdmissible );

    C._xSizeSource = B._xSizeSource;
    C._ySizeSource = B._ySizeSource;
    C._xSizeTarget = A._xSizeTarget;
    C._ySizeTarget = A._ySizeTarget;
    C._zSize = A._zSize;
    C._xSource = B._xSource;
    C._ySource = B._ySource;
    C._xTarget = A._xTarget;
    C._yTarget = A._yTarget;

    C._subcomms = A._subcomms;
    C._level = A._level;
    C._inSourceTeam = B._inSourceTeam;
    C._inTargetTeam = A._inTargetTeam;
    if( C._inSourceTeam && !C._inTargetTeam )
        C._rootOfOtherTeam = A._rootOfOtherTeam;
    else if( !C._inSourceTeam && C._inTargetTeam )
        C._rootOfOtherTeam = B._rootOfOtherTeam;
    C._localSourceOffset = B._localSourceOffset;
    C._localTargetOffset = A._localTargetOffset;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixMainPrecompute
( MapHMatrixContext& context,
  Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapHMatrixMainPrecompute");
#endif
    const DistQuasi2d& A = *this;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
        C._block.type = EMPTY;
        context.block.type = EMPTY;
        return;
    }
    const bool newC = ( C._block.type == EMPTY );
    if( newC )
        MapHMatrixFillMemberData( B, C );
    const bool inC = ( C._inSourceTeam || C._inTargetTeam );

    MPI_Comm team = A._subcomms->Subcomm( A._level );
    const int teamSize = mpi::CommSize( team );
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( C.Admissible() )
            {
                if( newC )
                {
                    if( inC ) // we're in one of the outer teams
                    {
                        C._block.type = DIST_LOW_RANK;
                        C._block.data.DF = new DistLowRank;
                        C._block.data.DF->rank = 0;
                    }
                    else // we're in the middle team
                    {
                        C._block.type = DIST_LOW_RANK_GHOST;
                        C._block.data.DFG = new DistLowRankGhost;
                        C._block.data.DF->rank = 0;
                        C._block.data.DFG->sourceRoot = B._rootOfOtherTeam;
                        C._block.data.DFG->targetRoot = A._rootOfOtherTeam;
                    }
                    context.block.type = DIST_LOW_RANK;
                    context.block.data.DF = 
                        new typename MapHMatrixContext::DistLowRankContext;
                }
                // TODO: Start F += H H 
            }
            else
            {
                if( newC )
                {
                    if( inC ) // we're in one of the outer teams
                    {
                        C._block.type = DIST_NODE;
                        C._block.data.N = C.NewNode();
                        for( int j=0; j<16; ++j )
                            C._block.data.N->children[j] = new DistQuasi2d;
                    }
                    else // we're in the middle team
                    {
                        C._block.type = DIST_NODE_GHOST;
                        C._block.data.NG = 
                            C.NewNodeGhost
                            ( B._rootOfOtherTeam, A._rootOfOtherTeam );
                        for( int j=0; j<16; ++j )
                            C._block.data.NG->children[j] = new DistQuasi2d;
                    }
                    context.block.type = DIST_NODE;
                    context.block.data.DN = 
                        new typename MapHMatrixContext::DistNodeContext;
                }
                // TODO: Recurse
            }
            break;
        case DIST_NODE_GHOST:
            // We must be in the left team
            if( C.Admissible() )
            {
                if( newC )
                {
                    C._block.type = DIST_LOW_RANK;
                    C._block.data.DF = new DistLowRank;
                    C._block.data.DF->rank = 0;
                    context.block.type = DIST_LOW_RANK;
                    context.block.data.DF = 
                        new typename MapHMatrixContext::DistLowRankContext;
                }
                // TODO: Start F += H H
            }
            else
            {
                if( newC )
                {
                    C._block.type = DIST_NODE;
                    C._block.data.N = C.NewNode();
                    for( int j=0; j<16; ++j )
                        C._block.data.N->children[j] = new DistQuasi2d;
                    context.block.type = DIST_NODE;
                    context.block.data.DN = 
                        new typename MapHMatrixContext::DistNodeContext;
                }
                // TODO: Recurse
            }
            break;
        case DIST_LOW_RANK:
            if( C.Admissible() )
            {
                if( newC )
                {
                    if( inC ) // we're in one of the outer teams
                    {
                        C._block.type = DIST_LOW_RANK;
                        C._block.data.DF = new DistLowRank;
                        C._block.data.DF->rank = 0;
                    }
                    else // we're in the middle team
                    {
                        C._block.type = DIST_LOW_RANK_GHOST;
                        C._block.data.DFG = new DistLowRankGhost;
                        C._block.data.DFG->rank = 0;
                        C._block.data.DFG->sourceRoot = B._rootOfOtherTeam;
                        C._block.data.DFG->targetRoot = A._rootOfOtherTeam;
                    }
                    context.block.type = DIST_LOW_RANK;
                    context.block.data.DF = 
                        new typename MapHMatrixContext::DistLowRankContext;
                }
                // TODO: Start F += H F
            }
            else
            {
                if( newC )
                {
                    if( inC ) // we're in one of the outer teams
                    {
                        C._block.type = DIST_NODE;
                        C._block.data.N = C.NewNode();
                        for( int j=0; j<16; ++j )
                            C._block.data.N->children[j] = new DistQuasi2d;
                    }
                    else // we're in the middle team
                    {
                        C._block.type = DIST_NODE_GHOST;
                        C._block.data.NG = 
                            C.NewNodeGhost
                            ( B._rootOfOtherTeam, A._rootOfOtherTeam );
                        for( int j=0; j<16; ++j )
                            C._block.data.NG->children[j] = new DistQuasi2d;
                    }
                    context.block.type = DIST_NODE;
                    context.block.data.DN = 
                        new typename MapHMatrixContext::DistNodeContext;
                }
                // TODO: Start H += H F
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // We must be in the left team    
            if( C.Admissible() ) 
            {
                if( newC ) 
                {
                    C._block.type = DIST_LOW_RANK;
                    C._block.data.DF = new DistLowRank;
                    C._block.data.DF->rank = 0;
                    context.block.type = DIST_LOW_RANK;
                    context.block.data.DF = 
                        new typename MapHMatrixContext::DistLowRankContext;
                }
                // TODO: Start F += H F
            }
            else
            {
                if( newC )
                {
                    C._block.type = DIST_NODE;
                    C._block.data.N = C.NewNode();
                    for( int j=0; j<16; ++j )
                        C._block.data.N->children[j] = new DistQuasi2d;
                    context.block.type = DIST_NODE;
                    context.block.data.DN = 
                        new typename MapHMatrixContext::DistNodeContext;
                }
                // TODO: Start H += H F
            }
            break;
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case DIST_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
            // HERE
        case DIST_LOW_RANK:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case SPLIT_NODE:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
        
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_LOW_RANK:
        
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case NODE:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case NODE:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case NODE_GHOST:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_LOW_RANK:
        
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case DIST_LOW_RANK:
        case DIST_LOW_RANK_GHOST:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case DIST_LOW_RANK_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_LOW_RANK:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case SPLIT_LOW_RANK:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
        case SPLIT_DENSE:
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case SPLIT_LOW_RANK_GHOST:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case LOW_RANK:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case NODE:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        case SPLIT_DENSE:
        case DENSE:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case LOW_RANK_GHOST:
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case SPLIT_DENSE:
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
        case SPLIT_DENSE:
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case SPLIT_DENSE_GHOST:
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case DENSE:
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        case SPLIT_DENSE:
        case DENSE:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case DENSE_GHOST:
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case EMPTY:
#ifndef RELEASE
        throw std::logic_error("A should not be empty");
#endif
    }
#ifndef RELEASE
    PopCallStack();
#endif
}


