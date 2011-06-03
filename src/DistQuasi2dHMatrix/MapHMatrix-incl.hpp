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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixSetUpC
( const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
        DistQuasi2dHMatrix<Scalar,Conjugated>& C,
        MapHMatrixContext& context ) const
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
    C._sourceRoot = B._sourceRoot;
    C._targetRoot = A._targetRoot;
    C._localSourceOffset = B._localSourceOffset;
    C._localTargetOffset = A._localTargetOffset;
    
    MPI_Comm team = _subcomms->Subcomm( _level );
    const int teamSize = mpi::CommSize( team );
    if( C.Admissible() ) // C is low-rank
    {
        if( teamSize > 1 )
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = DIST_LOW_RANK;
                C._block.data.DF = new DistLowRank;
                DistLowRank& DF = *C._block.data.DF;
                DF.rank = 0;
                if( C._inTargetTeam )
                    DF.ULocal.Resize( C.LocalHeight(), 0 );
                if( C._inSourceTeam )
                    DF.VLocal.Resize( C.LocalWidth(), 0 );
                context.block.type = DIST_LOW_RANK;
                context.block.data.DF = 
                    new typename MapHMatrixContext::DistLowRankContext;
            }
            else
            {
                C._block.type = DIST_LOW_RANK_GHOST;
                C._block.data.DFG = new DistLowRankGhost;
                DistLowRankGhost& DFG = *C._block.data.DFG;
                DFG.rank = 0;
                context.block.type = DIST_LOW_RANK_GHOST;
                context.block.data.DF = 
                    new typename MapHMatrixContext::DistLowRankContext;
            }
        }
        else // teamSize == 1
        {
            if( C._sourceRoot == C._targetRoot )
            {
                if( C._inSourceTeam || C._inTargetTeam )
                {
                    C._block.type = LOW_RANK;
                    C._block.data.F = new LowRank;
                    LowRank& F = *C._block.data.F;
                    F.U.Resize( C.Height(), 0 );
                    F.V.Resize( C.Width(), 0 );
                    context.block.type = LOW_RANK;
                    context.block.data.F = 
                        new typename MapHMatrixContext::LowRankContext;
                }
                else
                {
                    C._block.type = LOW_RANK_GHOST;
                    C._block.data.FG = new LowRankGhost;
                    LowRankGhost& FG = *C._block.data.FG;
                    FG.rank = 0;
                    context.block.type = LOW_RANK_GHOST;
                    context.block.data.F = 
                        new typename MapHMatrixContext::LowRankContext;
                }
            }
            else
            {
                if( C._inSourceTeam || C._inTargetTeam )
                {
                    C._block.type = SPLIT_LOW_RANK;
                    C._block.data.SF = new SplitLowRank;
                    SplitLowRank& SF = *C._block.data.SF;
                    SF.rank = 0;
                    if( C._inTargetTeam )
                        SF.D.Resize( C.Height(), 0 );
                    else
                        SF.D.Resize( C.Width(), 0 );
                    context.block.type = SPLIT_LOW_RANK;
                    context.block.data.SF =
                        new typename MapHMatrixContext::SplitLowRankContext;
                }
                else
                {
                    C._block.type = SPLIT_LOW_RANK_GHOST;
                    C._block.data.SFG = new SplitLowRankGhost;
                    SplitLowRankGhost& SFG = *C._block.data.SFG;
                    SFG.rank = 0;
                    context.block.type = SPLIT_LOW_RANK_GHOST;
                    context.block.data.SF = 
                        new typename MapHMatrixContext::SplitLowRankContext;
                }
            }
        }
    }
    else if( C._numLevels > 1 ) // C is hierarchical
    {
        if( teamSize > 1 )
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = DIST_NODE;
                C._block.data.N = C.NewNode();
                Node& node = *C._block.data.N;
                for( int j=0; j<16; ++j )
                    node.children[j] = new DistQuasi2d;
                context.block.type = DIST_NODE;
                context.block.data.DN = 
                    new typename MapHMatrixContext::DistNodeContext;
            }
            else
            {
                C._block.type = DIST_NODE_GHOST;
                C._block.data.N = C.NewNode();
                Node& node = *C._block.data.N;
                for( int j=0; j<16; ++j )
                    node.children[j] = new DistQuasi2d;
                context.block.type = DIST_NODE_GHOST;
                context.block.data.DN = 
                    new typename MapHMatrixContext::DistNodeContext;
            }
        }
        else
        {
            if( C._sourceRoot == C._targetRoot )
            {
                if( C._inSourceTeam || C._inTargetTeam )
                {
                    C._block.type = NODE;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = new DistQuasi2d;
                    context.block.type = NODE;
                    context.block.data.N = 
                        new typename MapHMatrixContext::NodeContext;
                }
                else
                {
                    C._block.type = NODE_GHOST;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = new DistQuasi2d;
                    context.block.type = NODE_GHOST;
                    context.block.data.N = 
                        new typename MapHMatrixContext::NodeContext;
                }
            }
            else
            {
                if( C._inSourceTeam || C._inTargetTeam )
                {
                    C._block.type = SPLIT_NODE;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = new DistQuasi2d;
                    context.block.type = SPLIT_NODE;
                    context.block.data.SN = 
                        new typename MapHMatrixContext::SplitNodeContext;
                }
                else
                {
                    C._block.type = SPLIT_NODE_GHOST;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = new DistQuasi2d;
                    context.block.type = SPLIT_NODE_GHOST;
                    context.block.data.SN = 
                        new typename MapHMatrixContext::SplitNodeContext;
                }
            }
        }
    }
    else // C is dense
    {
        if( C._sourceRoot == C._targetRoot )
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = DENSE;
                C._block.data.D = new Dense;
                Dense& D = *C._block.data.D;
                D.Resize( C.Height(), C.Width(), C.Height() );
                context.block.type = DENSE;
                context.block.data.D = 
                    new typename MapHMatrixContext::DenseContext;
            }
            else
            {
                C._block.type = DENSE_GHOST;
                C._block.data.DG = new DenseGhost;
                context.block.type = DENSE_GHOST;
                context.block.data.D = 
                    new typename MapHMatrixContext::DenseContext;
            }
        }
        else
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = SPLIT_DENSE;
                C._block.data.SD = new SplitDense;
                SplitDense& SD = *C._block.data.SD;
                if( C._inSourceTeam )
                    SD.D.Resize( C.Height(), C.Width() );
                context.block.type = SPLIT_DENSE;
                context.block.data.SD = 
                    new typename MapHMatrixContext::SplitDenseContext;
            }
            else
            {
                C._block.type = SPLIT_DENSE_GHOST;
                C._block.data.SDG = new SplitDenseGhost;
                context.block.type = SPLIT_DENSE_GHOST;
                context.block.data.SD = 
                    new typename MapHMatrixContext::SplitDenseContext;
            }
        }
    }
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

    if( C._block.type == EMPTY )
        A.MapHMatrixSetUpC( B, C, context );

    MPI_Comm team = _subcomms->Subcomm( _level );
    const int teamSize = mpi::CommSize( team );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( admissibleC )
            {
                // TODO: Start F += H H 
            }
            else
            {
                // TODO: Recurse
            }
            break;
        case DIST_NODE_GHOST:
            // We must be in the left team
            if( admissibleC )
            {
                // TODO: Start F += H H
            }
            else
            {
                // TODO: Recurse
            }
            break;
        case DIST_LOW_RANK:
            if( admissibleC )
            {
                // TODO: Start F += H F
            }
            else
            {
                // TODO: Start H += H F
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // We must be in the left team    
            if( admissibleC ) 
            {
                // TODO: Start F += H F
            }
            else
            {
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
            // We must be in the right team
            if( admissibleC )
            {
                // TODO: Start F += H H
            }
            else
            {
                // TODO: Start H += H H
            }
            break;
        case DIST_LOW_RANK:
            // We must be in the right team
            if( admissibleC )
            {
                // TODO: Start F += H F
            }
            else
            {
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
    case SPLIT_NODE:
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // We must be the middle process or both the left and right process
            if( admissibleC )
            {
                // TODO: Start F += H H
            }
            else
            {
                // TODO: Start H += H H
            }
            break;
        case SPLIT_NODE_GHOST:
            // We must be in the left team
            if( admissibleC )
            {
                // TODO: Start F += H H
            }
            else
            {
                // TODO: Start H += H H
            }
            break;
        case NODE:
            // We must be in the middle and right teams
            if( admissibleC )
            {
                // TODO: Start F += H H
            }
            else
            {
                // TODO: Start H += H H
            }
            break;
        case NODE_GHOST:
            // We must be in the left team
            if( admissibleC )
            {
                // TODO: Start F += H H
            }
            else
            {
                // TODO: Start H += H H
            }
            break;
        case SPLIT_LOW_RANK:
            // We are either the middle process or both the left and right
            if( admissibleC )
            {
                // TODO: Start F += H F
            }
            else
            {
                // TODO: Start H += H F
            }
            break;
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

