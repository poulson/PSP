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
    C.Clear();

    MapHMatrixContext context;
    MapMatrixPrecompute( context, alpha, B, C );
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPrecompute
( MapHMatrixContext& context,
  Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPrecompute");
#endif
    const DistQuasi2d& A = *this;

    if( !A._inTargetTeam && !B._inSourceTeam )
    {
        C._shell.type = EMPTY;
        context.shell.type = EMPTY;
        return;
    }

    MPI_Comm team = A._subcomms->Subcomm( A._level );
    const int teamSize = mpi::CommSize( team );
    if( context.shell.type == EMPTY )
    {
        C._numLevels = A._numLevels;
        C._maxRank = A._maxRank;
        C._sourceOffset = B._sourceOffset;
        C._targetOffset = A._targetOffset;
        C._stronglyAdmissible = 
            ( A._stronglyAdmissible || B._stronglyAdmissible );

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

        if( C.Admissible() )
        {
            if( teamSize > 1 )
            {
                C._shell.type = DIST_LOW_RANK;
                C._shell.data.DF = new DistLowRank;
                context.shell.type = DIST_LOW_RANK;
                context.shell.data.DF = 
                    new typename MapHMatrixContext::DistLowRankContext;
            }
            else // teamSize == 1
            {
                if( C._inSourceTeam && C._inTargetTeam )
                {
                    C._shell.type = LOW_RANK;    
                    C._shell.data.F = new LowRank;
                    context.shell.type = LOW_RANK;
                    context.shell.data.F = 
                        new typename MapHMatrixContext::LowRankContext;
                }
                else
                {
                    C._shell.type = SPLIT_LOW_RANK;
                    C._shell.data.SF = new SplitLowRank;
                    context.shell.type = SPLIT_LOW_RANK;
                    context.shell.data.SF = 
                        new typename MapHMatrixContext::SplitLowRankContext;
                }
            }
        }
        else if( C._numLevels > 1 )
        {
            context.shell.type = NODE;
            context.shell.data.N = new typename MapHMatrixContext::NodeContext;
            typename MapHMatrixContext::NodeContext& nodeContext = 
                *context.shell.data.N;
            if( teamSize > 1 )
            {
                C._shell.type = DIST_NODE;
                C._shell.data.N = C.NewNode();
            }
            else // teamSize == 1
            {
                if( C._inSourceTeam && C._inTargetTeam )
                {
                    C._shell.type = NODE;
                    C._shell.data.N = C.NewNode();
                }
                else
                {
                    C._shell.type = SPLIT_NODE;
                    C._shell.data.N = C.NewNode();
                }
            }
        }
        else
        {
            if( C._inSourceTeam && C._inTargetTeam )
            {
                C._shell.type = DENSE;
                C._shell.data.D = new Dense;
                context.shell.type = DENSE;
                context.shell.data.D = 
                    new typename MapHMatrixContext::DenseContext;
            }
            else
            {
                C._shell.type = SPLIT_DENSE;
                C._shell.data.SD = new SplitDense;
                context.shell.type = SPLIT_DENSE;
                context.shell.data.SD = 
                    new typename MapHMatrixContext::SplitDenseContext;
            }
        }
    }

    switch( C._shell.type )
    {
    case DIST_NODE:
    {
        Node& nodeC = *C._shell.data.N;
        typename MapHMatrixContext::DistNodeContext& distNodeContext = 
            *context.shell.data.DN;

        // Allow for distributed {H,F}
        if( A._shell.type == DIST_NODE &&
            B._shell.type == DIST_NODE )
        {
            const Node& nodeA = *A._shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( distNodeContext.Child(t,s), 
                          alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
        }
        else if( A._shell.type == DIST_NODE &&
                 B._shell.type == DIST_LOW_RANK )
        {
            const DenseMatrix<Scalar>& ULocalB = B._shell.data.DF->ULocal;

            const int key = A._sourceOffset;
            distNodeContext.denseContextMap[key] = new MapDenseMatrixContext;
            distNodeContext.ULocalMap[key] = 
                new DenseMatrix<Scalar>( C.LocalHeight(), ULocalB.Width() );

            MapMatrixPrecompute
            ( *distNodeContext.denseContextMap[key],
              alpha, ULocalB, *distNodeContext.ULocalMap[key] );
        }
        else if( A._shell.type == DIST_LOW_RANK &&
                 B._shell.type == DIST_NODE )
        {
            // TODO: MapMatrixPrecompute for dense 
        }
        else if( A._shell.type == DIST_LOW_RANK &&
                 B._shell.type == DIST_LOW_RANK )
        {
            // TODO: Local multiply into context
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case SPLIT_NODE:
    {
        Node& nodeC = *C._shell.data.N;
        typename MapHMatrixContext::SplitNodeContext& nodeContext = 
            *context.shell.data.SN;

        // Allow for split/serial {H,F}, where at least one is split
        if( A._shell.type == SPLIT_NODE &&
            B._shell.type == SPLIT_NODE )
        {
            const Node& nodeA = *A._shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == NODE )
        {
            const Node& nodeA = *A._shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == SPLIT_NODE )
        {
            const Node& nodeA = *A._shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == NODE )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case NODE:
    {
        Node& nodeC = *C._shell.data.N;
        typename MapHMatrixContext::NodeContext& nodeContext = 
            *context.shell.data.N;

        // Allow for split/serial {H,F}, where either none or both are split
        if( A._shell.type == SPLIT_NODE &&
            B._shell.type == SPLIT_NODE )
        {
            const Node& nodeA = *A._shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == NODE )
        {
            const Node& nodeA = *A._shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == NODE )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case DIST_LOW_RANK:
    {
        /*
        DistLowRank& DFC = *C._shell.data.DF;
        typename MapHMatrixContext::DistLowRankContext& DFCContext = 
            *context.shell.data.DF;
        */

        // Allow for distributed {H,F}
        if( A._shell.type == DIST_NODE &&
            B._shell.type == DIST_NODE )
        {
            // TODO: Start the randomized low-rank discovery
        }
        else if( A._shell.type == DIST_NODE &&
                 B._shell.type == DIST_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == DIST_LOW_RANK &&
                 B._shell.type == DIST_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == DIST_LOW_RANK &&
                 B._shell.type == DIST_LOW_RANK )
        {
            // TODO: Local multiply
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case SPLIT_LOW_RANK:
    {
        /*
        SplitLowRank& SFC = *C._shell.data.SF;
        typename MapHMatrixContext::SplitLowRankContext& SFCContext = 
            *context.shell.data.SF;
        */

        // Allow for split/serial {H,F,D} where at least one is split and 
        // recall that D and H cannot occur at the same level.
        if( A._shell.type == SPLIT_NODE &&
            B._shell.type == SPLIT_NODE )
        {
            // TODO: Start randomized low-rank discovery
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == NODE )
        {
            // TODO: Start MapMatrixPrecompute
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Start MapMatrixPrecompute for dense
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Start MapMatrixPrecompute for dense
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: Start MapMatrixPrecompute
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense 
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == NODE )
        {
            // TODO: Local multiply 
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case LOW_RANK:
    {
        /*
        LowRank& FC = *C._shell.data.F;
        typename MapHMatrixContext::LowRankContext& FCContext = 
            *context.shell.data.F;
        */

        // Allow for split/serial {H,F,D} where either none or both is split and
        // recall that D and H cannot occur at the same level.
        if( A._shell.type == SPLIT_NODE &&
            B._shell.type == SPLIT_NODE )
        {
            // TODO: Start randomized low-rank discovery
        }
        else if( A._shell.type == SPLIT_NODE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == NODE )
        {
            // TODO: Randomized low-rank discovery
        }
        else if( A._shell.type == NODE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == NODE )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiplies
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == DENSE )
        {
            // TODO: Local truncated SVD
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case SPLIT_DENSE:
    {
        /*
        SplitDense& SDC = *C._shell.data.SD;
        typename MapHMatrixContext::SplitDenseContext& SDCContext = 
            *context.shell.data.SD;
        */

        // Allow for split/serial {F,D}, where at least one must be split
        if( A._shell.type == SPLIT_LOW_RANK &&
            B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case DENSE:
    {
        /*
        Dense& DC = *C._shell.data.D;
        typename MapHMatrixContext::DenseContext& DCContext = 
            *context.shell.data.D;
        */

        // Allow for {F,D} where either none or both are split
        if( A._shell.type == SPLIT_LOW_RANK &&
            B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_LOW_RANK &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiplies
        }
        else if( A._shell.type == LOW_RANK &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiplies
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == SPLIT_DENSE &&
                 B._shell.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._shell.type == DENSE &&
                 B._shell.type == DENSE )
        {
            // TODO: Local multiply
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

