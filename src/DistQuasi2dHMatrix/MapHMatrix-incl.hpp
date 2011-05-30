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
    if( !A.Ghosted() || !B.Ghosted() )
        throw std::logic_error("A and B must have their ghost nodes");
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
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
        C._block.type = EMPTY;
        context.block.type = EMPTY;
        return;
    }

    // HERE...need to branch depending on which blocks are ghosts
    MPI_Comm team = A._subcomms->Subcomm( A._level );
    const int teamSize = mpi::CommSize( team );
    if( context.block.type == EMPTY )
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
                C._block.type = DIST_LOW_RANK;
                C._block.data.DF = new DistLowRank;
                context.block.type = DIST_LOW_RANK;
                context.block.data.DF = 
                    new typename MapHMatrixContext::DistLowRankContext;
            }
            else // teamSize == 1
            {
                if( C._inSourceTeam && C._inTargetTeam )
                {
                    C._block.type = LOW_RANK;    
                    C._block.data.F = new LowRank;
                    context.block.type = LOW_RANK;
                    context.block.data.F = 
                        new typename MapHMatrixContext::LowRankContext;
                }
                else
                {
                    C._block.type = SPLIT_LOW_RANK;
                    C._block.data.SF = new SplitLowRank;
                    context.block.type = SPLIT_LOW_RANK;
                    context.block.data.SF = 
                        new typename MapHMatrixContext::SplitLowRankContext;
                }
            }
        }
        else if( C._numLevels > 1 )
        {
            context.block.type = NODE;
            context.block.data.N = new typename MapHMatrixContext::NodeContext;
            if( teamSize > 1 )
            {
                C._block.type = DIST_NODE;
                C._block.data.N = C.NewNode();
            }
            else // teamSize == 1
            {
                if( C._inSourceTeam && C._inTargetTeam )
                {
                    C._block.type = NODE;
                    C._block.data.N = C.NewNode();
                }
                else
                {
                    C._block.type = SPLIT_NODE;
                    C._block.data.N = C.NewNode();
                }
            }
        }
        else
        {
            if( C._inSourceTeam && C._inTargetTeam )
            {
                C._block.type = DENSE;
                C._block.data.D = new Dense;
                context.block.type = DENSE;
                context.block.data.D = 
                    new typename MapHMatrixContext::DenseContext;
            }
            else
            {
                C._block.type = SPLIT_DENSE;
                C._block.data.SD = new SplitDense;
                context.block.type = SPLIT_DENSE;
                context.block.data.SD = 
                    new typename MapHMatrixContext::SplitDenseContext;
            }
        }
    }

    switch( C._block.type )
    {
    case DIST_NODE:
    {
        Node& nodeC = *C._block.data.N;
        typename MapHMatrixContext::DistNodeContext& distNodeContext = 
            *context.block.data.DN;

        // Allow for distributed {H,F}
        if( A._block.type == DIST_NODE &&
            B._block.type == DIST_NODE )
        {
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( distNodeContext.Child(t,s), 
                          alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
        }
        else if( A._block.type == DIST_NODE &&
                 B._block.type == DIST_LOW_RANK )
        {
            const int key = A._sourceOffset;
            if( A._inSourceTeam || A._inTargetTeam )
            {
                distNodeContext.denseContextMap[key] = 
                    new MapDenseMatrixContext;
                A.MapMatrixInitialize( *distNodeContext.denseContextMap[key] );
            }
            if( A._inSourceTeam )
            {
                const DenseMatrix<Scalar>& ULocalB = B._block.data.DF->ULocal;
                distNodeContext.ULocalMap[key] = 
                    new DenseMatrix<Scalar>( C.LocalHeight(), ULocalB.Width() );
                A.MapMatrixPrecompute
                ( *distNodeContext.denseContextMap[key],
                  alpha, ULocalB, *distNodeContext.ULocalMap[key] );
            }
        }
        else if( A._block.type == DIST_LOW_RANK &&
                 B._block.type == DIST_NODE )
        {
            const int key = A._sourceOffset;
            if( A._inSourceTeam || A._inTargetTeam )
            {
                distNodeContext.denseContextMap[key] = 
                    new MapDenseMatrixContext;
                B.HermitianTransposeMapMatrixInitialize
                ( *distNodeContext.denseContextMap[key] );
            }
            if( A._inSourceTeam )
            {
                const DenseMatrix<Scalar>& VLocalA = A._block.data.DF->VLocal;
                distNodeContext.VLocalMap[key] = 
                    new DenseMatrix<Scalar>( C.LocalWidth(), VLocalA.Width() );
                B.HermitianTransposeMapMatrixPrecompute
                ( *distNodeContext.denseContextMap[key],
                  (Scalar)1, VLocalA, *distNodeContext.VLocalMap[key] );
            }
        }
        else if( A._block.type == DIST_LOW_RANK &&
                 B._block.type == DIST_LOW_RANK )
        {
            if( A._inSourceTeam )
            {
                const DenseMatrix<Scalar>& VLocalA = A._block.data.DF->VLocal;
                const DenseMatrix<Scalar>& ULocalB = B._block.data.DF->ULocal;

                const int key = A._sourceOffset;
                distNodeContext.ZMap[key] = 
                    new DenseMatrix<Scalar>( VLocalA.Width(), ULocalB.Width() );
                DenseMatrix<Scalar>& Z = *distNodeContext.ZMap[key];

                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( option, 'N', Z.Height(), Z.Width(), VLocalA.Height(),
                  (Scalar)1, VLocalA.LockedBuffer(), VLocalA.LDim(),
                             ULocalB.LockedBuffer(), ULocalB.LDim(),
                  (Scalar)0, Z.Buffer(),             Z.LDim() );
            }
        }
#ifndef RELEASE
        else
            throw std::logic_error("Invalid H-matrix combination");
#endif
        break;
    }
    case SPLIT_NODE:
    {
        Node& nodeC = *C._block.data.N;
        typename MapHMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;

        // Allow for split/serial {H,F}, where at least one is split
        if( A._block.type == SPLIT_NODE &&
            B._block.type == SPLIT_NODE )
        {
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == NODE )
        {
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == NODE &&
                 B._block.type == SPLIT_NODE )
        {
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._block.type == NODE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == NODE )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
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
        Node& nodeC = *C._block.data.N;
        typename MapHMatrixContext::NodeContext& nodeContext = 
            *context.block.data.N;

        // Allow for split/serial {H,F}, where either none or both are split
        if( A._block.type == SPLIT_NODE &&
            B._block.type == SPLIT_NODE )
        {
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == NODE &&
                 B._block.type == NODE )
        {
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    for( int r=0; r<4; ++r )
                        nodeA.Child(t,r).MapMatrixPrecompute
                        ( nodeContext.Child(t,s), alpha, nodeB.Child(r,s),
                          nodeC.Child(t,s) );
        }
        else if( A._block.type == NODE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == NODE )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == LOW_RANK )
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
        DistLowRank& DFC = *C._block.data.DF;
        typename MapHMatrixContext::DistLowRankContext& DFCContext = 
            *context.block.data.DF;
        */

        // Allow for distributed {H,F}
        if( A._block.type == DIST_NODE &&
            B._block.type == DIST_NODE )
        {
            // TODO: Start the randomized low-rank discovery
        }
        else if( A._block.type == DIST_NODE &&
                 B._block.type == DIST_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == DIST_LOW_RANK &&
                 B._block.type == DIST_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == DIST_LOW_RANK &&
                 B._block.type == DIST_LOW_RANK )
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
        SplitLowRank& SFC = *C._block.data.SF;
        typename MapHMatrixContext::SplitLowRankContext& SFCContext = 
            *context.block.data.SF;
        */

        // Allow for split/serial {H,F,D} where at least one is split and 
        // recall that D and H cannot occur at the same level.
        if( A._block.type == SPLIT_NODE &&
            B._block.type == SPLIT_NODE )
        {
            // TODO: Start randomized low-rank discovery
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == NODE )
        {
            // TODO: Start MapMatrixPrecompute
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Start MapMatrixPrecompute for dense
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Start MapMatrixPrecompute for dense
        }
        else if( A._block.type == NODE &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: Start MapMatrixPrecompute
        }
        else if( A._block.type == NODE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense 
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == NODE )
        {
            // TODO: Local multiply 
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == DENSE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == DENSE &&
                 B._block.type == SPLIT_DENSE )
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
        LowRank& FC = *C._block.data.F;
        typename MapHMatrixContext::LowRankContext& FCContext = 
            *context.block.data.F;
        */

        // Allow for split/serial {H,F,D} where either none or both is split and
        // recall that D and H cannot occur at the same level.
        if( A._block.type == SPLIT_NODE &&
            B._block.type == SPLIT_NODE )
        {
            // TODO: Start randomized low-rank discovery
        }
        else if( A._block.type == SPLIT_NODE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == NODE &&
                 B._block.type == NODE )
        {
            // TODO: Randomized low-rank discovery
        }
        else if( A._block.type == NODE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_NODE )
        {
            // TODO: MapMatrixPrecompute for dense
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == NODE )
        {
            // TODO: Local H-matrix multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiplies
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == DENSE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == DENSE &&
                 B._block.type == DENSE )
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
        SplitDense& SDC = *C._block.data.SD;
        typename MapHMatrixContext::SplitDenseContext& SDCContext = 
            *context.block.data.SD;
        */

        // Allow for split/serial {F,D}, where at least one must be split
        if( A._block.type == SPLIT_LOW_RANK &&
            B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == DENSE )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == DENSE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == DENSE &&
                 B._block.type == SPLIT_DENSE )
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
        Dense& DC = *C._block.data.D;
        typename MapHMatrixContext::DenseContext& DCContext = 
            *context.block.data.D;
        */

        // Allow for {F,D} where either none or both are split
        if( A._block.type == SPLIT_LOW_RANK &&
            B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_LOW_RANK &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiplies
        }
        else if( A._block.type == LOW_RANK &&
                 B._block.type == DENSE )
        {
            // TODO: Local multiplies
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == SPLIT_DENSE &&
                 B._block.type == SPLIT_DENSE )
        {
            // TODO: Nothing...
        }
        else if( A._block.type == DENSE &&
                 B._block.type == LOW_RANK )
        {
            // TODO: Local multiply
        }
        else if( A._block.type == DENSE &&
                 B._block.type == DENSE )
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

