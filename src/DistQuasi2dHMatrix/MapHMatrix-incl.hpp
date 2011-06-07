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
    A.RequireRoot();
    if( !A.Ghosted() || !B.Ghosted() )
        throw std::logic_error("A and B must have their ghost nodes");
    C.Clear();

    A.MapHMatrixMainPrecompute( alpha, B, C );
    /*
    A.MapHMatrixMainSummations( alpha, B, C );
    A.MapHMatrixMainPassData( alpha, B, C );
    A.MapHMatrixMainBroadcasts( alpha, B, C );
    A.MapHMatrixMainPostcompute( alpha, B, C );
    A.MapHMatrixFHHPrecompute( alpha, B, C );
    A.MapHMatrixFHHPassData( alpha, B, C );
    A.MapHMatrixFHHPostcompute( alpha, B, C );
    A.MapHMatrixFHHFinalize( alpha, B, C );
    A.MapHMatrixRoundedAddition( alpha, B, C );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixSetUp
( const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
        DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapHMatrixSetUp");
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
                DF.ULocal.Resize( C.LocalHeight(), 0 );
                DF.VLocal.Resize( C.LocalWidth(), 0 );
            }
            else
            {
                C._block.type = DIST_LOW_RANK_GHOST;
                C._block.data.DFG = new DistLowRankGhost;
                DistLowRankGhost& DFG = *C._block.data.DFG;
                DFG.rank = 0;
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
                }
                else
                {
                    C._block.type = LOW_RANK_GHOST;
                    C._block.data.FG = new LowRankGhost;
                    LowRankGhost& FG = *C._block.data.FG;
                    FG.rank = 0;
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
                }
                else
                {
                    C._block.type = SPLIT_LOW_RANK_GHOST;
                    C._block.data.SFG = new SplitLowRankGhost;
                    SplitLowRankGhost& SFG = *C._block.data.SFG;
                    SFG.rank = 0;
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
            }
            else
            {
                C._block.type = DIST_NODE_GHOST;
                C._block.data.N = C.NewNode();
                Node& node = *C._block.data.N;
                for( int j=0; j<16; ++j )
                    node.children[j] = new DistQuasi2d;
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
                }
                else
                {
                    C._block.type = NODE_GHOST;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = new DistQuasi2d;
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
                }
                else
                {
                    C._block.type = SPLIT_NODE_GHOST;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = new DistQuasi2d;
                }
            }
        }
    }
    else // C is dense
    {
        // Delay the allocation of C's dense blocks at least until after the
        // major communication buffers have been freed
        if( C._sourceRoot == C._targetRoot )
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = DENSE;
                C._block.data.D = new Dense;
            }
            else
            {
                C._block.type = DENSE_GHOST;
                C._block.data.DG = new DenseGhost;
            }
        }
        else
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = SPLIT_DENSE;
                C._block.data.SD = new SplitDense;
            }
            else
            {
                C._block.type = SPLIT_DENSE_GHOST;
                C._block.data.SDG = new SplitDenseGhost;
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
( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapHMatrixMainPrecompute");
#endif
    const int paddedRank = C.MaxRank() + 4;
    const DistQuasi2d& A = *this;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
        C._block.type = EMPTY;
        return;
    }
    if( C._block.type == EMPTY )
        A.MapHMatrixSetUp( B, C );

    // Take care of the H += H H cases first
    const bool admissibleC = C.Admissible();
    if( !admissibleC )
    {
        switch( A._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            switch( B._block.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
            {
                // Start H += H H
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MapHMatrixMainPrecompute
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
                return;
                break;
            }
            default:
                break;
            }
            break;
        default:
            break;
        }
    }

    const int key = A._sourceOffset;
    switch( A._block.type )
    {
    case DIST_NODE:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.LocalHeight(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );
                    A._T2.Resize( A.LocalWidth(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, A._T2 );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.LocalWidth(), paddedRank ); 
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.LocalHeight(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, B._T1 );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, B._T1 );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }
        case DIST_NODE_GHOST:
        {
            // We must be in the left team
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.LocalHeight(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );

                    Dense dummy( A.LocalWidth(), paddedRank );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, dummy );
                    A._beganRowSpaceComp = true;
                }
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            // Start H/F += H F
            const DistLowRank& DFB = *B._block.data.DF;
            C._UMap[key] = new Dense( C.LocalHeight(), DFB.rank );
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            hmatrix_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MapDenseMatrixInitialize( denseContext );
            A.MapDenseMatrixPrecompute
            ( denseContext, alpha, DFB.ULocal, *C._UMap[key] );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We must be in the left team
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            A.MapDenseMatrixInitialize( denseContext );
            break;
        }
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            // We must be in the right team
            if( admissibleC )
            {
                // Start F += H H
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.LocalWidth(), paddedRank ); 
                    ParallelGaussianRandomVectors( B._Omega1 );

                    Dense dummy( B.LocalHeight(), paddedRank );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, dummy );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }

        case DIST_LOW_RANK:
            // We must be in the right team, so there is nothing to do yet
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case SPLIT_NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be the middle process or both the left and right process
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.LocalHeight(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );
                    A._T2.Resize( A.LocalWidth(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, A._T2 );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.LocalWidth(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.LocalHeight(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, B._T1 );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, B._T1 );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }
        case SPLIT_NODE_GHOST:
        {
            // We must be in the left team
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.Height(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );

                    Dense dummy( A.Width(), paddedRank );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, dummy );
                    A._beganRowSpaceComp = true;
                }
            }
            break;
        }
        case NODE:
        {
            // We must be in the middle and right teams
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.Width(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.Height(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, B._T1 );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, B._T1 );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }
        case NODE_GHOST:
        {
            // We must be in the left team
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.Height(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );

                    Dense dummy( A.Width(), paddedRank );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, dummy );
                    A._beganRowSpaceComp = true;
                }
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            const SplitLowRank& SFB = *B._block.data.SF;
            if( admissibleC )
            {
                // Start F += H F
                if( C._inTargetTeam )
                {
                    // Our process owns the left and right sides
                    C._denseContextMap[key] = new MapDenseMatrixContext;
                    MapDenseMatrixContext& denseContext = 
                        *C._denseContextMap[key];

                    A.MapDenseMatrixInitialize( denseContext );
                }
                else
                {
                    // We are the middle process
                    C._denseContextMap[key] = new MapDenseMatrixContext;
                    MapDenseMatrixContext& denseContext =
                        *C._denseContextMap[key];

                    Dense dummy( A.Height(), SFB.rank );
                    A.MapDenseMatrixInitialize( denseContext );
                    A.MapDenseMatrixPrecompute
                    ( denseContext, alpha, SFB.D, dummy );
                }
            }
            else
            {
                // Start H += H F
                if( C._inTargetTeam )
                {
                    // Our process owns the left and right sides
                    C._denseContextMap[key] = new MapDenseMatrixContext;
                    MapDenseMatrixContext& denseContext = 
                        *C._denseContextMap[key];

                    A.MapDenseMatrixInitialize( denseContext );
                }
                else
                {
                    // We are the middle process
                    C._denseContextMap[key] = new MapDenseMatrixContext;
                    MapDenseMatrixContext& denseContext = 
                        *C._denseContextMap[key];

                    Dense dummy( A.Height(), SFB.rank );
                    A.MapDenseMatrixInitialize( denseContext );
                    A.MapDenseMatrixPrecompute
                    ( denseContext, alpha, SFB.D, dummy );
                }
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
            A.MapDenseMatrixInitialize( denseContext );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We are the middle and right processes
            const LowRank& FB = *B._block.data.F;
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            Dense dummy( A.Height(), FB.U.Width() );
            A.MapDenseMatrixInitialize( denseContext );
            A.MapDenseMatrixPrecompute( denseContext, alpha, FB.U, dummy );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
            A.MapDenseMatrixInitialize( denseContext );
            break;
        }
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case SPLIT_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be the right process
            if( admissibleC )
            {
                // Start F += H H
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.Width(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );

                    Dense dummy( B.Height(), paddedRank );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, dummy );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }

        case SPLIT_LOW_RANK:
            // We are the right process, so there is nothing to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be in the left and middle teams
            if( admissibleC )
            {
                // Start F += H H
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.Height(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );
                    A._T2.Resize( A.Width(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, alpha, A._Omega2, A._T2 );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }
        case NODE:
        {
            // We own all of A, B, and C
            if( admissibleC )
            {
                // Start the F += H H update
                if( !A._beganRowSpaceComp )
                {
                    A._Omega2.Resize( A.Height(), paddedRank );
                    ParallelGaussianRandomVectors( A._Omega2 );
                    A._T2.Resize( A.Width(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMapDenseMatrixInitialize( A._T2Context );
                    A.AdjointMapDenseMatrixPrecompute
                    ( A._T2Context, alpha, A._Omega2, A._T2 );
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.Width(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.Height(), paddedRank );

                    hmatrix_tools::Scale( (Scalar)0, B._T1 );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, B._T1 );
                }
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // Start H/F += H F
            // We are the left and middle processes
            const SplitLowRank& SFB = *B._block.data.SF;
            C._UMap[key] = new Dense( C.Height(), SFB.rank );
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            hmatrix_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MapDenseMatrixInitialize( denseContext );
            A.MapDenseMatrixPrecompute
            ( denseContext, alpha, SFB.D, *C._UMap[key] );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We own all of A, B, and C
            const LowRank& FB = *B._block.data.F;
            C._UMap[key] = new Dense( C.Height(), FB.Rank() );
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            hmatrix_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MapDenseMatrixInitialize( denseContext );
            A.MapDenseMatrixPrecompute
            ( denseContext, alpha, FB.U, *C._UMap[key] );
            break;
        }
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be the right process
            if( admissibleC )
            {
                // Start F += H H
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.Width(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );

                    Dense dummy( B.Height(), paddedRank );
                    B.MapDenseMatrixInitialize( B._T1Context );
                    B.MapDenseMatrixPrecompute
                    ( B._T1Context, alpha, B._Omega1, dummy );
                    B._beganColSpaceComp = true;
                }
            }
            break;
        }

        case SPLIT_LOW_RANK:
            // We are the right process, so there is nothing to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A._block.data.DF;
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            // Start H/F += F H
            C._VMap[key] = new Dense( C.LocalWidth(), DFA.rank );
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            hmatrix_tools::Scale( (Scalar)0, *C._VMap[key] );
            if( Conjugated )
            {
                B.AdjointMapDenseMatrixInitialize( denseContext );
                B.AdjointMapDenseMatrixPrecompute
                ( denseContext, Conj(alpha), DFA.VLocal, *C._VMap[key] );
            }
            else
            {
                B.TransposeMapDenseMatrixInitialize( denseContext );
                B.TransposeMapDenseMatrixPrecompute
                ( denseContext, alpha, DFA.VLocal, *C._VMap[key] );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            // Start H/F += F F
            if( A._inSourceTeam )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                C._ZMap[key] = new Dense( DFA.rank, DFB.rank );
                Dense& ZC = *C._ZMap[key];

                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( option, 'N', DFA.rank, DFB.rank, A.LocalWidth(),
                  (Scalar)1, DFA.VLocal.LockedBuffer(), DFA.VLocal.LDim(),
                             DFB.ULocal.LockedBuffer(), DFB.ULocal.LDim(),
                  (Scalar)0, ZC.Buffer(),               ZC.LDim() );
            }
            break;
        }

        case DIST_NODE_GHOST:
        case DIST_LOW_RANK_GHOST:
            // We're in the left team, so there is nothing to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            // Start H/F += F H
            // We are in the right team
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            if( Conjugated )
                B.AdjointMapDenseMatrixInitialize( denseContext );
            else
                B.TransposeMapDenseMatrixInitialize( denseContext );
            break;
        }

        case DIST_LOW_RANK:
            // We are in the right team, so there is nothing to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A._block.data.SF;
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We are either the middle process or both the left and right
            if( A._inSourceTeam )
            {
                C._denseContextMap[key] = new MapDenseMatrixContext;
                MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
                
                Dense dummy( B.Width(), SFA.rank );
                if( Conjugated )
                {
                    B.AdjointMapDenseMatrixInitialize( denseContext );
                    B.AdjointMapDenseMatrixPrecompute
                    ( denseContext, Conj(alpha), SFA.D, dummy );
                }
                else
                {
                    B.TransposeMapDenseMatrixInitialize( denseContext );
                    B.TransposeMapDenseMatrixPrecompute
                    ( denseContext, alpha, SFA.D, dummy );
                }
            }
            else
            {
                C._denseContextMap[key] = new MapDenseMatrixContext;
                MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

                if( Conjugated )
                    B.AdjointMapDenseMatrixInitialize( denseContext );
                else
                    B.TransposeMapDenseMatrixInitialize( denseContext );
            }
            break;
        }
        case NODE:
        {
            // We are the middle and right process
            C._VMap[key] = new Dense( B.Width(), SFA.rank );
            C._denseContextMap[key] = new MapDenseMatrixContext;
            Dense& CV = *C._VMap[key];
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
                
            hmatrix_tools::Scale( (Scalar)0, CV );
            if( Conjugated )
            {
                B.AdjointMapDenseMatrixInitialize( denseContext );
                B.AdjointMapDenseMatrixPrecompute
                ( denseContext, Conj(alpha), SFA.D, CV );
            }
            else
            {
                B.TransposeMapDenseMatrixInitialize( denseContext );
                B.TransposeMapDenseMatrixPrecompute
                ( denseContext, alpha, SFA.D, CV );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            if( A._inSourceTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._ZMap[key] = new Dense( SFA.rank, SFB.rank );
                Dense& ZC = *C._ZMap[key];
                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( option, 'N', SFA.rank, SFB.rank, A.Width(),
                  (Scalar)1, SFA.D.LockedBuffer(), SFA.D.LDim(),
                             SFB.D.LockedBuffer(), SFB.D.LDim(),
                  (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case LOW_RANK:
        {
            // We must be the middle and right process
            const LowRank& FB = *B._block.data.F;
            C._ZMap[key] = new Dense( SFA.rank, FB.Rank() );
            Dense& ZC = *C._ZMap[key];
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', SFA.rank, FB.Rank(), A.Width(),
              (Scalar)1, SFA.D.LockedBuffer(), SFA.D.LDim(),
                         FB.U.LockedBuffer(),  FB.U.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case DENSE:
        {
            // We must be both the middle and right process
            C._VMap[key] = new Dense( B.Width(), SFA.rank );
            const Dense& DB = *B._block.data.D;
            Dense& VC = *C._VMap[key];

            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', B.Width(), SFA.rank, B.Height(),
              (Scalar)1, DB.LockedBuffer(),    DB.LDim(),
                         SFA.D.LockedBuffer(), SFA.D.LDim(),
              (Scalar)0, VC.Buffer(),          VC.LDim() );
            break;
        }

        case SPLIT_NODE_GHOST:
        case NODE_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        case SPLIT_DENSE:
        case SPLIT_DENSE_GHOST:
        case DENSE_GHOST:
            // We are the left process, so there is no work to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case SPLIT_LOW_RANK_GHOST:
    {
        // We must be the right process
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];

            if( Conjugated )
                B.AdjointMapDenseMatrixInitialize( denseContext );
            else
                B.TransposeMapDenseMatrixInitialize( denseContext );
            break;
        }

        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:
            // There is nothing for us to compute yet
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case LOW_RANK:
    {
        const LowRank& FA = *A._block.data.F;
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be the left and middle process
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
                
            Dense dummy( B.Width(), FA.Rank() );
            if( Conjugated )
            {
                B.AdjointMapDenseMatrixInitialize( denseContext );
                B.AdjointMapDenseMatrixPrecompute
                ( denseContext, Conj(alpha), FA.V, dummy );
            }
            else
            {
                B.TransposeMapDenseMatrixInitialize( denseContext );
                B.TransposeMapDenseMatrixPrecompute
                ( denseContext, alpha, FA.V, dummy );
            }
            break;
        }
        case NODE:
        {
            // We must own all of A, B, and C
            C._VMap[key] = new Dense( B.Width(), FA.Rank() );
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
                
            hmatrix_tools::Scale( (Scalar)0, *C._VMap[key] );
            if( Conjugated )
            {
                B.AdjointMapDenseMatrixInitialize( denseContext );
                B.AdjointMapDenseMatrixPrecompute
                ( denseContext, Conj(alpha), FA.V, *C._VMap[key] );
            }
            else
            {
                B.TransposeMapDenseMatrixInitialize( denseContext );
                B.TransposeMapDenseMatrixPrecompute
                ( denseContext, alpha, FA.V, *C._VMap[key] );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We must be the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._ZMap[key] = new Dense( FA.Rank(), SFB.rank );
            Dense& ZC = *C._ZMap[key];

            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', FA.Rank(), SFB.rank, B.Height(),
              (Scalar)1, FA.V.LockedBuffer(),  FA.V.LDim(),
                         SFB.D.LockedBuffer(), SFB.D.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case LOW_RANK:
        {
            // We must own all of A, B, and C
            const LowRank& FB = *B._block.data.F;
            C._ZMap[key] = new Dense( FA.Rank(), FB.Rank() );
            Dense& ZC = *C._ZMap[key];

            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', FA.Rank(), FB.Rank(), B.Height(),
              (Scalar)1, FA.V.LockedBuffer(), FA.V.LDim(),
                         FB.U.LockedBuffer(), FB.U.LDim(),
              (Scalar)0, ZC.Buffer(),         ZC.LDim() );
            break;
        }
        case SPLIT_DENSE:
        {
            // We must be the left and middle process, but there is no
            // work to be done (split dense owned by right process)
            break;
        }
        case DENSE:
        {
            // We must own all of A, B, and C
            C._VMap[key] = new Dense( B.Width(), FA.Rank() );
            const Dense& DB = *B._block.data.D;
            Dense& VC = *C._VMap[key];

            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', B.Width(), FA.Rank(), B.Height(),
              (Scalar)1, DB.LockedBuffer(),   DB.LDim(),
                         FA.V.LockedBuffer(), FA.V.LDim(),
              (Scalar)0, VC.Buffer(),         VC.LDim() );
            break;
        }
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case LOW_RANK_GHOST:
    {
        // We must be the right process
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            C._denseContextMap[key] = new MapDenseMatrixContext;
            MapDenseMatrixContext& denseContext = *C._denseContextMap[key];
                
            if( Conjugated )
                B.AdjointMapDenseMatrixInitialize( denseContext );
            else
                B.TransposeMapDenseMatrixInitialize( denseContext );
            break;
        }

        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:
            // There is nothing to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDense& SDA = *A._block.data.SD;
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            if( A._inSourceTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._ZMap[key] = new Dense( A.Height(), SFB.rank );
                Dense& ZC = *C._ZMap[key];

                blas::Gemm
                ( 'N', 'N', A.Height(), SFB.rank, A.Width(),
                  alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                             SFB.D.LockedBuffer(), SFB.D.LDim(),
                  (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case LOW_RANK:
        {
            const LowRank& FB = *B._block.data.F;
            C._ZMap[key] = new Dense( A.Height(), FB.Rank() );
            Dense& ZC = *C._ZMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), FB.Rank(), A.Width(),
              alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                         FB.U.LockedBuffer(),  FB.U.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case DENSE:
        {
            const Dense& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                C._DMap[key] = new Dense( A.Height(), B.Width() );
                hmatrix_tools::MatrixMatrix
                ( alpha, SDA.D, DB, (Scalar)0, *C._DMap[key] );
            }
            else
            {
                // D += D D
                SplitDense& SDC = *C._block.data.SD;
                hmatrix_tools::MatrixMatrix
                ( alpha, SDA.D, DB, (Scalar)1, SDC.D );
            }
            break;
        }

        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        case SPLIT_DENSE:
        case SPLIT_DENSE_GHOST:
        case DENSE_GHOST:
            // There is nothing for us to do
            break;

        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
        // There is nothing for us to do
        break;
    case DENSE:
    {
        const Dense& DA = *A._block.data.D;
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._ZMap[key] = new Dense( A.Height(), SFB.rank );
            Dense& ZC = *C._ZMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), SFB.rank, A.Width(),
              alpha,     DA.LockedBuffer(),    DA.LDim(),
                         SFB.D.LockedBuffer(), SFB.D.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case LOW_RANK:
        {
            const LowRank& FB = *B._block.data.F;
            if( admissibleC )
            {
                LowRank& FC = *C._block.data.F;
                LowRank update;
                hmatrix_tools::MatrixMatrix( alpha, DA, FB, update );
                hmatrix_tools::MatrixUpdateRounded
                ( C.MaxRank(), (Scalar)1, update, (Scalar)1, FC );
            }
            else
            {
                Dense& DC = *C._block.data.D;
                hmatrix_tools::MatrixMatrix( alpha, DA, FB, (Scalar)1, DC );
            }
            break;
        }
        case SPLIT_DENSE:
        {
            // There is nothing to do
            break;
        }
        case DENSE:
        {
            const Dense& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                C._DMap[key] = new Dense( A.Height(), B.Width() );
                hmatrix_tools::MatrixMatrix
                ( alpha, DA, DB, (Scalar)0, *C._DMap[key] );
            }
            else
            {
                // D += D D
                Dense& DC = *C._block.data.D;
                hmatrix_tools::MatrixMatrix( alpha, DA, DB, (Scalar)1, DC );
            }
            break;
        }
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case DENSE_GHOST:
        // There is nothing for us to do
        break;
    case EMPTY:
#ifndef RELEASE
        throw std::logic_error("A should not be empty");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

