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
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    if( !A.Ghosted() || !B.Ghosted() )
        throw std::logic_error("A and B must have their ghost nodes");
    C.Clear();

    A.MultiplyHMatMainPrecompute( alpha, B, C );

    A.MultiplyHMatMainSummations( B, C );
    A.MultiplyHMatMainPassData( alpha, B, C );
    A.MultiplyHMatMainBroadcasts( B, C );

    /*
    A.MultiplyHMatMainPostcompute( alpha, B, C );

    A.MultiplyHMatFHHPrecompute( alpha, B, C );
    A.MultiplyHMatFHHPassData( alpha, B, C );
    A.MultiplyHMatFHHPostcompute( alpha, B, C );
    A.MultiplyHMatFHHFinalize( alpha, B, C );

    A.MultiplyHMatRoundedAddition( alpha, B, C );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatSetUp
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatSetUp");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

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

    C._teams = A._teams;
    C._level = A._level;
    C._inSourceTeam = B._inSourceTeam;
    C._inTargetTeam = A._inTargetTeam;
    C._sourceRoot = B._sourceRoot;
    C._targetRoot = A._targetRoot;
    C._localSourceOffset = B._localSourceOffset;
    C._localTargetOffset = A._localTargetOffset;
    
    MPI_Comm team = _teams->Team( _level );
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
                    C._block.data.F = new LowRank<Scalar,Conjugated>;
                    LowRank<Scalar,Conjugated>& F = *C._block.data.F;
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
                    node.children[j] = new DistQuasi2dHMat<Scalar,Conjugated>;
            }
            else
            {
                C._block.type = DIST_NODE_GHOST;
                C._block.data.N = C.NewNode();
                Node& node = *C._block.data.N;
                for( int j=0; j<16; ++j )
                    node.children[j] = new DistQuasi2dHMat<Scalar,Conjugated>;
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
                        node.children[j] = 
                            new DistQuasi2dHMat<Scalar,Conjugated>;
                }
                else
                {
                    C._block.type = NODE_GHOST;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = 
                            new DistQuasi2dHMat<Scalar,Conjugated>;
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
                        node.children[j] = 
                            new DistQuasi2dHMat<Scalar,Conjugated>;
                }
                else
                {
                    C._block.type = SPLIT_NODE_GHOST;
                    C._block.data.N = C.NewNode();
                    Node& node = *C._block.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = 
                            new DistQuasi2dHMat<Scalar,Conjugated>;
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
                C._block.data.D = new Dense<Scalar>;
            }
            else
                C._block.type = DENSE_GHOST;
        }
        else
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = SPLIT_DENSE;
                C._block.data.SD = new SplitDense;
            }
            else
                C._block.type = SPLIT_DENSE_GHOST;
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPrecompute
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPrecompute");
#endif
    const int paddedRank = C.MaxRank() + 4;
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
        C._block.type = EMPTY;
        return;
    }
    if( C._block.type == EMPTY )
        A.MultiplyHMatSetUp( B, C );

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
                            nodeA.Child(t,r).MultiplyHMatMainPrecompute
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

                    hmat_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, A._T2 );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.LocalWidth(), paddedRank ); 
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.LocalHeight(), paddedRank );

                    hmat_tools::Scale( (Scalar)0, B._T1 );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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

                    Dense<Scalar> dummy( A.LocalWidth(), paddedRank );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
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
            C._UMap[key] = new Dense<Scalar>( C.LocalHeight(), DFB.rank );
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( densePair.second, DFB.rank );
            A.MultiplyDensePrecompute
            ( densePair.second, alpha, DFB.ULocal, *C._UMap[key] );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We must be in the left team
            const DistLowRankGhost& DFGB = *B._block.data.DFG;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            A.MultiplyDenseInitialize( densePair.second, DFGB.rank );
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

                    Dense<Scalar> dummy( B.LocalHeight(), paddedRank );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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

                    hmat_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
                    ( A._T2Context, Conj(alpha), A._Omega2, A._T2 );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.LocalWidth(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.LocalHeight(), paddedRank );

                    hmat_tools::Scale( (Scalar)0, B._T1 );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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

                    Dense<Scalar> dummy( A.Width(), paddedRank );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
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
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.Width(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.Height(), paddedRank );

                    hmat_tools::Scale( (Scalar)0, B._T1 );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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

                    Dense<Scalar> dummy( A.Width(), paddedRank );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
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
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                }
                else
                {
                    // We are the middle process
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    Dense<Scalar> dummy( A.Height(), SFB.rank );
                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                    A.MultiplyDensePrecompute
                    ( densePair.second, alpha, SFB.D, dummy );
                }
            }
            else
            {
                // Start H += H F
                if( C._inTargetTeam )
                {
                    // Our process owns the left and right sides
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                }
                else
                {
                    // We are the middle process
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    Dense<Scalar> dummy( A.Height(), SFB.rank );
                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                    A.MultiplyDensePrecompute
                    ( densePair.second, alpha, SFB.D, dummy );
                }
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;
            A.MultiplyDenseInitialize( densePair.second, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We are the middle and right processes
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            Dense<Scalar> dummy( A.Height(), FB.U.Width() );
            A.MultiplyDenseInitialize( densePair.second, FB.Rank() );
            A.MultiplyDensePrecompute( densePair.second, alpha, FB.U, dummy );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;
            A.MultiplyDenseInitialize( densePair.second, FGB.rank );
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

                    Dense<Scalar> dummy( B.Height(), paddedRank );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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

                    hmat_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
                    ( A._T2Context, alpha, A._Omega2, A._T2 );
                    A._beganRowSpaceComp = true;
                }
                if( !B._beganColSpaceComp )
                {
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
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

                    hmat_tools::Scale( (Scalar)0, A._T2 );
                    A.AdjointMultiplyDenseInitialize
                    ( A._T2Context, paddedRank );
                    A.AdjointMultiplyDensePrecompute
                    ( A._T2Context, alpha, A._Omega2, A._T2 );
                }
                if( !B._beganColSpaceComp )
                {
                    B._Omega1.Resize( B.Width(), paddedRank );
                    ParallelGaussianRandomVectors( B._Omega1 );
                    B._T1.Resize( B.Height(), paddedRank );

                    hmat_tools::Scale( (Scalar)0, B._T1 );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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
            C._UMap[key] = new Dense<Scalar>( C.Height(), SFB.rank );
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( densePair.second, SFB.rank );
            A.MultiplyDensePrecompute
            ( densePair.second, alpha, SFB.D, *C._UMap[key] );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._UMap[key] = new Dense<Scalar>( C.Height(), FB.Rank() );
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( densePair.second, FB.Rank() );
            A.MultiplyDensePrecompute
            ( densePair.second, alpha, FB.U, *C._UMap[key] );
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

                    Dense<Scalar> dummy( B.Height(), paddedRank );
                    B.MultiplyDenseInitialize( B._T1Context, paddedRank );
                    B.MultiplyDensePrecompute
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
            C._VMap[key] = new Dense<Scalar>( C.LocalWidth(), DFA.rank );

            hmat_tools::Scale( (Scalar)0, *C._VMap[key] );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, DFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), DFA.VLocal, *C._VMap[key] );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, DFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, DFA.VLocal, *C._VMap[key] );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            // Start H/F += F F
            if( A._inSourceTeam )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                C._ZMap[key] = new Dense<Scalar>( DFA.rank, DFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

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
            const DistLowRankGhost& DFGA = *A._block.data.DFG;
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, DFGA.rank );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, DFGA.rank );
            }
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
                Dense<Scalar> dummy( B.Width(), SFA.rank );
                if( Conjugated )
                {
                    C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                    AdjointMultiplyDensePair& pair = 
                        *C._adjointDensePairMap[key];
                    pair.first = &B;

                    B.AdjointMultiplyDenseInitialize( pair.second, SFA.rank );
                    B.AdjointMultiplyDensePrecompute
                    ( pair.second, Conj(alpha), SFA.D, dummy );
                }
                else
                {
                    C._transposeDensePairMap[key] = 
                        new TransposeMultiplyDensePair;
                    TransposeMultiplyDensePair& pair = 
                        *C._transposeDensePairMap[key];
                    pair.first = &B;

                    B.TransposeMultiplyDenseInitialize( pair.second, SFA.rank );
                    B.TransposeMultiplyDensePrecompute
                    ( pair.second, alpha, SFA.D, dummy );
                }
            }
            else
            {
                if( Conjugated )
                {
                    C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                    AdjointMultiplyDensePair& pair = 
                        *C._adjointDensePairMap[key];
                    pair.first = &B;

                    B.AdjointMultiplyDenseInitialize( pair.second, SFA.rank );
                }
                else
                {
                    C._transposeDensePairMap[key] = 
                        new TransposeMultiplyDensePair;
                    TransposeMultiplyDensePair& pair = 
                        *C._transposeDensePairMap[key];
                    pair.first = &B;

                    B.TransposeMultiplyDenseInitialize( pair.second, SFA.rank );
                }
            }
            break;
        }
        case NODE:
        {
            // We are the middle and right process
            C._VMap[key] = new Dense<Scalar>( B.Width(), SFA.rank );
            Dense<Scalar>& CV = *C._VMap[key];
                
            hmat_tools::Scale( (Scalar)0, CV );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, SFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), SFA.D, CV );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, SFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, SFA.D, CV );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            if( A._inSourceTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._ZMap[key] = new Dense<Scalar>( SFA.rank, SFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];
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
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._ZMap[key] = new Dense<Scalar>( SFA.rank, FB.Rank() );
            Dense<Scalar>& ZC = *C._ZMap[key];
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
            C._VMap[key] = new Dense<Scalar>( B.Width(), SFA.rank );
            const Dense<Scalar>& DB = *B._block.data.D;
            Dense<Scalar>& VC = *C._VMap[key];

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
            const SplitLowRankGhost& SFGA = *A._block.data.SFG;
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, SFGA.rank );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, SFGA.rank );
            }
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
        const LowRank<Scalar,Conjugated>& FA = *A._block.data.F;
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be the left and middle process
            Dense<Scalar> dummy( B.Width(), FA.Rank() );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), FA.V, dummy );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, FA.V, dummy );
            }
            break;
        }
        case NODE:
        {
            // We must own all of A, B, and C
            C._VMap[key] = new Dense<Scalar>( B.Width(), FA.Rank() );
                
            hmat_tools::Scale( (Scalar)0, *C._VMap[key] );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), FA.V, *C._VMap[key] );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, FA.V, *C._VMap[key] );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We must be the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._ZMap[key] = new Dense<Scalar>( FA.Rank(), SFB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

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
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._ZMap[key] = new Dense<Scalar>( FA.Rank(), FB.Rank() );
            Dense<Scalar>& ZC = *C._ZMap[key];

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
            C._VMap[key] = new Dense<Scalar>( B.Width(), FA.Rank() );
            const Dense<Scalar>& DB = *B._block.data.D;
            Dense<Scalar>& VC = *C._VMap[key];

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
            const LowRankGhost& FGA = *A._block.data.FG;
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, FGA.rank );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, FGA.rank );
            }
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
                C._ZMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

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
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._ZMap[key] = new Dense<Scalar>( A.Height(), FB.Rank() );
            Dense<Scalar>& ZC = *C._ZMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), FB.Rank(), A.Width(),
              alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                         FB.U.LockedBuffer(),  FB.U.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case DENSE:
        {
            const Dense<Scalar>& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                C._DMap[key] = new Dense<Scalar>( A.Height(), B.Width() );
                hmat_tools::Multiply
                ( alpha, SDA.D, DB, (Scalar)0, *C._DMap[key] );
            }
            else
            {
                // D += D D
                SplitDense& SDC = *C._block.data.SD;
                hmat_tools::Multiply
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
        const Dense<Scalar>& DA = *A._block.data.D;
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._ZMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), SFB.rank, A.Width(),
              alpha,     DA.LockedBuffer(),    DA.LDim(),
                         SFB.D.LockedBuffer(), SFB.D.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case LOW_RANK:
        {
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            if( admissibleC )
            {
                LowRank<Scalar,Conjugated>& FC = *C._block.data.F;
                LowRank<Scalar,Conjugated> update;
                hmat_tools::Multiply( alpha, DA, FB, update );
                hmat_tools::RoundedUpdate
                ( C.MaxRank(), (Scalar)1, update, (Scalar)1, FC );
            }
            else
            {
                Dense<Scalar>& DC = *C._block.data.D;
                hmat_tools::Multiply( alpha, DA, FB, (Scalar)1, DC );
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
            const Dense<Scalar>& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                C._DMap[key] = new Dense<Scalar>( A.Height(), B.Width() );
                hmat_tools::Multiply
                ( alpha, DA, DB, (Scalar)0, *C._DMap[key] );
            }
            else
            {
                // D += D D
                Dense<Scalar>& DC = *C._block.data.D;
                hmat_tools::Multiply( alpha, DA, DB, (Scalar)1, DC );
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

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummations
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummations");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each reduce
    // (the first and last comms are unneeded)
    const int numLevels = _teams->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    A.MultiplyHMatMainSummationsCountA( sizes );
    B.MultiplyHMatMainSummationsCountB( sizes );
    C.MultiplyHMatMainSummationsCountC( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatMainSummationsPackA( buffer, offsets );
    B.MultiplyHMatMainSummationsPackB( buffer, offsets );
    C.MultiplyHMatMainSummationsPackC( buffer, offsets );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    for( int i=0; i<numReduces; ++i )
    {
        if( sizes[i] != 0 )
        {
            MPI_Comm team = _teams->Team( i+1 );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, &buffer[offsets[i]], sizes[i],
                  0, MPI_SUM, team );
            else
                mpi::Reduce
                ( &buffer[offsets[i]], 0, sizes[i], 0, MPI_SUM, team );
        }
    }

    // Unpack the reduced buffers (only roots of teamunicators have data)
    A.MultiplyHMatMainSummationsUnpackA( buffer, offsets );
    B.MultiplyHMatMainSummationsUnpackB( buffer, offsets );
    C.MultiplyHMatMainSummationsUnpackC( buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsCountA
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseSummationsCount( sizes, _T2Context.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsCountA( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseSummationsPack( _T2Context, buffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsPackA
                ( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseSummationsUnpack
            ( _T2Context, buffer, offsets );

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsUnpackA
                ( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsCountB
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseSummationsCount( sizes, _T1Context.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsCountB( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseSummationsPack( _T1Context, buffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsPackB
                ( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseSummationsUnpack( _T1Context, buffer, offsets );

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsUnpackB
                ( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsCountC
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsCountC");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    {
        // Handle the distributed summations at this level
        {
            const std::map<int,Dense<Scalar>*>& baseMap = _ZMap.BaseMap();
            typename std::map<int,Dense<Scalar>*>::const_iterator itr;
            for( itr=baseMap.begin(); itr!=baseMap.end(); ++itr )
            { 
                const Dense<Scalar>& Z = *(*itr).second;
                sizes[_level-1] += Z.Height()*Z.Width();
            }
        }

        // Handle the H-mat/dense multiplications
        {
            const std::map<int,MultiplyDensePair*>& baseMap = 
                _densePairMap.BaseMap();
            typename std::map<int,MultiplyDensePair*>::const_iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                const MultiplyDensePair& pair = *(*itr).second;
                pair.first->MultiplyDenseSummationsCount
                ( sizes, pair.second.numRhs );
            }
        }

        // Handle the transposed H-mat/dense multiplications
        {
            const std::map<int,TransposeMultiplyDensePair*>& baseMap = 
                _transposeDensePairMap.BaseMap();
            typename 
                std::map<int,TransposeMultiplyDensePair*>::const_iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                const TransposeMultiplyDensePair& pair = *(*itr).second;
                pair.first->TransposeMultiplyDenseSummationsCount
                ( sizes, pair.second.numRhs );
            }
        }

        // Handle the adjoint H-mat/dense multiplications
        {
            const std::map<int,AdjointMultiplyDensePair*>& baseMap = 
                _adjointDensePairMap.BaseMap();
            typename 
                std::map<int,AdjointMultiplyDensePair*>::const_iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                const AdjointMultiplyDensePair& pair = *(*itr).second;
                pair.first->TransposeMultiplyDenseSummationsCount
                ( sizes, pair.second.numRhs );
            }
        }
        break;
    }
    default:
        break;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsCountC( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsPackC
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsPackC");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    {
        // Handle the distributed summations at this level
        {
            const std::map<int,Dense<Scalar>*>& baseMap = _ZMap.BaseMap();
            typename std::map<int,Dense<Scalar>*>::const_iterator itr;
            for( itr=baseMap.begin(); itr!=baseMap.end(); ++itr )
            { 
                const Dense<Scalar>& Z = *(*itr).second;
                if( Z.Height() == Z.LDim() )
                {
                    std::memcpy
                    ( &buffer[offsets[_level-1]], Z.LockedBuffer(),
                      Z.Height()*Z.Width()*sizeof(Scalar) );
                    offsets[_level-1] += Z.Height()*Z.Width();
                }
                else
                {
                    for( int j=0; j<Z.Width(); ++j )
                    {
                        std::memcpy
                        ( &buffer[offsets[_level-1]], Z.LockedBuffer(0,j),
                          Z.Height()*sizeof(Scalar) );
                        offsets[_level-1] += Z.Height();
                    }
                }
            }
        }

        // Handle the H-mat/dense multiplications
        {
            const std::map<int,MultiplyDensePair*>& baseMap = 
                _densePairMap.BaseMap();
            typename std::map<int,MultiplyDensePair*>::const_iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                const MultiplyDensePair& pair = *(*itr).second;
                pair.first->MultiplyDenseSummationsPack
                ( pair.second, buffer, offsets );
            }
        }

        // Handle the transposed H-mat/dense multiplications
        {
            const std::map<int,TransposeMultiplyDensePair*>& baseMap = 
                _transposeDensePairMap.BaseMap();
            typename 
                std::map<int,TransposeMultiplyDensePair*>::const_iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                const TransposeMultiplyDensePair& pair = *(*itr).second;
                pair.first->TransposeMultiplyDenseSummationsPack
                ( pair.second, buffer, offsets );
            }
        }

        // Handle the adjoint H-mat/dense multiplications
        {
            const std::map<int,AdjointMultiplyDensePair*>& baseMap = 
                _adjointDensePairMap.BaseMap();
            typename 
                std::map<int,AdjointMultiplyDensePair*>::const_iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                const AdjointMultiplyDensePair& pair = *(*itr).second;
                pair.first->TransposeMultiplyDenseSummationsPack
                ( pair.second, buffer, offsets );
            }
        }
        break;
    }
    default:
        break;
    }
    
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsPackC
                ( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsUnpackC
( const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsUnpackC");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    {
        // Handle the distributed summations at this level
        {
            std::map<int,Dense<Scalar>*>& baseMap = _ZMap.BaseMap();
            typename std::map<int,Dense<Scalar>*>::iterator itr;
            for( itr=baseMap.begin(); itr!=baseMap.end(); ++itr )
            { 
                Dense<Scalar>& Z = *(*itr).second;
                if( Z.Height() == Z.LDim() )
                {
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_level-1]],
                      Z.Height()*Z.Width()*sizeof(Scalar) );
                    offsets[_level-1] += Z.Height()*Z.Width();
                }
                else
                {
                    for( int j=0; j<Z.Width(); ++j )
                    {
                        std::memcpy
                        ( Z.Buffer(0,j), &buffer[offsets[_level-1]],
                          Z.Height()*sizeof(Scalar) );
                        offsets[_level-1] += Z.Height();
                    }
                }
            }
        }

        // Handle the H-mat/dense multiplications
        {
            std::map<int,MultiplyDensePair*>& baseMap = _densePairMap.BaseMap();
            typename std::map<int,MultiplyDensePair*>::iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                MultiplyDensePair& pair = *(*itr).second;
                pair.first->MultiplyDenseSummationsUnpack
                ( pair.second, buffer, offsets );
            }
        }

        // Handle the transposed H-mat/dense multiplications
        {
            std::map<int,TransposeMultiplyDensePair*>& baseMap = 
                _transposeDensePairMap.BaseMap();
            typename std::map<int,TransposeMultiplyDensePair*>::iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                TransposeMultiplyDensePair& pair = *(*itr).second;
                pair.first->TransposeMultiplyDenseSummationsUnpack
                ( pair.second, buffer, offsets );
            }
        }

        // Handle the adjoint H-mat/dense multiplications
        {
            std::map<int,AdjointMultiplyDensePair*>& baseMap = 
                _adjointDensePairMap.BaseMap();
            typename std::map<int,AdjointMultiplyDensePair*>::iterator itr;
            for( itr!=baseMap.begin(); itr!=baseMap.end(); ++itr )
            {
                AdjointMultiplyDensePair& pair = *(*itr).second;
                pair.first->TransposeMultiplyDenseSummationsUnpack
                ( pair.second, buffer, offsets );
            }
        }
        break;
    }
    default:
        break;
    }
    
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSummationsUnpackC
                ( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassData
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassData");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // 1) Compute send and recv sizes
    MPI_Comm comm = _teams->Team( 0 );
    const int p = mpi::CommSize( comm );
    std::vector<int> sendSizes(p,0), recvSizes(p,0);
    A.MultiplyHMatMainPassDataCountA( sendSizes, recvSizes );
    B.MultiplyHMatMainPassDataCountB( sendSizes, recvSizes );
    A.MultiplyHMatMainPassDataCountC( B, C, sendSizes, recvSizes );

    // 2) Allocate buffers
    // TODO

    // 3) Pack sends
    // TODO

    // 4) MPI_Alltoallv
    // TODO

    // 5) Unpack recvs
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataCountA
( std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, _T2Context.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainPassDataCountA
                ( sendSizes, recvSizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataCountB
( std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _beganColSpaceComp )
            MultiplyDensePassDataCount
            ( sendSizes, recvSizes, _T1Context.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainPassDataCountB
                ( sendSizes, recvSizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataCountC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataCountC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

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
                            nodeA.Child(t,r).MultiplyHMatMainPassDataCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendSizes, recvSizes );
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

    // HERE: Need to carefully think about possiblity of cases where there
    //       was no computation in the Precompute phase, but data still needs
    //       to be passed here.
    /*
    const int key = A._sourceOffset;
    switch( A._block.type )
    {
    case DIST_NODE:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            break;
        case DIST_LOW_RANK:
        {
            // HERE
            // Start H/F += H F
            const DistLowRank& DFB = *B._block.data.DF;
            C._UMap[key] = new Dense<Scalar>( C.LocalHeight(), DFB.rank );
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( densePair.second, DFB.rank );
            A.MultiplyDensePrecompute
            ( densePair.second, alpha, DFB.ULocal, *C._UMap[key] );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We must be in the left team
            const DistLowRankGhost& DFGB = *B._block.data.DFG;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            A.MultiplyDenseInitialize( densePair.second, DFGB.rank );
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
        break;
    case SPLIT_NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            break;
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
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                }
                else
                {
                    // We are the middle process
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    Dense<Scalar> dummy( A.Height(), SFB.rank );
                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                    A.MultiplyDensePrecompute
                    ( densePair.second, alpha, SFB.D, dummy );
                }
            }
            else
            {
                // Start H += H F
                if( C._inTargetTeam )
                {
                    // Our process owns the left and right sides
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                }
                else
                {
                    // We are the middle process
                    C._densePairMap[key] = new MultiplyDensePair;
                    MultiplyDensePair& densePair = *C._densePairMap[key];
                    densePair.first = &A;

                    Dense<Scalar> dummy( A.Height(), SFB.rank );
                    A.MultiplyDenseInitialize( densePair.second, SFB.rank );
                    A.MultiplyDensePrecompute
                    ( densePair.second, alpha, SFB.D, dummy );
                }
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;
            A.MultiplyDenseInitialize( densePair.second, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We are the middle and right processes
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            Dense<Scalar> dummy( A.Height(), FB.U.Width() );
            A.MultiplyDenseInitialize( densePair.second, FB.Rank() );
            A.MultiplyDensePrecompute( densePair.second, alpha, FB.U, dummy );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;
            A.MultiplyDenseInitialize( densePair.second, FGB.rank );
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
        break;
    case NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case NODE:
            break;
        case SPLIT_LOW_RANK:
        {
            // Start H/F += H F
            // We are the left and middle processes
            const SplitLowRank& SFB = *B._block.data.SF;
            C._UMap[key] = new Dense<Scalar>( C.Height(), SFB.rank );
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( densePair.second, SFB.rank );
            A.MultiplyDensePrecompute
            ( densePair.second, alpha, SFB.D, *C._UMap[key] );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._UMap[key] = new Dense<Scalar>( C.Height(), FB.Rank() );
            C._densePairMap[key] = new MultiplyDensePair;
            MultiplyDensePair& densePair = *C._densePairMap[key];
            densePair.first = &A;

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( densePair.second, FB.Rank() );
            A.MultiplyDensePrecompute
            ( densePair.second, alpha, FB.U, *C._UMap[key] );
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
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A._block.data.DF;
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            // Start H/F += F H
            C._VMap[key] = new Dense<Scalar>( C.LocalWidth(), DFA.rank );

            hmat_tools::Scale( (Scalar)0, *C._VMap[key] );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, DFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), DFA.VLocal, *C._VMap[key] );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, DFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, DFA.VLocal, *C._VMap[key] );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            // Start H/F += F F
            if( A._inSourceTeam )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                C._ZMap[key] = new Dense<Scalar>( DFA.rank, DFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

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
            const DistLowRankGhost& DFGA = *A._block.data.DFG;
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, DFGA.rank );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, DFGA.rank );
            }
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
                Dense<Scalar> dummy( B.Width(), SFA.rank );
                if( Conjugated )
                {
                    C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                    AdjointMultiplyDensePair& pair = 
                        *C._adjointDensePairMap[key];
                    pair.first = &B;

                    B.AdjointMultiplyDenseInitialize( pair.second, SFA.rank );
                    B.AdjointMultiplyDensePrecompute
                    ( pair.second, Conj(alpha), SFA.D, dummy );
                }
                else
                {
                    C._transposeDensePairMap[key] = 
                        new TransposeMultiplyDensePair;
                    TransposeMultiplyDensePair& pair = 
                        *C._transposeDensePairMap[key];
                    pair.first = &B;

                    B.TransposeMultiplyDenseInitialize( pair.second, SFA.rank );
                    B.TransposeMultiplyDensePrecompute
                    ( pair.second, alpha, SFA.D, dummy );
                }
            }
            else
            {
                if( Conjugated )
                {
                    C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                    AdjointMultiplyDensePair& pair = 
                        *C._adjointDensePairMap[key];
                    pair.first = &B;

                    B.AdjointMultiplyDenseInitialize( pair.second, SFA.rank );
                }
                else
                {
                    C._transposeDensePairMap[key] = 
                        new TransposeMultiplyDensePair;
                    TransposeMultiplyDensePair& pair = 
                        *C._transposeDensePairMap[key];
                    pair.first = &B;

                    B.TransposeMultiplyDenseInitialize( pair.second, SFA.rank );
                }
            }
            break;
        }
        case NODE:
        {
            // We are the middle and right process
            C._VMap[key] = new Dense<Scalar>( B.Width(), SFA.rank );
            Dense<Scalar>& CV = *C._VMap[key];
                
            hmat_tools::Scale( (Scalar)0, CV );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, SFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), SFA.D, CV );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, SFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, SFA.D, CV );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            if( A._inSourceTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._ZMap[key] = new Dense<Scalar>( SFA.rank, SFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];
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
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._ZMap[key] = new Dense<Scalar>( SFA.rank, FB.Rank() );
            Dense<Scalar>& ZC = *C._ZMap[key];
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
            C._VMap[key] = new Dense<Scalar>( B.Width(), SFA.rank );
            const Dense<Scalar>& DB = *B._block.data.D;
            Dense<Scalar>& VC = *C._VMap[key];

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
            const SplitLowRankGhost& SFGA = *A._block.data.SFG;
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, SFGA.rank );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, SFGA.rank );
            }
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
        const LowRank<Scalar,Conjugated>& FA = *A._block.data.F;
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            // We must be the left and middle process
            Dense<Scalar> dummy( B.Width(), FA.Rank() );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), FA.V, dummy );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, FA.V, dummy );
            }
            break;
        }
        case NODE:
        {
            // We must own all of A, B, and C
            C._VMap[key] = new Dense<Scalar>( B.Width(), FA.Rank() );
                
            hmat_tools::Scale( (Scalar)0, *C._VMap[key] );
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( pair.second, Conj(alpha), FA.V, *C._VMap[key] );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( pair.second, alpha, FA.V, *C._VMap[key] );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We must be the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._ZMap[key] = new Dense<Scalar>( FA.Rank(), SFB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

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
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._ZMap[key] = new Dense<Scalar>( FA.Rank(), FB.Rank() );
            Dense<Scalar>& ZC = *C._ZMap[key];

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
            C._VMap[key] = new Dense<Scalar>( B.Width(), FA.Rank() );
            const Dense<Scalar>& DB = *B._block.data.D;
            Dense<Scalar>& VC = *C._VMap[key];

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
            const LowRankGhost& FGA = *A._block.data.FG;
            if( Conjugated )
            {
                C._adjointDensePairMap[key] = new AdjointMultiplyDensePair;
                AdjointMultiplyDensePair& pair = *C._adjointDensePairMap[key];
                pair.first = &B;

                B.AdjointMultiplyDenseInitialize( pair.second, FGA.rank );
            }
            else
            {
                C._transposeDensePairMap[key] = new TransposeMultiplyDensePair;
                TransposeMultiplyDensePair& pair = 
                    *C._transposeDensePairMap[key];
                pair.first = &B;

                B.TransposeMultiplyDenseInitialize( pair.second, FGA.rank );
            }
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
                C._ZMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

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
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._ZMap[key] = new Dense<Scalar>( A.Height(), FB.Rank() );
            Dense<Scalar>& ZC = *C._ZMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), FB.Rank(), A.Width(),
              alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                         FB.U.LockedBuffer(),  FB.U.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case DENSE:
        {
            const Dense<Scalar>& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                C._DMap[key] = new Dense<Scalar>( A.Height(), B.Width() );
                hmat_tools::Multiply
                ( alpha, SDA.D, DB, (Scalar)0, *C._DMap[key] );
            }
            else
            {
                // D += D D
                SplitDense& SDC = *C._block.data.SD;
                hmat_tools::Multiply
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
        const Dense<Scalar>& DA = *A._block.data.D;
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._ZMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), SFB.rank, A.Width(),
              alpha,     DA.LockedBuffer(),    DA.LDim(),
                         SFB.D.LockedBuffer(), SFB.D.LDim(),
              (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            break;
        }
        case LOW_RANK:
        {
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            if( admissibleC )
            {
                LowRank<Scalar,Conjugated>& FC = *C._block.data.F;
                LowRank<Scalar,Conjugated> update;
                hmat_tools::Multiply( alpha, DA, FB, update );
                hmat_tools::RoundedUpdate
                ( C.MaxRank(), (Scalar)1, update, (Scalar)1, FC );
            }
            else
            {
                Dense<Scalar>& DC = *C._block.data.D;
                hmat_tools::Multiply( alpha, DA, FB, (Scalar)1, DC );
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
            const Dense<Scalar>& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                C._DMap[key] = new Dense<Scalar>( A.Height(), B.Width() );
                hmat_tools::Multiply
                ( alpha, DA, DB, (Scalar)0, *C._DMap[key] );
            }
            else
            {
                // D += D D
                Dense<Scalar>& DC = *C._block.data.D;
                hmat_tools::Multiply( alpha, DA, DB, (Scalar)1, DC );
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
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcasts
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcasts");
#endif
    /*
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    A.MultiplyHMatMainBroadcastsCountA( sizes );
    B.MultiplyHMatMainBroadcastsCountB( sizes );
    C.MultiplyHMatMainBroadcastsCountC( sizes );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of teamunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatMainBroadcastsPackA( buffer, offsets );
    B.MultiplyHMatMainBroadcastsPackB( buffer, offsets );
    C.MultiplyHMatMainBroadcastsPackC( buffer, offsets );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( sizes[i] != 0 )
        {
            MPI_Comm team = _teams->Team( i+1 );
            mpi::Broadcast( &buffer[offsets[i]], sizes[i], 0, team );
        }
    }

    // Unpack the broadcasted buffers
    A.MultiplyHMatMainBroadcastsUnpackA( buffer, offsets );
    B.MultiplyHMatMainBroadcastsUnpackB( buffer, offsets );
    C.MultiplyHMatMainBroadcastsUnpackC( buffer, offsets );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountC
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountC");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    {
        // HERE
        break;
    }
    default:
        break;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsCountC( sizes );
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

