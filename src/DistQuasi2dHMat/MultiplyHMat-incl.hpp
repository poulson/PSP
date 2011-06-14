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
    A.MultiplyHMatMainPostcompute( alpha, B, C );

    A.MultiplyHMatFHHPrecompute( alpha, B, C );
    A.MultiplyHMatFHHSummations( alpha, B, C );
    A.MultiplyHMatFHHPassData( alpha, B, C );
    A.MultiplyHMatFHHBroadcasts( alpha, B, C );
    A.MultiplyHMatFHHPostcompute( alpha, B, C );
    /*
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
    
    MPI_Comm team = _teams->Team( A._level );
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
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const int paddedRank = C.MaxRank() + 4;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
        C._block.type = EMPTY;
        return;
    }
    if( C._block.type == EMPTY )
        A.MultiplyHMatSetUp( B, C );

    // Handle all H H cases here
    const bool admissibleC = C.Admissible();
    if( !admissibleC )
    {
        // Take care of the H += H H cases first
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
    else
    {
        // Handle precomputation of A's row space
        if( !A._beganRowSpaceComp )
        {
            switch( A._block.type )
            {
            case DIST_NODE:
            case SPLIT_NODE:
            case NODE:
                switch( B._block.type )
                {
                case DIST_NODE:
                case DIST_NODE_GHOST:
                case SPLIT_NODE:
                case SPLIT_NODE_GHOST:
                case NODE:
                case NODE_GHOST:
                {
                    if( A._inTargetTeam )
                    {
                        A._rowOmega.Resize( A.LocalHeight(), paddedRank );
                        ParallelGaussianRandomVectors( A._rowOmega );
                    }
                    if( A._inSourceTeam )
                    {
                        A._rowT.Resize( A.LocalWidth(), paddedRank );
                        hmat_tools::Scale( (Scalar)0, A._rowT );
                    }
                    A.AdjointMultiplyDenseInitialize
                    ( A._rowContext, paddedRank );
                    if( A._inSourceTeam && A._inTargetTeam )
                    {
                        A.AdjointMultiplyDensePrecompute
                        ( A._rowContext, (Scalar)1, A._rowOmega, A._rowT );
                    }
                    else if( A._inSourceTeam )
                    {
                        Dense<Scalar> dummy( 0, paddedRank );
                        A.AdjointMultiplyDensePrecompute
                        ( A._rowContext, (Scalar)1, dummy, A._rowT );
                    }
                    else // A._inTargetTeam
                    {
                        Dense<Scalar> dummy( 0, paddedRank );
                        A.AdjointMultiplyDensePrecompute
                        ( A._rowContext, (Scalar)1, A._rowOmega, dummy );
                    }
                    A._beganRowSpaceComp = true;
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
        // Handle precomputation of B's column space
        if( !B._beganColSpaceComp )
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
                case SPLIT_NODE:
                case NODE:
                    if( B._inSourceTeam )
                    {
                        B._colOmega.Resize( B.LocalWidth(), paddedRank ); 
                        ParallelGaussianRandomVectors( B._colOmega );
                    }
                    if( B._inTargetTeam )
                    {
                        B._colT.Resize( B.LocalHeight(), paddedRank );
                        hmat_tools::Scale( (Scalar)0, B._colT );
                    }
                    B.MultiplyDenseInitialize( B._colContext, paddedRank );
                    if( B._inSourceTeam && B._inTargetTeam )
                    {
                        B.MultiplyDensePrecompute
                        ( B._colContext, (Scalar)1, B._colOmega, B._colT );
                    }
                    else if( B._inSourceTeam )
                    {
                        Dense<Scalar> dummy( 0, paddedRank );
                        B.MultiplyDensePrecompute
                        ( B._colContext, (Scalar)1, B._colOmega, dummy );
                    }
                    else // B._inTargetTeam
                    {
                        Dense<Scalar> dummy( 0, paddedRank );
                        B.MultiplyDensePrecompute
                        ( B._colContext, (Scalar)1, dummy, B._colT );
                    }
                    B._beganColSpaceComp = true;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
        }
    }

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
            // Start H/F += H F
            const DistLowRank& DFB = *B._block.data.DF;
            C._UMap[key] = new Dense<Scalar>( C.LocalHeight(), DFB.rank );
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( context, DFB.rank );
            A.MultiplyDensePrecompute
            ( context, alpha, DFB.ULocal, *C._UMap[key] );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We must be in the left team
            const DistLowRankGhost& DFGB = *B._block.data.DFG;
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            A.MultiplyDenseInitialize( context, DFGB.rank );
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
            // Start H/F += H F
            const SplitLowRank& SFB = *B._block.data.SF;
            if( C._inTargetTeam )
            {
                // Our process owns the left and right sides
                C._mainContextMap[key] = new MultiplyDenseContext;
                MultiplyDenseContext& context = *C._mainContextMap[key];

                A.MultiplyDenseInitialize( context, SFB.rank );
            }
            else
            {
                // We are the middle process
                C._mainContextMap[key] = new MultiplyDenseContext;
                MultiplyDenseContext& context = *C._mainContextMap[key];

                Dense<Scalar> dummy( 0, SFB.rank );
                A.MultiplyDenseInitialize( context, SFB.rank );
                A.MultiplyDensePrecompute( context, alpha, SFB.D, dummy );
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            A.MultiplyDenseInitialize( context, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We are the middle and right processes
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            Dense<Scalar> dummy( 0, FB.Rank() );
            A.MultiplyDenseInitialize( context, FB.Rank() );
            A.MultiplyDensePrecompute( context, alpha, FB.U, dummy );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            A.MultiplyDenseInitialize( context, FGB.rank );
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
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( context, SFB.rank );
            A.MultiplyDensePrecompute( context, alpha, SFB.D, *C._UMap[key] );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._UMap[key] = new Dense<Scalar>( C.Height(), FB.Rank() );
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];

            hmat_tools::Scale( (Scalar)0, *C._UMap[key] );
            A.MultiplyDenseInitialize( context, FB.Rank() );
            A.MultiplyDensePrecompute( context, alpha, FB.U, *C._UMap[key] );
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
    case SPLIT_NODE_GHOST:
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
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
            {
                B.AdjointMultiplyDenseInitialize( context, DFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( context, Conj(alpha), DFA.VLocal, *C._VMap[key] );
            }
            else
            {
                B.TransposeMultiplyDenseInitialize( context, DFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( context, alpha, DFA.VLocal, *C._VMap[key] );
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
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
                B.AdjointMultiplyDenseInitialize( context, DFGA.rank );
            else
                B.TransposeMultiplyDenseInitialize( context, DFGA.rank );
            break;
        }
        case DIST_LOW_RANK:
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
                Dense<Scalar> dummy( 0, SFA.rank );
                C._mainContextMap[key] = new MultiplyDenseContext;
                MultiplyDenseContext& context = *C._mainContextMap[key];
                if( Conjugated )
                {
                    B.AdjointMultiplyDenseInitialize( context, SFA.rank );
                    B.AdjointMultiplyDensePrecompute
                    ( context, Conj(alpha), SFA.D, dummy );
                }
                else
                {
                    B.TransposeMultiplyDenseInitialize( context, SFA.rank );
                    B.TransposeMultiplyDensePrecompute
                    ( context, alpha, SFA.D, dummy );
                }
            }
            else
            {
                C._mainContextMap[key] = new MultiplyDenseContext;
                MultiplyDenseContext& context = *C._mainContextMap[key];
                if( Conjugated )
                    B.AdjointMultiplyDenseInitialize( context, SFA.rank );
                else
                    B.TransposeMultiplyDenseInitialize( context, SFA.rank );
            }
            break;
        }
        case NODE:
        {
            // We are the middle and right process
            C._VMap[key] = new Dense<Scalar>( B.Width(), SFA.rank );
            Dense<Scalar>& CV = *C._VMap[key];
                
            hmat_tools::Scale( (Scalar)0, CV );
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
            {
                B.AdjointMultiplyDenseInitialize( context, SFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( context, Conj(alpha), SFA.D, CV );
            }
            else
            {
                B.TransposeMultiplyDenseInitialize( context, SFA.rank );
                B.TransposeMultiplyDensePrecompute( context, alpha, SFA.D, CV );
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
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
                B.AdjointMultiplyDenseInitialize( context, SFGA.rank );
            else
                B.TransposeMultiplyDenseInitialize( context, SFGA.rank );
            break;
        }
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:
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
            Dense<Scalar> dummy( 0, FA.Rank() );
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
            {
                B.AdjointMultiplyDenseInitialize( context, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( context, Conj(alpha), FA.V, dummy );
            }
            else
            {
                B.TransposeMultiplyDenseInitialize( context, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( context, alpha, FA.V, dummy );
            }
            break;
        }
        case NODE:
        {
            // We must own all of A, B, and C
            C._VMap[key] = new Dense<Scalar>( B.Width(), FA.Rank() );
                
            hmat_tools::Scale( (Scalar)0, *C._VMap[key] );
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
            {
                B.AdjointMultiplyDenseInitialize( context, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( context, Conj(alpha), FA.V, *C._VMap[key] );
            }
            else
            {
                B.TransposeMultiplyDenseInitialize( context, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( context, alpha, FA.V, *C._VMap[key] );
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
            // We must be the left and middle process, but there is no
            // work to be done (split dense owned by right process)
            break;
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
            C._mainContextMap[key] = new MultiplyDenseContext;
            MultiplyDenseContext& context = *C._mainContextMap[key];
            if( Conjugated )
                B.AdjointMultiplyDenseInitialize( context, FGA.rank );
            else
                B.TransposeMultiplyDenseInitialize( context, FGA.rank );
            break;
        }
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:
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
            C._UMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
            Dense<Scalar>& UC = *C._UMap[key];

            blas::Gemm
            ( 'N', 'N', A.Height(), SFB.rank, A.Width(),
              alpha,     DA.LockedBuffer(),    DA.LDim(),
                         SFB.D.LockedBuffer(), SFB.D.LDim(),
              (Scalar)0, UC.Buffer(),          UC.LDim() );
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
            break;
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
    case EMPTY:
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
    A.MultiplyHMatMainSummationsCountC( B, C, sizes );

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
    A.MultiplyHMatMainSummationsPackC( B, C, buffer, offsets );

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

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatMainSummationsUnpackA( buffer, offsets );
    B.MultiplyHMatMainSummationsUnpackB( buffer, offsets );
    A.MultiplyHMatMainSummationsUnpackC( B, C, buffer, offsets );
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
            TransposeMultiplyDenseSummationsCount( sizes, _rowContext.numRhs );

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
            TransposeMultiplyDenseSummationsPack( _rowContext, buffer, offsets );

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
            ( _rowContext, buffer, offsets );

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
            MultiplyDenseSummationsCount( sizes, _colContext.numRhs );

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
            MultiplyDenseSummationsPack( _colContext, buffer, offsets );

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
            MultiplyDenseSummationsUnpack( _colContext, buffer, offsets );

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
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsCountC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                const Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSummationsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
            }
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            A.MultiplyDenseSummationsCount( sizes, DFB.rank );
            break;
        }
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            B.TransposeMultiplyDenseSummationsCount( sizes, DFA.rank );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inSourceTeam )
                sizes[A._level-1] += DFA.rank*DFB.rank;
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsPackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B, 
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsPackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                const Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSummationsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
        case DIST_LOW_RANK:
            A.MultiplyDenseSummationsPack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            B.TransposeMultiplyDenseSummationsPack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inSourceTeam )
            {
                std::memcpy
                ( &buffer[offsets[A._level-1]], C._ZMap[key]->LockedBuffer(),
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[A._level-1] += DFA.rank*DFB.rank;
            }
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSummationsUnpackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSummationsUnpackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSummationsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
        case DIST_LOW_RANK:
            A.MultiplyDenseSummationsUnpack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            B.TransposeMultiplyDenseSummationsUnpack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inSourceTeam )
            {
                std::memcpy
                ( C._ZMap[key]->Buffer(), &buffer[offsets[A._level-1]],
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[A._level-1] += DFA.rank*DFB.rank;
            }
            break;
        }
        default:
            break;
        }
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
    int totalSendSize=0, totalRecvSize=0;
    for( int i=0; i<p; ++i )
    {
        totalSendSize += sendSizes[i];
        totalRecvSize += recvSizes[i];
    }
    std::vector<Scalar> sendBuffer(totalSendSize), recvBuffer(totalRecvSize);
    std::vector<int> sendOffsets( p ), recvOffsets( p );
    for( int i=0,offset=0; i<p; offset+=sendSizes[i],++i )
        sendOffsets[i] = offset;
    for( int i=0,offset=0; i<p; offset+=recvSizes[i],++i )
        recvOffsets[i] = offset;

    // 3) Pack sends
    std::vector<int> offsets = sendOffsets;
    A.MultiplyHMatMainPassDataPackA( sendBuffer, offsets );
    B.MultiplyHMatMainPassDataPackB( sendBuffer, offsets );
    A.MultiplyHMatMainPassDataPackC( B, C, sendBuffer, offsets );

    // 4) MPI_Alltoallv
    mpi::AllToAllV
    ( &sendBuffer[0], &sendSizes[0], &sendOffsets[0],
      &recvBuffer[0], &recvSizes[0], &recvOffsets[0], comm );

    // 5) Unpack recvs
    offsets = recvOffsets;
    A.MultiplyHMatMainPassDataUnpackA( recvBuffer, offsets );
    B.MultiplyHMatMainPassDataUnpackB( recvBuffer, offsets );
    A.MultiplyHMatMainPassDataUnpackC( B, C, recvBuffer, offsets );
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
            ( sendSizes, recvSizes, _rowContext.numRhs );

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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataPackA
( std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _beganRowSpaceComp )
        {
            if( _inTargetTeam )
            {
                TransposeMultiplyDensePassDataPack
                ( _rowContext, _rowOmega, sendBuffer, offsets );
            }
            else
            {
                Dense<Scalar> dummy( 0, _rowContext.numRhs );
                TransposeMultiplyDensePassDataPack
                ( _rowContext, dummy, sendBuffer, offsets );
            }
        }

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainPassDataPackA
                ( sendBuffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataUnpackA
( const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDensePassDataUnpack
            ( _rowContext, recvBuffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainPassDataUnpackA
                ( recvBuffer, offsets );
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
            ( sendSizes, recvSizes, _colContext.numRhs );

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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataPackB
( std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _beganColSpaceComp )
            MultiplyDensePassDataPack( _colContext, sendBuffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainPassDataPackB
                ( sendBuffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataUnpackB
( const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _beganColSpaceComp )
            MultiplyDensePassDataUnpack( _colContext, recvBuffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainPassDataUnpackB
                ( recvBuffer, offsets );
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
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
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
                const Node& nodeC = *C._block.data.N;
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

    MPI_Comm team = _teams->Team( _level );
    const int teamRank = mpi::CommRank( team );
    switch( A._block.type )
    {
    case DIST_NODE:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case DIST_LOW_RANK:
        {
            // Pass data count for H/F += H F
            const DistLowRank& DFB = *B._block.data.DF;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, DFB.rank );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Pass data count for H/F += H F. This should only contribute
            // to the recv sizes.
            const DistLowRankGhost& DFGB = *B._block.data.DFG;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, DFGB.rank );
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
    case SPLIT_NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/F += H F
            const SplitLowRank& SFB = *B._block.data.SF;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, SFB.rank );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass datal for H/F += H F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Pass data for H/F += H F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, FB.Rank() );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for H/F += H F
            const LowRankGhost& FGB = *B._block.data.FG;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, FGB.rank );
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
    case NODE:
        // The only possiblities are recursion, F += H H, and H/F += H F; the
        // first two are not handled here, and the last does not require any
        // work here because the precompute step handled everything.
        break;
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
        // The only non-recursive possibilities are H/F += H F and F += H H;
        // the former does not require our participation here and the latter
        // is handled by CountA and CountB.
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A._block.data.DF;
        switch( B._block.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, DFA.rank );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( teamRank == 0 && (A._inSourceTeam != A._inTargetTeam) )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                if( A._inSourceTeam )
                    sendSizes[A._targetRoot] += DFA.rank*DFB.rank;
                if( A._inTargetTeam )
                    recvSizes[A._sourceRoot] += DFA.rank*DFB.rank;
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F
            if( teamRank == 0 )
            {
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                recvSizes[A._sourceRoot] += DFA.rank*DFGB.rank;
            }
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
        const DistLowRankGhost& DFGA = *A._block.data.DFG;
        switch( B._block.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, DFGA.rank );
            break;
        case DIST_LOW_RANK:
            // Pass data for for H/F += F F is between other (two) team(s)
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
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, SFA.rank );
            break;
        case SPLIT_NODE_GHOST:
            // Pass data for H/F += F H is between other two processes
            break;
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/D/F += F F
            // We're either the middle process or both the left and right
            const SplitLowRank& SFB = *B._block.data.SF;
            if( A._inSourceTeam )
                sendSizes[A._targetRoot] += SFA.rank*SFB.rank;
            else
                recvSizes[A._sourceRoot] += SFA.rank*SFB.rank;
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            recvSizes[A._sourceRoot] += SFA.rank*SFGB.rank;
            break;
        }
        case LOW_RANK:
        {
            // Pass data for H/D/F += F F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            sendSizes[A._targetRoot] += SFA.rank*FB.Rank();
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const LowRankGhost& FGB = *B._block.data.FG;
            recvSizes[A._sourceRoot] += SFA.rank*FGB.rank;
            break;
        }
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B._inTargetTeam )
                sendSizes[B._sourceRoot] += B.Height()*SFA.rank;
            else
                recvSizes[B._targetRoot] += B.Height()*SFA.rank;
            break;
        }
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // Pass data for D/F += F D is in other process
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
        const SplitLowRankGhost& SFGA = *A._block.data.SFG; 
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, SFGA.rank );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is between other two processes
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            recvSizes[B._targetRoot] += B.Height()*SFGA.rank;
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
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, FA.Rank() );
            break;
        case NODE:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
            // There is no pass data
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            sendSizes[B._sourceRoot] += B.Height()*FA.Rank();
            break;
        case DENSE:
            // There is no pass data
            break;
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
        const LowRankGhost& FGA = *A._block.data.FG;
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, FGA.rank );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is in other process
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            recvSizes[B._targetRoot] += B.Height()*FGA.rank;
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
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            // Pass data for D/F += D F
            const SplitLowRank& SFB = *B._block.data.SF;
            if( A._inSourceTeam )
                sendSizes[A._targetRoot] += A.Height()*SFB.rank;
            else
                recvSizes[A._sourceRoot] += A.Height()*SFB.rank;
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            recvSizes[A._sourceRoot] += A.Height()*SFGB.rank;
            break;
        }
        case LOW_RANK:
        {
            // Pass data for D/F += D F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            sendSizes[A._targetRoot] += A.Height()*FB.Rank();
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const LowRankGhost& FGB = *B._block.data.FG;
            recvSizes[A._sourceRoot] += A.Height()*FGB.rank;
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B._inSourceTeam )
                recvSizes[B._targetRoot] += A.Height()*A.Width();
            else
                sendSizes[B._sourceRoot] += A.Height()*A.Width();
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
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
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
            recvSizes[B._targetRoot] += A.Height()*A.Width();
            break;
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
            break;
        case SPLIT_DENSE:
            sendSizes[B._sourceRoot] += A.Height()*A.Width();
            break;
        case DENSE:
            break;
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case DENSE_GHOST:
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
            recvSizes[B._targetRoot] += A.Height()*A.Width();
            break;
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataPackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataPackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Take care of the H += H H cases first
    const int key = A._sourceOffset;
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
                            nodeA.Child(t,r).MultiplyHMatMainPassDataPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendBuffer, offsets );
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

    MPI_Comm team = _teams->Team( _level );
    const int teamRank = mpi::CommRank( team );
    switch( A._block.type )
    {
    case DIST_NODE:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case DIST_LOW_RANK:
        case DIST_LOW_RANK_GHOST:
            // Pass data pack for H/F += H F
            A.MultiplyDensePassDataPack
            ( *C._mainContextMap[key], sendBuffer, offsets );
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
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
            // Pass data pack for H/F += H F
            A.MultiplyDensePassDataPack
            ( *C._mainContextMap[key], sendBuffer, offsets );
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
        // The only possiblities are recursion, F += H H, and H/F += H F; the
        // first two are not handled here, and the last does not require any
        // work here because the precompute step handled everything.
        break;
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
        // The only non-recursive possibilities are H/F += H F and F += H H;
        // the former does not require our participation here and the latter
        // is handled by CountA and CountB.
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A._block.data.DF;
        switch( B._block.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            if( A._inSourceTeam )
                B.TransposeMultiplyDensePassDataPack
                ( *C._mainContextMap[key], DFA.VLocal, sendBuffer, offsets );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( teamRank == 0 && (A._inSourceTeam != A._inTargetTeam) )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                if( A._inSourceTeam )
                {
                    Dense<Scalar>& ZC = *C._ZMap[key];
                    std::memcpy
                    ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[A._targetRoot] += DFA.rank*DFB.rank;
                    ZC.Clear();
                }
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F is only receiving here
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
        // We, at most, receive here
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A._block.data.SF;
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            if( A._inSourceTeam )
                B.TransposeMultiplyDensePassDataPack
                ( *C._mainContextMap[key], SFA.D, sendBuffer, offsets );
            break;
        case SPLIT_NODE_GHOST:
            // Pass data for H/F += F H is between other two processes
            break;
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/D/F += F F
            // We're either the middle process or both the left and right
            const SplitLowRank& SFB = *B._block.data.SF;
            if( A._inSourceTeam )
            {
                Dense<Scalar>& ZC = *C._ZMap[key];
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                  SFA.rank*SFB.rank*sizeof(Scalar) );
                offsets[A._targetRoot] += SFA.rank*SFB.rank;
                ZC.Clear();
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
            // Pass data for H/D/F += F F is just a receive for us
            break;
        case LOW_RANK:
        {
            // Pass data for H/D/F += F F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            Dense<Scalar>& ZC = *C._ZMap[key];
            std::memcpy
            ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
              SFA.rank*FB.Rank()*sizeof(Scalar) );
            offsets[A._targetRoot] += SFA.rank*FB.Rank();
            ZC.Clear();
            break;
        }
        case LOW_RANK_GHOST:
            // Pass data for H/D/F += F F is just a receive for us
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B._inTargetTeam )
            {
                std::memcpy
                ( &sendBuffer[offsets[B._sourceRoot]], SFA.D.LockedBuffer(),
                  B.Height()*SFA.rank*sizeof(Scalar) );
                offsets[B._sourceRoot] += B.Height()*SFA.rank;
            }
            break;
        }
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // Pass data for D/F += F D is in other process
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
        break;
    case LOW_RANK:
    {
        const LowRank<Scalar,Conjugated>& FA = *A._block.data.F;
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataPack
            ( *C._mainContextMap[key], FA.V, sendBuffer, offsets );
            break;
        case NODE:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
            // There is no pass data
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            std::memcpy
            ( &sendBuffer[offsets[B._sourceRoot]], FA.V.LockedBuffer(),
              B.Height()*FA.Rank()*sizeof(Scalar) );
            offsets[B._sourceRoot] += B.Height()*FA.Rank();
            break;
        }
        case DENSE:
            // There is no pass data
            break;
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
    case LOW_RANK_GHOST:
        break;
    case SPLIT_DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            // Pass data for D/F += D F
            const SplitLowRank& SFB = *B._block.data.SF;
            if( A._inSourceTeam )
            {
                Dense<Scalar>& ZC = *C._ZMap[key];
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                  A.Height()*SFB.rank*sizeof(Scalar) );
                offsets[A._targetRoot] += A.Height()*SFB.rank;
                ZC.Clear();
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
            break;
        case LOW_RANK:
        {
            // Pass data for D/F += D F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            Dense<Scalar>& ZC = *C._ZMap[key];
            std::memcpy
            ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(), 
              A.Height()*FB.Rank()*sizeof(Scalar) );
            offsets[A._targetRoot] += A.Height()*FB.Rank();
            ZC.Clear();
            break;
        }
        case LOW_RANK_GHOST:
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B._inTargetTeam )
            {
                const SplitDense& SDA = *A._block.data.SD;
                std::memcpy
                ( &sendBuffer[offsets[B._sourceRoot]], SDA.D.LockedBuffer(),
                  A.Height()*A.Width()*sizeof(Scalar) );
                offsets[B._sourceRoot] += A.Height()*A.Width();
            }
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
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
        break;
    case DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case LOW_RANK:
            break;
        case SPLIT_DENSE:
        {
            const Dense<Scalar>& DA = *A._block.data.D;
            std::memcpy
            ( &sendBuffer[offsets[B._sourceRoot]], DA.LockedBuffer(),
              A.Height()*A.Width()*sizeof(Scalar) );
            offsets[B._sourceRoot] += A.Height()*A.Width();
            break;
        }
        case DENSE:
            break;
        default:
#ifndef RELEASE
            throw std::logic_error("Invalid H-matrix combination");
#endif
            break;
        }
        break;
    }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataUnpackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataUnpackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Take care of the H += H H cases first
    const int key = A._sourceOffset;
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
                            nodeA.Child(t,r).MultiplyHMatMainPassDataUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              recvBuffer, offsets );
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

    MPI_Comm team = _teams->Team( _level );
    const int teamRank = mpi::CommRank( team );
    switch( A._block.type )
    {
    case DIST_NODE:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case DIST_LOW_RANK:
        case DIST_LOW_RANK_GHOST:
        {
            // Pass data unpack for H/F += H F
            A.MultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
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
    case SPLIT_NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the UnpackA/UnpackB subroutines.
            break;
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
        {
            // Pass data for H/F += H F
            A.MultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
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
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A._block.data.DF;
        switch( B._block.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( teamRank == 0 && A._inTargetTeam && !A._inSourceTeam )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                C._ZMap[key] = new Dense<Scalar>( DFA.rank, DFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += DFA.rank*DFB.rank;
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F
            if( teamRank == 0 )
            {
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                C._ZMap[key] = new Dense<Scalar>( DFA.rank, DFGB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  DFA.rank*DFGB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += DFA.rank*DFGB.rank;
            }
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
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
            break;
        case DIST_LOW_RANK:
            // Pass data for for H/F += F F is between other (two) team(s)
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
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
            break;
        case SPLIT_NODE_GHOST:
            // Pass data for H/F += F H is between other two processes
            break;
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/D/F += F F
            // We're either the middle process or both the left and right
            const SplitLowRank& SFB = *B._block.data.SF;
            if( A._inTargetTeam )
            {
                C._ZMap[key] = new Dense<Scalar>( SFA.rank, SFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  SFA.rank*SFB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += SFA.rank*SFB.rank;
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._ZMap[key] = new Dense<Scalar>( SFA.rank, SFGB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            std::memcpy
            ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
              SFA.rank*SFGB.rank*sizeof(Scalar) );
            offsets[A._sourceRoot] += SFA.rank*SFGB.rank;
            break;
        }
        case LOW_RANK:
            // Pass data for H/D/F += F F is a send
            break;
        case LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const LowRankGhost& FGB = *B._block.data.FG;
            C._ZMap[key] = new Dense<Scalar>( SFA.rank, FGB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            std::memcpy
            ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
              SFA.rank*FGB.rank*sizeof(Scalar) );
            offsets[A._sourceRoot] += SFA.rank*FGB.rank;
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            if( B._inSourceTeam )
            {
                C._ZMap[key] = new Dense<Scalar>( B.Height(), SFA.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  B.Height()*SFA.rank*sizeof(Scalar) );
                offsets[B._targetRoot] += B.Height()*SFA.rank;
            }
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // Pass data for D/F += F D is in other process
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
        const SplitLowRankGhost& SFGA = *A._block.data.SFG; 
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is between other two processes
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            if( B._inSourceTeam )
            {
                C._ZMap[key] = new Dense<Scalar>( B.Height(), SFGA.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  B.Height()*SFGA.rank*sizeof(Scalar) );
                offsets[B._targetRoot] += B.Height()*SFGA.rank;
            }
            break;
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
        const LowRankGhost& FGA = *A._block.data.FG;
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( *C._mainContextMap[key], recvBuffer, offsets );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is in other process
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            C._ZMap[key] = new Dense<Scalar>( B.Height(), FGA.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            std::memcpy
            ( ZC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
              B.Height()*FGA.rank*sizeof(Scalar) );
            offsets[B._targetRoot] += B.Height()*FGA.rank;
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
    case SPLIT_DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            // Pass data for D/F += D F
            if( A._inTargetTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._ZMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  A.Height()*SFB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += A.Height()*SFB.rank;
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._ZMap[key] = new Dense<Scalar>( A.Height(), SFGB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            std::memcpy
            ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
              A.Height()*SFGB.rank*sizeof(Scalar) );
            offsets[A._sourceRoot] += A.Height()*SFGB.rank;
            break;
        }
        case LOW_RANK:
            break;
        case LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const LowRankGhost& FGB = *B._block.data.FG;
            C._ZMap[key] = new Dense<Scalar>( A.Height(), FGB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];

            std::memcpy
            ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
              A.Height()*FGB.rank*sizeof(Scalar) );
            offsets[A._sourceRoot] += A.Height()*FGB.rank;
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B._inSourceTeam )
            {
                C._DMap[key] = new Dense<Scalar>( A.Height(), A.Width() );
                Dense<Scalar>& DC = *C._DMap[key];

                std::memcpy
                ( DC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  A.Height()*A.Width()*sizeof(Scalar) );
                offsets[B._targetRoot] += A.Height()*A.Width();
            }
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
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
    case DENSE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
        {
            C._DMap[key] = new Dense<Scalar>( A.Height(), A.Width() );
            Dense<Scalar>& DC = *C._DMap[key];

            std::memcpy
            ( DC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
              A.Height()*A.Width()*sizeof(Scalar) );
            offsets[B._targetRoot] += A.Height()*A.Width();
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
    case NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case LOW_RANK:
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcasts
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcasts");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    A.MultiplyHMatMainBroadcastsCountA( sizes );
    B.MultiplyHMatMainBroadcastsCountB( sizes );
    A.MultiplyHMatMainBroadcastsCountC( B, C, sizes );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatMainBroadcastsPackA( buffer, offsets );
    B.MultiplyHMatMainBroadcastsPackB( buffer, offsets );
    A.MultiplyHMatMainBroadcastsPackC( B, C, buffer, offsets );

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
    A.MultiplyHMatMainBroadcastsUnpackC( B, C, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountA
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseBroadcastsCount( sizes, _rowContext.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsCountA( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseBroadcastsPack( _rowContext, buffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsPackA
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseBroadcastsUnpack
            ( _rowContext, buffer, offsets );

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsUnpackA
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountB
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseBroadcastsCount( sizes, _colContext.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsCountB( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseBroadcastsPack( _colContext, buffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsPackB
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseBroadcastsUnpack( _colContext, buffer, offsets );

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainBroadcastsUnpackB
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
  const DistQuasi2dHMat<Scalar,Conjugated>& C, 
  std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                const Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
            }
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            A.MultiplyDenseBroadcastsCount( sizes, DFB.rank );
            break;
        }
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            B.TransposeMultiplyDenseBroadcastsCount( sizes, DFA.rank );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inTargetTeam )
                sizes[A._level-1] += DFA.rank*DFB.rank;
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsPackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B, 
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsPackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                const Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
        case DIST_LOW_RANK:
            A.MultiplyDenseBroadcastsPack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            B.TransposeMultiplyDenseBroadcastsPack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( A._inTargetTeam && teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[A._level-1]], C._ZMap[key]->LockedBuffer(),
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[A._level-1] += DFA.rank*DFB.rank;
            }
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsUnpackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsUnpackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
        case DIST_LOW_RANK:
            A.MultiplyDenseBroadcastsUnpack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            B.TransposeMultiplyDenseBroadcastsUnpack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inTargetTeam )
            {
                std::memcpy
                ( C._ZMap[key]->Buffer(), &buffer[offsets[A._level-1]],
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[A._level-1] += DFA.rank*DFB.rank;
            }
            break;
        }
        default:
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcompute
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcompute");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A.MultiplyHMatMainPostcomputeA();
    B.MultiplyHMatMainPostcomputeB();
    A.MultiplyHMatMainPostcomputeC( alpha, B, C );
    C.MultiplyHMatMainPostcomputeCCleanup();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeA() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeA");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Handle postcomputation of A's row space
    if( A._beganRowSpaceComp )
    {
        const int numRhs = A._rowContext.numRhs;
        if( A._inSourceTeam && A._inTargetTeam )
        {
            A.AdjointMultiplyDensePostcompute
            ( A._rowContext, (Scalar)1, A._rowOmega, A._rowT );
        }
        else if( A._inSourceTeam )
        {
            Dense<Scalar> dummy( 0, numRhs );
            A.AdjointMultiplyDensePostcompute
            ( A._rowContext, (Scalar)1, dummy, A._rowT );
        }
        else // A._inTargetTeam
        {
            Dense<Scalar> dummy( 0, numRhs );
            A.AdjointMultiplyDensePostcompute
            ( A._rowContext, (Scalar)1, A._rowOmega, dummy );
        }
        A._rowContext.Clear();
    }

    switch( A._block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& nodeA = *A._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).MultiplyHMatMainPostcomputeA();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeB() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeB");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& B = *this;

    // Handle postcomputation of B's column space
    if( B._beganColSpaceComp )
    {
        const int numRhs = B._colContext.numRhs;
        if( B._inSourceTeam && B._inTargetTeam )
        {
            B.MultiplyDensePostcompute
            ( B._colContext, (Scalar)1, B._colOmega, B._colT );
        }
        else if( B._inSourceTeam )
        {
            Dense<Scalar> dummy( 0, numRhs );
            B.MultiplyDensePostcompute
            ( B._colContext, (Scalar)1, B._colOmega, dummy );
        }
        else // B._inTargetTeam
        {
            Dense<Scalar> dummy( 0, numRhs );
            B.MultiplyDensePostcompute
            ( B._colContext, (Scalar)1, dummy, B._colT );
        }
        B._colContext.Clear();
    }

    switch( B._block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& nodeB = *B._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeB.Child(t,s).MultiplyHMatMainPostcomputeB();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeC
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    
    // Handle all H H recursion here
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
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainPostcomputeC
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

    // Handle the non-recursive part of the postcompute
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
            const DistLowRank& DFB = *B._block.data.DF;
            A.MultiplyDensePostcompute
            ( *C._mainContextMap[key], alpha, DFB.ULocal, *C._UMap[key] );
            C._mainContextMap[key]->Clear();
            if( C._inSourceTeam )
            {
                C._VMap[key] = new Dense<Scalar>;
                hmat_tools::Copy( DFB.VLocal, *C._VMap[key] );
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            const DistLowRankGhost& DFGB = *B._block.data.DFG; 
            Dense<Scalar> dummy( 0, DFGB.rank );
            A.MultiplyDensePostcompute
            ( *C._mainContextMap[key], alpha, dummy, *C._UMap[key] );
            C._mainContextMap[key]->Clear();
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
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            C._VMap[key] = new Dense<Scalar>( B.LocalWidth(), DFB.rank );
            hmat_tools::Copy( DFB.VLocal, *C._VMap[key] );
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
            if( C._inTargetTeam )
            {
                C._UMap[key] = new Dense<Scalar>( C.Height(), SFB.rank );
                C._VMap[key] = new Dense<Scalar>( C.Width(), SFB.rank );
                Dense<Scalar> dummy( 0, SFB.rank );
                A.MultiplyDensePostcompute
                ( *C._mainContextMap[key], alpha, dummy, *C._UMap[key] );
                hmat_tools::Copy( SFB.D, *C._VMap[key] );
            }
            C._mainContextMap[key]->Clear();
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._UMap[key] = new Dense<Scalar>( C.Height(), SFGB.rank );
            Dense<Scalar> dummy( 0, SFGB.rank );
            A.MultiplyDensePostcompute
            ( *C._mainContextMap[key], alpha, dummy, *C._UMap[key] );
            C._mainContextMap[key]->Clear();
            break;
        }
        case LOW_RANK:
        {
            // We are the middle and right process
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._VMap[key] = new Dense<Scalar>( C.Width(), FB.Rank() );
            hmat_tools::Copy( FB.V, *C._VMap[key] );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We are the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._UMap[key] = new Dense<Scalar>( C.Height(), FGB.rank );
            Dense<Scalar> dummy( 0, FGB.rank );
            A.MultiplyDensePostcompute
            ( *C._mainContextMap[key], alpha, dummy, *C._UMap[key] );
            C._mainContextMap[key]->Clear();
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
    case NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case NODE:
            break;
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFB.D, *C._VMap[key] );
            break;
        }
        case LOW_RANK:
        {
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( FB.V, *C._VMap[key] );
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
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
            break;
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF; 
            C._VMap[key] = new Dense<Scalar>( B.LocalWidth(), SFB.rank );
            hmat_tools::Copy( SFB.D, *C._VMap[key] );
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
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A._block.data.DF;
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            MultiplyDenseContext& context = *C._mainContextMap[key];

            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( DFA.ULocal, *C._UMap[key] );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), DFA.VLocal, *C._VMap[key] );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, DFA.VLocal, *C._VMap[key] );
            context.Clear();
            break;
        }
        case DIST_NODE_GHOST:
        {
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( DFA.ULocal, *C._UMap[key] );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            C._UMap[key] = new Dense<Scalar>( A.LocalHeight(), DFB.rank );
            C._VMap[key] = new Dense<Scalar>;
            Dense<Scalar>& ZC = *C._ZMap[key];
            Dense<Scalar>& UC = *C._UMap[key];
            blas::Gemm
            ( 'N', 'N', A.LocalHeight(), DFB.rank, DFA.rank,
              alpha,     DFA.ULocal.LockedBuffer(), DFA.ULocal.LDim(),
                         ZC.LockedBuffer(),         ZC.LDim(),
              (Scalar)0, UC.Buffer(),               UC.LDim() );
            ZC.Clear();
            hmat_tools::Copy( DFB.VLocal, *C._VMap[key] );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            const DistLowRankGhost& DFGB = *B._block.data.DFG; 
            C._UMap[key] = new Dense<Scalar>( A.LocalHeight(), DFGB.rank );
            Dense<Scalar>& ZC = *C._ZMap[key];
            Dense<Scalar>& UC = *C._UMap[key];
            blas::Gemm
            ( 'N', 'N', A.LocalHeight(), DFGB.rank, DFA.rank,
              alpha,     DFA.ULocal.LockedBuffer(), DFA.ULocal.LDim(),
                         ZC.LockedBuffer(),         ZC.LDim(),
              (Scalar)0, UC.Buffer(),               UC.LDim() );
            ZC.Clear();
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
    case DIST_LOW_RANK_GHOST:
    {
        const DistLowRankGhost& DFGA = *A._block.data.DFG;
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            MultiplyDenseContext& context = *C._mainContextMap[key];
            Dense<Scalar> dummy( 0, DFGA.rank );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), dummy, *C._VMap[key] );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, *C._VMap[key] );
            context.Clear();
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( DFB.VLocal, *C._VMap[key] );
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
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A._block.data.SF;
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // We are either the middle process or both the left and the 
            // right. The middle process doesn't have any work left.
            if( A._inTargetTeam )
            {
                MultiplyDenseContext& context = *C._mainContextMap[key];
                C._UMap[key] = new Dense<Scalar>;
                hmat_tools::Copy( SFA.D, *C._UMap[key] );
                Dense<Scalar> dummy( 0, SFA.rank );
                if( Conjugated )
                    B.AdjointMultiplyDensePostcompute
                    ( context, Conj(alpha), dummy, *C._VMap[key] );
                else
                    B.TransposeMultiplyDensePostcompute
                    ( context, alpha, dummy, *C._VMap[key] );
                context.Clear();
            }
            break;
        case SPLIT_NODE_GHOST:
            // We are the left process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFA.D, *C._UMap[key] );
            break;
        case NODE:
            // The precompute is not needed
            break;
        case NODE_GHOST:
            // We are the left process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFA.D, *C._UMap[key] );
            break;
        case SPLIT_LOW_RANK:
            // We are either the middle process or both the left and right.
            // The middle process is done.
            if( A._inTargetTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._UMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
                C._VMap[key] = new Dense<Scalar>;
                Dense<Scalar>& UC = *C._UMap[key];
                Dense<Scalar>& ZC = *C._ZMap[key];
                blas::Gemm
                ( 'N', 'N', A.Height(), SFB.rank, SFA.rank,
                  alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0, UC.Buffer(),          UC.LDim() );
                ZC.Clear();
                hmat_tools::Copy( SFB.D, *C._VMap[key] );
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._UMap[key] = new Dense<Scalar>( A.Height(), SFGB.rank );
            Dense<Scalar>& UC = *C._UMap[key];
            Dense<Scalar>& ZC = *C._ZMap[key];
            blas::Gemm
            ( 'N', 'N', A.Height(), SFGB.rank, SFA.rank,
              alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                         ZC.LockedBuffer(),    ZC.LDim(),
              (Scalar)0, UC.Buffer(),          UC.LDim() );
            ZC.Clear();
            break;
        }
        case LOW_RANK:
        {
            // We must be the middle and right process
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( FB.V, *C._VMap[key] );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We must be the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._UMap[key] = new Dense<Scalar>( A.Height(), FGB.rank );
            Dense<Scalar>& UC = *C._UMap[key];
            Dense<Scalar>& ZC = *C._ZMap[key];
            blas::Gemm
            ( 'N', 'N', A.Height(), FGB.rank, SFA.rank,
              alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                         ZC.LockedBuffer(),    ZC.LDim(),
              (Scalar)0, UC.Buffer(),          UC.LDim() );
            ZC.Clear();
            break;
        }
        case SPLIT_DENSE:
            // We are either the middle process or both the left and right
            if( A._inTargetTeam )
            {
                const SplitDense& SDB = *B._block.data.SD;
                C._UMap[key] = new Dense<Scalar>;
                C._VMap[key] = new Dense<Scalar>( C.Width(), SFA.rank );
                hmat_tools::Copy( SFA.D, *C._UMap[key] );
                Dense<Scalar>& VC = *C._VMap[key];
                Dense<Scalar>& ZC = *C._ZMap[key];
                if( Conjugated )
                    blas::Gemm
                    ( 'C', 'N', C.Width(), SFA.rank, B.Height(),
                      Conj(alpha), SDB.D.LockedBuffer(), SDB.D.LDim(),
                                   ZC.LockedBuffer(),    ZC.LDim(),
                      (Scalar)0,   VC.Buffer(),          VC.LDim() );
                else
                    blas::Gemm
                    ( 'T', 'N', C.Width(), SFA.rank, B.Height(),
                      alpha,     SDB.D.LockedBuffer(), SDB.D.LDim(),
                                 ZC.LockedBuffer(),    ZC.LDim(),
                      (Scalar)0, VC.Buffer(),          VC.LDim() );
                ZC.Clear();
            }
            break;
        case SPLIT_DENSE_GHOST:
            // We are the left process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFA.D, *C._UMap[key] );
            break;
        case DENSE:
            // We are the middle and right process, there is nothing left to do
            break;
        case DENSE_GHOST:
            // We are the left process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFA.D, *C._UMap[key] );
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
        // We are the right process
        const SplitLowRankGhost& SFGA = *A._block.data.SFG;
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            MultiplyDenseContext& context = *C._mainContextMap[key];
            Dense<Scalar> dummy( 0, SFGA.rank );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), dummy, *C._VMap[key] );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, *C._VMap[key] );
            context.Clear();
            break;
        }
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFB.D, *C._VMap[key] );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            C._VMap[key] = new Dense<Scalar>( C.Width(), SFGA.rank );
            Dense<Scalar>& VC = *C._VMap[key];
            Dense<Scalar>& ZC = *C._ZMap[key];
            if( Conjugated )
                blas::Gemm
                ( 'C', 'N', C.Width(), SFGA.rank, B.Height(),
                  Conj(alpha), SDB.D.LockedBuffer(), SDB.D.LDim(),
                               ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0,   VC.Buffer(),          VC.LDim() );
            else
                blas::Gemm
                ( 'T', 'N', C.Width(), SFGA.rank, B.Height(),
                  alpha,     SDB.D.LockedBuffer(), SDB.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0, VC.Buffer(),          VC.LDim() );
            ZC.Clear();
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
    case LOW_RANK:
    {
        const LowRank<Scalar,Conjugated>& FA = *A._block.data.F;
        switch( B._block.type )
        {
        case SPLIT_NODE:
            // We are the left and middle process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( FA.U, *C._UMap[key] );
            break;
        case NODE:
            // We own all of A, B, and C
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( FA.U, *C._UMap[key] );
            break;
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            C._UMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
            Dense<Scalar>& UC = *C._UMap[key];
            Dense<Scalar>& ZC = *C._ZMap[key];
            blas::Gemm
            ( 'N', 'N', A.Height(), SFB.rank, A.Width(),
              alpha,     FA.U.LockedBuffer(), FA.U.LDim(),
                         ZC.LockedBuffer(),   ZC.LDim(),
              (Scalar)0, UC.Buffer(),         UC.LDim() );
            break;
        }
        case LOW_RANK:
        {
            // We own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._UMap[key] = new Dense<Scalar>( A.Height(), FB.Rank() );
            C._VMap[key] = new Dense<Scalar>;
            Dense<Scalar>& UC = *C._UMap[key];
            Dense<Scalar>& ZC = *C._ZMap[key];
            blas::Gemm
            ( 'N', 'N', A.Height(), FB.Rank(), A.Width(),
              alpha,     FA.U.LockedBuffer(), FA.U.LDim(),
                         ZC.LockedBuffer(),   ZC.LDim(),
              (Scalar)0, UC.Buffer(),         UC.LDim() );
            hmat_tools::Copy( FB.V, *C._VMap[key] );
            break;
        }
        case SPLIT_DENSE:
        {
            // We are the left and middle process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( FA.U, *C._UMap[key] );
            break;
        }
        case DENSE:
        {
            // We own all of A, B, and C
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( FA.U, *C._UMap[key] );
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
        // We are the right process
        const LowRankGhost& FGA = *A._block.data.FG;
        switch( B._block.type )
        {
        case SPLIT_NODE:
        {
            MultiplyDenseContext& context = *C._mainContextMap[key];
            Dense<Scalar> dummy( 0, FGA.rank );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), dummy, *C._VMap[key] );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, *C._VMap[key] );
            context.Clear();
            break;
        }
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFB.D, *C._VMap[key] );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            C._VMap[key] = new Dense<Scalar>( C.Width(), FGA.rank );
            Dense<Scalar>& VC = *C._VMap[key];
            Dense<Scalar>& ZC = *C._ZMap[key];
            if( Conjugated )
                blas::Gemm
                ( 'C', 'N', C.Width(), FGA.rank, B.Height(),
                  Conj(alpha), SDB.D.LockedBuffer(), SDB.D.LDim(),
                               ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0,   VC.Buffer(),          VC.LDim() );
            else
                blas::Gemm
                ( 'T', 'N', C.Width(), FGA.rank, B.Height(),
                  alpha,     SDB.D.LockedBuffer(), SDB.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0, VC.Buffer(),          VC.LDim() );
            ZC.Clear();
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
    case SPLIT_DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            // We are either the middle process or the left and right
            if( A._inTargetTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._UMap[key] = new Dense<Scalar>; 
                C._VMap[key] = new Dense<Scalar>;
                // TODO: This could be removed by modifying the PassData
                //       unpacking routine to perform this step.
                hmat_tools::Copy( *C._ZMap[key], *C._UMap[key] );
                hmat_tools::Copy( SFB.D, *C._VMap[key] );
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            C._UMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( *C._ZMap[key], *C._UMap[key] );
            break;
        }
        case LOW_RANK:
        {
            // We are the middle and right process
             const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;    
             C._VMap[key] = new Dense<Scalar>;
             hmat_tools::Copy( FB.V, *C._VMap[key] );
             break;
        }
        case LOW_RANK_GHOST:
        {
            // We are the left process
            C._UMap[key] = new Dense<Scalar>; 
            // TODO: This could be removed by modifying the PassData
            //       unpacking routine to perform this step.
            hmat_tools::Copy( *C._ZMap[key], *C._UMap[key] );
            break;
        }
        case SPLIT_DENSE:
            if( C._inSourceTeam )
            {
                const SplitDense& SDB = *B._block.data.SD;
                Dense<Scalar>& DC = *C._DMap[key];
                if( admissibleC )
                {
                    Dense<Scalar> D = *C._DMap[key];
                    blas::Gemm
                    ( 'N', 'N', C.Height(), C.Width(), A.Width(),
                      alpha,     D.LockedBuffer(),     D.LDim(),
                                 SDB.D.LockedBuffer(), SDB.D.LDim(),
                      (Scalar)0, DC.Buffer(),          DC.LDim() );
                }
                else
                {
                    Dense<Scalar>& D = *C._block.data.D;
                    blas::Gemm
                    ( 'N', 'N', C.Height(), C.Width(), A.Width(),
                      alpha,     DC.LockedBuffer(),    DC.LDim(),
                                 SDB.D.LockedBuffer(), SDB.D.LDim(),
                      (Scalar)1, D.Buffer(),           D.LDim() );
                }
                DC.Clear();
            }
            break;
        case SPLIT_DENSE_GHOST:
            // We are the left process.
            break;
        case DENSE:
            // We are the right process and there is nothing left to do.
            break;
        case DENSE_GHOST:
            // We are the left process.
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
    {
        // We are the right process
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFB.D, *C._VMap[key] );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            Dense<Scalar>& DC = *C._DMap[key];
            if( admissibleC )
            {
                Dense<Scalar> D = *C._DMap[key];
                blas::Gemm
                ( 'N', 'N', C.Height(), C.Width(), A.Width(),
                  alpha,     D.LockedBuffer(),     D.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  (Scalar)0, DC.Buffer(),          DC.LDim() );
            }
            else
            {
                Dense<Scalar>& D = C._block.data.SD->D;
                blas::Gemm
                ( 'N', 'N', C.Height(), C.Width(), A.Width(),
                  alpha,     DC.LockedBuffer(),    DC.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  (Scalar)1, D.Buffer(),           D.LDim() );
            }
            DC.Clear();
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
    case DENSE:
        break;
    case DENSE_GHOST:
    {
        // We are the right process
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap[key] = new Dense<Scalar>;
            hmat_tools::Copy( SFB.D, *C._VMap[key] );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            Dense<Scalar>& DC = *C._DMap[key];
            if( admissibleC )
            {
                Dense<Scalar> D = *C._DMap[key];
                blas::Gemm
                ( 'N', 'N', C.Height(), C.Width(), A.Width(),
                  alpha,     D.LockedBuffer(),     D.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  (Scalar)0, DC.Buffer(),          DC.LDim() );
            }
            else
            {
                Dense<Scalar>& D = C._block.data.SD->D;
                blas::Gemm
                ( 'N', 'N', C.Height(), C.Width(), A.Width(),
                  alpha,     DC.LockedBuffer(),    DC.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  (Scalar)1, D.Buffer(),           D.LDim() );
            }
            DC.Clear();
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
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeCCleanup()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeCCleanup");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& C = *this;
    C._mainContextMap.Clear();
    C._ZMap.Clear();

    switch( C._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        Node& nodeC = *C._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeC.Child(t,s).MultiplyHMatMainPostcomputeCCleanup();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPrecompute
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPrecompute");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const int paddedRank = C.MaxRank() + 4;

    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                {
                    C._colFHHContextMap[key] = new MultiplyDenseContext;
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    context.numRhs = paddedRank;
                    if( A._inTargetTeam )
                    {
                        C._colXMap[key] = 
                            new Dense<Scalar>( A.LocalHeight(), paddedRank );
                        Dense<Scalar>& X = *C._colXMap[key];
                        hmat_tools::Scale( (Scalar)0, X );
                    }
                    if( A._inSourceTeam && A._inTargetTeam )
                    {
                        Dense<Scalar>& X = *C._colXMap[key];
                        A.MultiplyDensePrecompute( context, alpha, B._colT, X );
                    }
                    else if( A._inSourceTeam )
                    {
                        Dense<Scalar> dummy( 0, paddedRank );
                        A.MultiplyDensePrecompute
                        ( context, alpha, B._colT, dummy );
                    }
                    else // A._inTargetTeam
                    {
                        Dense<Scalar>& X = *C._colXMap[key];
                        Dense<Scalar> dummy( 0, paddedRank );
                        A.MultiplyDensePrecompute( context, alpha, dummy, X );
                    }
                }
                if( B._inSourceTeam || B._inTargetTeam )
                {
                    C._rowFHHContextMap[key] = new MultiplyDenseContext;
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    context.numRhs = paddedRank;
                    if( B._inSourceTeam )
                    {
                        C._rowXMap[key] = 
                            new Dense<Scalar>( B.LocalWidth(), paddedRank );
                        Dense<Scalar>& X = *C._rowXMap[key];
                        hmat_tools::Scale( (Scalar)0, X );
                    }
                    if( B._inSourceTeam && B._inTargetTeam )
                    {
                        Dense<Scalar>& X = *C._rowXMap[key];
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), A._rowT, X );
                    }
                    else if( B._inTargetTeam )
                    {
                        Dense<Scalar> dummy( 0, paddedRank );
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), A._rowT, dummy );
                    }
                    else // B._inSourceTeam
                    {
                        Dense<Scalar>& X = *C._rowXMap[key];
                        Dense<Scalar> dummy( 0, paddedRank );
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), dummy, X );
                    }
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPrecompute
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
            }
            break;
        }
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSummations
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSummations");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each reduce
    // (the first and last comms are unneeded)
    const int numLevels = _teams->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    // TODO
    /*
    A.MultiplyHMatFHHSummationsCount( B, C, sizes );
    */

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    // TODO
    /*
    A.MultiplyHMatFHHSummationsPack( B, C, buffer, offsets );
    */

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

    // Unpack the reduced buffers (only roots of communicators have data)
    // TODO
    /*
    A.MultiplyHMatFHHSummationsUnpack( B, C, buffer, offsets );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassData
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassData");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // 1) Compute send and recv sizes
    MPI_Comm comm = _teams->Team( 0 );
    const int p = mpi::CommSize( comm );
    std::vector<int> sendSizes(p,0), recvSizes(p,0);
    // TODO
    /*
    A.MultiplyHMatFHHPassDataCount( B, C, sendSizes, recvSizes );
    */

    // 2) Allocate buffers
    int totalSendSize=0, totalRecvSize=0;
    for( int i=0; i<p; ++i )
    {
        totalSendSize += sendSizes[i];
        totalRecvSize += recvSizes[i];
    }
    std::vector<Scalar> sendBuffer(totalSendSize), recvBuffer(totalRecvSize);
    std::vector<int> sendOffsets( p ), recvOffsets( p );
    for( int i=0,offset=0; i<p; offset+=sendSizes[i],++i )
        sendOffsets[i] = offset;
    for( int i=0,offset=0; i<p; offset+=recvSizes[i],++i )
        recvOffsets[i] = offset;

    // 3) Pack sends
    std::vector<int> offsets = sendOffsets;
    // TODO
    /*
    A.MultiplyHMatFHHPassDataPack( B, C, sendBuffer, offsets );
    */

    // 4) MPI_Alltoallv
    mpi::AllToAllV
    ( &sendBuffer[0], &sendSizes[0], &sendOffsets[0],
      &recvBuffer[0], &recvSizes[0], &recvOffsets[0], comm );

    // 5) Unpack recvs
    offsets = recvOffsets;
    // TODO
    /*
    A.MultiplyHMatFHHPassDataUnpack( B, C, recvBuffer, offsets );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcasts
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcasts");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    // TODO
    /*
    A.MultiplyHMatFHHBroadcastsCount( B, C, sizes );
    */

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    // TODO
    /*
    A.MultiplyHMatFHHBroadcastsPack( B, C, buffer, offsets );
    */

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
    // TODO
    /*
    A.MultiplyHMatFHHBroadcastsUnpack( B, C, buffer, offsets );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcompute
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcompute");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A.MultiplyHMatFHHPostcomputeC( alpha, B, C );
    C.MultiplyHMatFHHPostcomputeCCleanup();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcomputeC
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcomputeC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const int paddedRank = C.MaxRank() + 4;

    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                // Finish computing A B Omega1
                if( A._inSourceTeam && A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    Dense<Scalar>& X = *C._colXMap[key];
                    A.MultiplyDensePostcompute( context, alpha, B._colT, X );
                }
                else if( A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    Dense<Scalar>& X = *C._colXMap[key];
                    Dense<Scalar> dummy( 0, paddedRank );
                    A.MultiplyDensePostcompute( context, alpha, dummy, X );
                }

                // Finish computing B' A' Omega2
                if( B._inSourceTeam && B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    Dense<Scalar>& X = *C._rowXMap[key];
                    B.AdjointMultiplyDensePostcompute
                    ( context, Conj(alpha), A._rowT, X );
                }
                else if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    Dense<Scalar>& X = *C._rowXMap[key];
                    Dense<Scalar> dummy( 0, paddedRank );
                    B.AdjointMultiplyDensePostcompute
                    ( context, Conj(alpha), dummy, X );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPostcomputeC
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s) );
            }
            break;
        }
        default:
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcomputeCCleanup()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcomputeCCleanup");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& C = *this;
    C._colFHHContextMap.Clear();
    C._rowFHHContextMap.Clear();

    switch( C._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        Node& nodeC = *C._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeC.Child(t,s).MultiplyHMatFHHPostcomputeCCleanup();
        break;
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

