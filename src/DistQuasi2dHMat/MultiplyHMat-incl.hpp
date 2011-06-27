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
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
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
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();

    A.MultiplyHMatFormGhostRanks( B );

    A.MultiplyHMatMainPrecompute( alpha, B, C );
    A.MultiplyHMatMainSums( B, C );
    A.MultiplyHMatMainPassData( alpha, B, C );
    A.MultiplyHMatMainBroadcasts( B, C );
    A.MultiplyHMatMainPostcompute( alpha, B, C );

    A.MultiplyHMatFHHPrecompute( alpha, B, C );
    A.MultiplyHMatFHHSums( alpha, B, C );
    A.MultiplyHMatFHHPassData( alpha, B, C );
    A.MultiplyHMatFHHBroadcasts( alpha, B, C );
    A.MultiplyHMatFHHPostcompute( alpha, B, C );
    A.MultiplyHMatFHHFinalize( B, C );

    C.MultiplyHMatUpdates();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanks
( DistQuasi2dHMat<Scalar,Conjugated>& B ) 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanks");
#endif
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatFormGhostRanksCount( B, sendSizes, recvSizes );
    // HERE
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFormGhostRanksCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B ,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFormGhostRanksCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // HERE
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE_GHOST:
        case NODE_GHOST:
        {

            break;
        }
        case DIST_LOW_RANK_GHOST:
        {

            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {

            break;
        }
        case LOW_RANK_GHOST:
        {

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
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPrecompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const int sampleRank = SampleRank( C.MaxRank() );
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
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
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
                        A._rowOmega.Resize( A.LocalHeight(), sampleRank );
                        ParallelGaussianRandomVectors( A._rowOmega );
                    }
                    if( A._inSourceTeam )
                    {
                        A._rowT.Resize( A.LocalWidth(), sampleRank );
                        hmat_tools::Scale( (Scalar)0, A._rowT );
                    }
                    A.AdjointMultiplyDenseInitialize
                    ( A._rowContext, sampleRank );
                    if( A._inSourceTeam && A._inTargetTeam )
                    {
                        A.AdjointMultiplyDensePrecompute
                        ( A._rowContext, (Scalar)1, A._rowOmega, A._rowT );
                    }
                    else if( A._inSourceTeam )
                    {
                        Dense<Scalar> dummy( 0, sampleRank );
                        A.AdjointMultiplyDensePrecompute
                        ( A._rowContext, (Scalar)1, dummy, A._rowT );
                    }
                    else // A._inTargetTeam
                    {
                        Dense<Scalar> dummy( 0, sampleRank );
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
                        B._colOmega.Resize( B.LocalWidth(), sampleRank ); 
                        ParallelGaussianRandomVectors( B._colOmega );
                    }
                    if( B._inTargetTeam )
                    {
                        B._colT.Resize( B.LocalHeight(), sampleRank );
                        hmat_tools::Scale( (Scalar)0, B._colT );
                    }
                    B.MultiplyDenseInitialize( B._colContext, sampleRank );
                    if( B._inSourceTeam && B._inTargetTeam )
                    {
                        B.MultiplyDensePrecompute
                        ( B._colContext, (Scalar)1, B._colOmega, B._colT );
                    }
                    else if( B._inSourceTeam )
                    {
                        Dense<Scalar> dummy( 0, sampleRank );
                        B.MultiplyDensePrecompute
                        ( B._colContext, (Scalar)1, B._colOmega, dummy );
                    }
                    else // B._inTargetTeam
                    {
                        Dense<Scalar> dummy( 0, sampleRank );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSums
( DistQuasi2dHMat<Scalar,Conjugated>& B, DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSums");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each reduce
    const unsigned numTeamLevels = _teams->NumLevels();
    const unsigned numReduces = numTeamLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    A.MultiplyHMatMainSumsCountA( sizes );
    B.MultiplyHMatMainSumsCountB( sizes );
    A.MultiplyHMatMainSumsCountC( B, C, sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatMainSumsPackA( buffer, offsets );
    B.MultiplyHMatMainSumsPackB( buffer, offsets );
    A.MultiplyHMatMainSumsPackC( B, C, buffer, offsets );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    A._teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatMainSumsUnpackA( buffer, offsets );
    B.MultiplyHMatMainSumsUnpackB( buffer, offsets );
    A.MultiplyHMatMainSumsUnpackC( B, C, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsCountA
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseSumsCount( sizes, _rowContext.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSumsCountA( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseSumsPack( _rowContext, buffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSumsPackA( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganRowSpaceComp )
            TransposeMultiplyDenseSumsUnpack( _rowContext, buffer, offsets );

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSumsUnpackA( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsCountB
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseSumsCount( sizes, _colContext.numRhs );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSumsCountB( sizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseSumsPack( _colContext, buffer, offsets );

        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSumsPackB( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _beganColSpaceComp )
            MultiplyDenseSumsUnpack( _colContext, buffer, offsets );

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatMainSumsUnpackB( buffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsCountC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsCountC");
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
                            nodeA.Child(t,r).MultiplyHMatMainSumsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
            }
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            A.MultiplyDenseSumsCount( sizes, DFB.rank );
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
            B.TransposeMultiplyDenseSumsCount( sizes, DFA.rank );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inSourceTeam )
            {
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                sizes[teamLevel] += DFA.rank*DFB.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsPackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B, 
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsPackC");
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
                            nodeA.Child(t,r).MultiplyHMatMainSumsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
        case DIST_LOW_RANK:
            A.MultiplyDenseSumsPack( *C._mainContextMap[key], buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            B.TransposeMultiplyDenseSumsPack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inSourceTeam )
            {
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                std::memcpy
                ( &buffer[offsets[teamLevel]], C._ZMap[key]->LockedBuffer(),
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[teamLevel] += DFA.rank*DFB.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsUnpackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsUnpackC");
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
                            nodeA.Child(t,r).MultiplyHMatMainSumsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
        case DIST_LOW_RANK:
            A.MultiplyDenseSumsUnpack
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
            B.TransposeMultiplyDenseSumsUnpack
            ( *C._mainContextMap[key], buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFA = *A._block.data.DF;
            const DistLowRank& DFB = *B._block.data.DF;
            if( A._inSourceTeam )
            {
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                std::memcpy
                ( C._ZMap[key]->Buffer(), &buffer[offsets[teamLevel]],
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[teamLevel] += DFA.rank*DFB.rank;
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
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassData");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // 1) Compute send and recv sizes
    MPI_Comm comm = _teams->Team( 0 );
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatMainPassDataCountA( sendSizes, recvSizes );
    B.MultiplyHMatMainPassDataCountB( sendSizes, recvSizes );
    A.MultiplyHMatMainPassDataCountC( B, C, sendSizes, recvSizes );

    // 2) Allocate buffers
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
    }

    // Fill the send buffer
    std::vector<Scalar> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatMainPassDataPackA( sendBuffer, offsets );
    B.MultiplyHMatMainPassDataPackB( sendBuffer, offsets );
    A.MultiplyHMatMainPassDataPackC( B, C, sendBuffer, offsets );

    // Start the non-blocking sends
    const int numSends = sendSizes.size();
    std::vector<MPI_Request> sendRequests( numSends );
    int offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Start the non-blocking recvs
    const int numRecvs = recvSizes.size();
    std::vector<MPI_Request> recvRequests( numRecvs );
    std::vector<Scalar> recvBuffer( totalRecvSize );
    offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    for( int i=0; i<numRecvs; ++i )
        mpi::Wait( recvRequests[i] );
    A.MultiplyHMatMainPassDataUnpackA( recvBuffer, recvOffsets );
    B.MultiplyHMatMainPassDataUnpackB( recvBuffer, recvOffsets );
    A.MultiplyHMatMainPassDataUnpackC( B, C, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataCountA
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
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
( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets ) 
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

        Node& node = *_block.data.N;
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
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets ) 
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

        Node& node = *_block.data.N;
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
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
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
( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets )
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

        Node& node = *_block.data.N;
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
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets )
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

        Node& node = *_block.data.N;
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
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
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
                    AddToMap( sendSizes, A._targetRoot, DFA.rank*DFB.rank );
                if( A._inTargetTeam )
                    AddToMap( recvSizes, A._sourceRoot, DFA.rank*DFB.rank );
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F
            if( teamRank == 0 )
            {
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                AddToMap( recvSizes, A._sourceRoot, DFA.rank*DFGB.rank );
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
                AddToMap( sendSizes, A._targetRoot, SFA.rank*SFB.rank );
            else
                AddToMap( recvSizes, A._sourceRoot, SFA.rank*SFB.rank );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            AddToMap( recvSizes, A._sourceRoot, SFA.rank*SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Pass data for H/D/F += F F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            AddToMap( sendSizes, A._targetRoot, SFA.rank*FB.Rank() );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const LowRankGhost& FGB = *B._block.data.FG;
            AddToMap( recvSizes, A._sourceRoot, SFA.rank*FGB.rank );
            break;
        }
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B._inTargetTeam )
                AddToMap( sendSizes, B._sourceRoot, B.Height()*SFA.rank );
            else
                AddToMap( recvSizes, B._targetRoot, B.Height()*SFA.rank );
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
            AddToMap( recvSizes, B._targetRoot, B.Height()*SFGA.rank );
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
            AddToMap( sendSizes, B._sourceRoot, B.Height()*FA.Rank() );
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
            AddToMap( recvSizes, B._targetRoot, B.Height()*FGA.rank );
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
                AddToMap( sendSizes, A._targetRoot, A.Height()*SFB.rank );
            else
                AddToMap( recvSizes, A._sourceRoot, A.Height()*SFB.rank );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            AddToMap( recvSizes, A._sourceRoot, A.Height()*SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Pass data for D/F += D F
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            AddToMap( sendSizes, A._targetRoot, A.Height()*FB.Rank() );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const LowRankGhost& FGB = *B._block.data.FG;
            AddToMap( recvSizes, A._sourceRoot, A.Height()*FGB.rank );
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B._inSourceTeam )
                AddToMap( recvSizes, B._targetRoot, A.Height()*A.Width() );
            else
                AddToMap( sendSizes, B._sourceRoot, A.Height()*A.Width() );
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
            AddToMap( recvSizes, B._targetRoot, A.Height()*A.Width() );
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
            AddToMap( sendSizes, B._sourceRoot, A.Height()*A.Width() );
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
            AddToMap( recvSizes, B._targetRoot, A.Height()*A.Width() );
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
  std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets ) const
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
                if( A._inSourceTeam && DFA.rank != 0 && DFB.rank != 0 )
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
            if( A._inSourceTeam && SFA.rank != 0 && SFB.rank != 0 )
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
            if( SFA.rank != 0 && FB.Rank() != 0 )
            {
                Dense<Scalar>& ZC = *C._ZMap[key];
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                  SFA.rank*FB.Rank()*sizeof(Scalar) );
                offsets[A._targetRoot] += SFA.rank*FB.Rank();
                ZC.Clear();
            }
            break;
        }
        case LOW_RANK_GHOST:
            // Pass data for H/D/F += F F is just a receive for us
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B._inTargetTeam && B.Height() != 0 && SFA.rank != 0 )
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
            if( B.Height() != 0 && FA.Rank() != 0 )
            {
                std::memcpy
                ( &sendBuffer[offsets[B._sourceRoot]], FA.V.LockedBuffer(),
                  B.Height()*FA.Rank()*sizeof(Scalar) );
                offsets[B._sourceRoot] += B.Height()*FA.Rank();
            }
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
            if( A._inSourceTeam && A.Height() != 0 && SFB.rank != 0 )
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
            if( A.Height() != 0 && FB.Rank() != 0 )
            {
                Dense<Scalar>& ZC = *C._ZMap[key];
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(), 
                  A.Height()*FB.Rank()*sizeof(Scalar) );
                offsets[A._targetRoot] += A.Height()*FB.Rank();
                ZC.Clear();
            }
            break;
        }
        case LOW_RANK_GHOST:
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B._inTargetTeam && A.Height() != 0 && A.Width() != 0 )
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
            if( A.Height() != 0 && A.Width() != 0 )
            {
                const Dense<Scalar>& DA = *A._block.data.D;
                std::memcpy
                ( &sendBuffer[offsets[B._sourceRoot]], DA.LockedBuffer(),
                  A.Height()*A.Width()*sizeof(Scalar) );
                offsets[B._sourceRoot] += A.Height()*A.Width();
            }
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
  const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets ) const
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
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    C._ZMap[key] = new Dense<Scalar>( DFA.rank, DFB.rank );
                    Dense<Scalar>& ZC = *C._ZMap[key];

                    std::memcpy
                    ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[A._sourceRoot] += DFA.rank*DFB.rank;
                }
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F
            if( teamRank == 0 )
            {
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                if( DFA.rank != 0 && DFGB.rank != 0 )
                {
                    C._ZMap[key] = new Dense<Scalar>( DFA.rank, DFGB.rank );
                    Dense<Scalar>& ZC = *C._ZMap[key];

                    std::memcpy
                    ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                      DFA.rank*DFGB.rank*sizeof(Scalar) );
                    offsets[A._sourceRoot] += DFA.rank*DFGB.rank;
                }
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
            if( A._inTargetTeam && SFA.rank != 0 && SFB.rank != 0 )
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
            if( SFA.rank != 0 && SFGB.rank != 0 )
            {
                C._ZMap[key] = new Dense<Scalar>( SFA.rank, SFGB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  SFA.rank*SFGB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += SFA.rank*SFGB.rank;
            }
            break;
        }
        case LOW_RANK:
            // Pass data for H/D/F += F F is a send
            break;
        case LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const LowRankGhost& FGB = *B._block.data.FG;
            if( SFA.rank != 0 && FGB.rank != 0 )
            {
                C._ZMap[key] = new Dense<Scalar>( SFA.rank, FGB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  SFA.rank*FGB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += SFA.rank*FGB.rank;
            }
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            if( B._inSourceTeam && B.Height() != 0 && SFA.rank != 0 )
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
            if( B._inSourceTeam && B.Height() != 0 && SFGA.rank != 0 )
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
            if( B.Height() != 0 && FGA.rank != 0 )
            {
                C._ZMap[key] = new Dense<Scalar>( B.Height(), FGA.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  B.Height()*FGA.rank*sizeof(Scalar) );
                offsets[B._targetRoot] += B.Height()*FGA.rank;
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
    case SPLIT_DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            // Pass data for D/F += D F
            if( A._inTargetTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                if( A.Height() != 0 && SFB.rank != 0 )
                {
                    C._ZMap[key] = new Dense<Scalar>( A.Height(), SFB.rank );
                    Dense<Scalar>& ZC = *C._ZMap[key];

                    std::memcpy
                    ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                      A.Height()*SFB.rank*sizeof(Scalar) );
                    offsets[A._sourceRoot] += A.Height()*SFB.rank;
                }
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            if( A.Height() != 0 && SFGB.rank != 0 )
            {
                C._ZMap[key] = new Dense<Scalar>( A.Height(), SFGB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  A.Height()*SFGB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += A.Height()*SFGB.rank;
            }
            break;
        }
        case LOW_RANK:
            break;
        case LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const LowRankGhost& FGB = *B._block.data.FG;
            if( A.Height() != 0 && FGB.rank != 0 )
            {
                C._ZMap[key] = new Dense<Scalar>( A.Height(), FGB.rank );
                Dense<Scalar>& ZC = *C._ZMap[key];

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  A.Height()*FGB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += A.Height()*FGB.rank;
            }
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B._inSourceTeam && A.Height() != 0 && A.Width() != 0 )
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
            if( A.Height() != 0 && A.Width() != 0 )
            {
                C._DMap[key] = new Dense<Scalar>( A.Height(), A.Width() );
                Dense<Scalar>& DC = *C._DMap[key];

                std::memcpy
                ( DC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  A.Height()*A.Width()*sizeof(Scalar) );
                offsets[B._targetRoot] += A.Height()*A.Width();
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
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcasts");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each broadcast
    const unsigned numTeamLevels = _teams->NumLevels();
    const unsigned numBroadcasts = numTeamLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    A.MultiplyHMatMainBroadcastsCountA( sizes );
    B.MultiplyHMatMainBroadcastsCountB( sizes );
    A.MultiplyHMatMainBroadcastsCountC( B, C, sizes );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( unsigned i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatMainBroadcastsPackA( buffer, offsets );
    B.MultiplyHMatMainBroadcastsPackB( buffer, offsets );
    A.MultiplyHMatMainBroadcastsPackC( B, C, buffer, offsets );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    A._teams->TreeBroadcasts( buffer, sizes, offsets );

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
            TransposeMultiplyDenseBroadcastsPack
            ( _rowContext, buffer, offsets );

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
( const std::vector<Scalar>& buffer, std::vector<int>& offsets )
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
( const std::vector<Scalar>& buffer, std::vector<int>& offsets )
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
            {
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                sizes[teamLevel] += DFA.rank*DFB.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsPackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B, 
        DistQuasi2dHMat<Scalar,Conjugated>& C,
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
                Node& nodeC = *C._block.data.N;
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
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                std::memcpy
                ( &buffer[offsets[teamLevel]], C._ZMap[key]->LockedBuffer(),
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[teamLevel] += DFA.rank*DFB.rank;
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
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                std::memcpy
                ( C._ZMap[key]->Buffer(), &buffer[offsets[teamLevel]],
                  DFA.rank*DFB.rank*sizeof(Scalar) );
                offsets[teamLevel] += DFA.rank*DFB.rank;
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
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeA()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeA");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

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
        Node& nodeA = *A._block.data.N;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeB()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeB");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& B = *this;

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
        Node& nodeB = *B._block.data.N;
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
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPrecompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const int sampleRank = SampleRank( C.MaxRank() );

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
                    context.numRhs = sampleRank;
                    if( A._inTargetTeam )
                    {
                        C._colXMap[key] = 
                            new Dense<Scalar>( A.LocalHeight(), sampleRank );
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
                        Dense<Scalar> dummy( 0, sampleRank );
                        A.MultiplyDensePrecompute
                        ( context, alpha, B._colT, dummy );
                    }
                    else // A._inTargetTeam
                    {
                        Dense<Scalar>& X = *C._colXMap[key];
                        Dense<Scalar> dummy( 0, sampleRank );
                        A.MultiplyDensePrecompute( context, alpha, dummy, X );
                    }
                }
                if( B._inSourceTeam || B._inTargetTeam )
                {
                    C._rowFHHContextMap[key] = new MultiplyDenseContext;
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    context.numRhs = sampleRank;
                    if( B._inSourceTeam )
                    {
                        C._rowXMap[key] = 
                            new Dense<Scalar>( B.LocalWidth(), sampleRank );
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
                        Dense<Scalar> dummy( 0, sampleRank );
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), A._rowT, dummy );
                    }
                    else // B._inSourceTeam
                    {
                        Dense<Scalar>& X = *C._rowXMap[key];
                        Dense<Scalar> dummy( 0, sampleRank );
                        B.AdjointMultiplyDensePrecompute
                        ( context, Conj(alpha), dummy, X );
                    }
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSums
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSums");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each reduce
    const unsigned numTeamLevels = _teams->NumLevels();
    const unsigned numReduces = numTeamLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    A.MultiplyHMatFHHSumsCount( B, C, sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatFHHSumsPack( B, C, buffer, offsets );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    A._teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatFHHSumsUnpack( B, C, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._inSourceTeam )
                    A.MultiplyDenseSumsCount( sizes, sampleRank );
                if( B._inTargetTeam )
                    B.TransposeMultiplyDenseSumsCount
                    ( sizes, sampleRank );
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsPack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsPack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseSumsPack( context, buffer, offsets );
                }
                if( B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseSumsPack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHSumsUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHSumsUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseSumsUnpack( context, buffer, offsets );
                }
                if( B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseSumsUnpack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassData
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassData");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // 1) Compute send and recv sizes
    MPI_Comm comm = _teams->Team( 0 );
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFHHPassDataCount( B, C, sendSizes, recvSizes );

    // 2) Allocate buffers
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
    }

    // Fill the send buffer
    std::vector<Scalar> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatFHHPassDataPack( B, C, sendBuffer, offsets );

    // Start the non-blocking sends
    const int numSends = sendSizes.size();
    std::vector<MPI_Request> sendRequests( numSends );
    int offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Start the non-blocking recvs
    const int numRecvs = recvSizes.size();
    std::vector<MPI_Request> recvRequests( numRecvs );
    std::vector<Scalar> recvBuffer( totalRecvSize );
    offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    for( int i=0; i<numRecvs; ++i )
        mpi::Wait( recvRequests[i] );
    A.MultiplyHMatFHHPassDataUnpack( B, C, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void 
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassDataCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                    A.MultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
                if( B._inSourceTeam || B._inTargetTeam )
                    B.TransposeMultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendSizes, recvSizes );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassDataPack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataPack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDensePassDataPack
                    ( context, sendBuffer, offsets );
                }

                if( B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDensePassDataPack
                    ( context, A._rowT, sendBuffer, offsets );
                }
                else if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    Dense<Scalar> dummy( 0, sampleRank );
                    B.TransposeMultiplyDensePassDataPack
                    ( context, dummy, sendBuffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              sendBuffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPassDataUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPassDataUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();

    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A._inSourceTeam || A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDensePassDataUnpack
                    ( context, recvBuffer, offsets );
                }
                if( B._inSourceTeam || B._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDensePassDataUnpack
                    ( context, recvBuffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              recvBuffer, offsets );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcasts
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcasts");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute the message sizes for each broadcast
    const unsigned numTeamLevels = _teams->NumLevels();
    const unsigned numBroadcasts = numTeamLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    A.MultiplyHMatFHHBroadcastsCount( B, C, sizes );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( unsigned i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    A.MultiplyHMatFHHBroadcastsPack( B, C, buffer, offsets );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    A._teams->TreeBroadcasts( buffer, sizes, offsets );

    // Unpack the broadcasted buffers
    A.MultiplyHMatFHHBroadcastsUnpack( B, C, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsCount
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsCount");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._inTargetTeam )
                    A.MultiplyDenseBroadcastsCount( sizes, sampleRank );
                if( B._inSourceTeam )
                    B.TransposeMultiplyDenseBroadcastsCount
                    ( sizes, sampleRank );
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsPack
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsPack");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseBroadcastsPack( context, buffer, offsets );
                }
                if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseBroadcastsPack
                    ( context, buffer, offsets );
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
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHBroadcastsUnpack
( DistQuasi2dHMat<Scalar,Conjugated>& B,
  DistQuasi2dHMat<Scalar,Conjugated>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHBroadcastsUnpack");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A._inTargetTeam )
                {
                    MultiplyDenseContext& context = *C._colFHHContextMap[key];
                    A.MultiplyDenseBroadcastsUnpack( context, buffer, offsets );
                }
                if( B._inSourceTeam )
                {
                    MultiplyDenseContext& context = *C._rowFHHContextMap[key];
                    B.TransposeMultiplyDenseBroadcastsUnpack
                    ( context, buffer, offsets );
                }
            }
            else
            {
                Node& nodeA = *A._block.data.N;
                Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHPostcompute
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHPostcompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

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
    const int sampleRank = SampleRank( C.MaxRank() );

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
                    Dense<Scalar> dummy( 0, sampleRank );
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
                    Dense<Scalar> dummy( 0, sampleRank );
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

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalize
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalize");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    const int r = SampleRank( C.MaxRank() );
    const unsigned numTeamLevels = C._teams->NumLevels();
    std::vector<int> numQRs(numTeamLevels,0), 
                     numTargetFHH(numTeamLevels,0), 
                     numSourceFHH(numTeamLevels,0);
    C.MultiplyHMatFHHFinalizeCounts( numQRs, numTargetFHH, numSourceFHH );

    // Set up the space for the packed 2r x r matrices and taus.
    int numTotalQRs=0, qrTotalSize=0, tauTotalSize=0;
    std::vector<int> XOffsets(numTeamLevels),
                     qrOffsets(numTeamLevels), qrPieceSizes(numTeamLevels), 
                     tauOffsets(numTeamLevels), tauPieceSizes(numTeamLevels);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = C._teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        XOffsets[teamLevel] = numTotalQRs;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        qrPieceSizes[teamLevel] = log2TeamSize*(r*r+r);
        tauPieceSizes[teamLevel] = (log2TeamSize+1)*r;

        numTotalQRs += numQRs[teamLevel];
        qrTotalSize += numQRs[teamLevel]*qrPieceSizes[teamLevel];
        tauTotalSize += numQRs[teamLevel]*tauPieceSizes[teamLevel];
    }

    std::vector<Dense<Scalar>*> Xs( numTotalQRs );
    std::vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize ),
                        work( lapack::QRWorkSize( r ) );

    // Form our contributions to Omega2' (alpha A B Omega1) updates here, 
    // before we overwrite the _colXMap and _rowXMap results.
    // The distributed summations will not occur until after the parallel
    // QR factorizations.
    std::vector<int> leftOffsets(numTeamLevels), middleOffsets(numTeamLevels), 
                     rightOffsets(numTeamLevels);
    int totalAllReduceSize = 0;
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        leftOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numTargetFHH[teamLevel]*r*r;
        middleOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numTargetFHH[teamLevel]*r*r;
        rightOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numSourceFHH[teamLevel]*r*r;
    }
    std::vector<Scalar> allReduceBuffer( totalAllReduceSize );
    A.MultiplyHMatFHHFinalizeMiddleUpdates
    ( B, C, allReduceBuffer, middleOffsets );

    // Perform the large local QR's and pack into the QR buffer as appropriate
    C.MultiplyHMatFHHFinalizeLocalQR
    ( Xs, XOffsets, qrBuffer, qrOffsets, tauBuffer, tauOffsets, work );

    // Reset the offset vectors
    numTotalQRs = qrTotalSize = tauTotalSize = 0;
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        XOffsets[teamLevel] = numTotalQRs;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;
    }

    // Perform the combined distributed TSQR factorizations.
    // This could almost certainly be simplified...
    const int numSteps = numTeamLevels-1;
    for( int step=0; step<numSteps; ++step )
    {
        MPI_Comm team = C._teams->Team( step );
        const int teamSize = mpi::CommSize( team );
        const unsigned teamRank = mpi::CommRank( team );
        const bool haveAnotherComm = ( step < numSteps-1 );
        // only valid result if we have a next step...
        const bool rootOfNextStep = !(teamRank & 0x100); 
        const int passes = 2*step;

        // Flip the first bit of our rank in this team to get our partner,
        // and then check if our bit is 0 to see if we're the root
        const unsigned firstPartner = teamRank ^ 0x1;
        const bool firstRoot = !(teamRank & 0x1);

        // Count the messages to send/recv to/from firstPartner
        int msgSize = 0;
        for( int j=0; j<numSteps-step; ++j )
            msgSize += numQRs[j]*(r*r+r)/2;
        std::vector<Scalar> sendBuffer( msgSize ), recvBuffer( msgSize );

        // Pack the messages for the firstPartner
        int sendOffset = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            for( int k=0; k<numQRs[l]; ++k )
            {
                if( firstRoot )
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffsets[l]+k*qrPieceSizes[l]+
                                    passes*(r*r+r)+(j*j+j)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
                else
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffsets[l]+k*qrPieceSizes[l]+
                                    passes*(r*r+r)+(j*j+j)+(j+1)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
            }
        }

        // Exchange with our first partner
        mpi::SendRecv
        ( &sendBuffer[0], msgSize, firstPartner, 0,
          &recvBuffer[0], msgSize, firstPartner, 0, team );

        if( teamSize == 4 )
        {
            // Flip the second bit in our team rank to get our partner, and
            // then check if our bit is 0 to see if we're the root
            const unsigned secondPartner = teamRank ^ 0x10;
            const bool secondRoot = !(teamRank & 0x10);

            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step and into the next 
            // send buffer in a single sweep.
            sendOffset = 0;
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const unsigned thisQROffset = 
                        qrOffsets[l]+k*qrPieceSizes[l]+passes*(r*r+r);
                    const unsigned thisTauOffset = 
                        tauOffsets[l]+k*tauPieceSizes[l]+(passes+1)*r;

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[thisQROffset], &tauBuffer[thisTauOffset],
                      &work[0] );
                    if( secondRoot )
                    {
                        // Copy into the upper triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
                    else
                    {
                        // Copy into the lower triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)+
                                        (j+1)],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[thisQROffset+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
                }
            }
            
            // Exchange with our second partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, secondPartner, 0,
              &recvBuffer[0], msgSize, secondPartner, 0, team );
            
            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step when necessary.
            recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const unsigned thisQROffset = 
                        qrOffsets[l]+k*qrPieceSizes[l]+(passes+1)*(r*r+r);
                    const unsigned thisTauOffset = 
                        tauOffsets[l]+k*tauPieceSizes[l]+(passes+2)*r;

                    if( secondRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }

                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[thisQROffset], &tauBuffer[thisTauOffset],
                      &work[0] );
                    if( haveAnotherComm )
                    {
                        if( rootOfNextStep )
                        {
                            // Copy into the upper triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)],
                                  &qrBuffer[thisQROffset+(j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                        else
                        {
                            // Copy into the lower triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[thisQROffset+(r*r+r)+(j*j+j)+(j+1)],
                                  &qrBuffer[thisQROffset+(j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                    }
                }
            }
        }
        else // teamSize == 2
        {
            // Exchange with our partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, firstPartner, 0,
              &recvBuffer[0], msgSize, firstPartner, 0, team );

            // Unpack the recv messages and perform the QR factorizations
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const unsigned thisQROffset = 
                        qrOffsets[l]+k*qrPieceSizes[l]+passes*(r*r+r);
                    const unsigned thisTauOffset = 
                        tauOffsets[l]+k*tauPieceSizes[l]+(passes+1)*r;

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[thisQROffset+(j*j+j)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[thisQROffset], &tauBuffer[thisTauOffset],
                      &work[0] );
                }
            }
        }
    }

    // Explicitly form the Q's
    Dense<Scalar> Z( 2*r, r );
    std::vector<Scalar> applyQWork( r, 1 );
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = C._teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );
        const unsigned teamRank = mpi::CommRank( team );
        const Scalar* thisQRLevel = &qrBuffer[qrOffsets[teamLevel]];
        const Scalar* thisTauLevel = &tauBuffer[tauOffsets[teamLevel]];

        for( int k=0; k<numQRs[teamLevel]; ++k )
        {
            const Scalar* thisQRPiece = &thisQRLevel[k*qrPieceSizes[teamLevel]];
            const Scalar* thisTauPiece = 
                &thisTauLevel[k*tauPieceSizes[teamLevel]];
            const Scalar* lastQRStage = &thisQRPiece[(log2TeamSize-1)*(r*r+r)];
            const Scalar* lastTauStage = &thisTauPiece[log2TeamSize*r];

            if( log2TeamSize > 0 )
            {
                // Form the identity matrix in the top r x r submatrix
                // of a zeroed 2r x r matrix.
                std::memset( Z.Buffer(), 0, 2*r*r*sizeof(Scalar) );
                for( int j=0; j<r; ++j )
                    Z.Set(j,j,(Scalar)1);
                // Backtransform the last stage
                hmat_tools::ApplyPackedQFromLeft
                ( r, lastQRStage, lastTauStage, Z, &work[0] );
                // Take care of the middle stages before handling the large 
                // original stage.
                for( int commStage=log2TeamSize-2; commStage>=0; --commStage )
                {
                    const bool rootOfLastStage = 
                        !(teamRank & (1u<<(commStage+1)));
                    if( rootOfLastStage )
                    {
                        // Zero the bottom half of Z
                        for( int j=0; j<r; ++j )
                            std::memset( Z.Buffer(r,j), 0, r*sizeof(Scalar) );
                    }
                    else
                    {
                        // Move the bottom half to the top half and zero the
                        // bottom half
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( Z.Buffer(0,j), Z.LockedBuffer(r,j), 
                              r*sizeof(Scalar) );
                            std::memset( Z.Buffer(r,j), 0, r*sizeof(Scalar) );
                        }
                    }
                    hmat_tools::ApplyPackedQFromLeft
                    ( r, &thisQRPiece[commStage*(r*r+r)], 
                      &thisTauPiece[(commStage+1)*r], Z, &work[0] );
                }
                // Take care of the original stage. Do so by forming Y := X, 
                // then zeroing X and placing our piece of Z at its top.
                Dense<Scalar>& X = *Xs[XOffsets[teamLevel]+k]; 
                const int m = X.Height();
                const int minDim = std::min( m, r );
                Dense<Scalar> Y;
                hmat_tools::Copy( X, Y );
                hmat_tools::Scale( (Scalar)0, X );
                const bool rootOfLastStage = !(teamRank & 0x1);
                if( rootOfLastStage )
                {
                    // Copy the first minDim rows of the top half of Z into
                    // the top of X
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( X.Buffer(0,j), Z.LockedBuffer(0,j), 
                          minDim*sizeof(Scalar) );
                }
                else
                {
                    // Copy the first minDim rows of the bottom half of Z into 
                    // the top of X
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( X.Buffer(0,j), Z.LockedBuffer(r,j), 
                          minDim*sizeof(Scalar) );
                }
                work.resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, minDim,
                  Y.LockedBuffer(), Y.LDim(), &thisTauPiece[0],
                  X.Buffer(),       X.LDim(), &work[0], work.size() );
            }
            else // this team only contains one process
            {
                Dense<Scalar>& X = *Xs[XOffsets[teamLevel]+k]; 
                const int m = X.Height();
                const int minDim = std::min(m,r);

                // Make a copy of X and then form the left part of identity.
                Dense<Scalar> Y; 
                Y = X;
                hmat_tools::Scale( (Scalar)0, X );
                for( int j=0; j<minDim; ++j )
                    X.Set(j,j,(Scalar)1);
                // Backtransform the last stage
                work.resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, minDim,
                  Y.LockedBuffer(), Y.LDim(), &thisTauPiece[0],
                  X.Buffer(),       X.LDim(), &work[0], work.size() );
            }
        }
    }
    XOffsets.clear();
    qrOffsets.clear(); qrPieceSizes.clear(); qrBuffer.clear();
    tauOffsets.clear(); tauPieceSizes.clear(); tauBuffer.clear();
    work.clear();
    Z.Clear();
    applyQWork.clear();

    A.MultiplyHMatFHHFinalizeOuterUpdates
    ( B, C, allReduceBuffer, leftOffsets, rightOffsets );

    // Perform a custom AllReduce on the buffers to finish forming
    // Q1' Omega2, Omega2' (alpha A B Omega1), and Q2' Omega1
    {
        // Reset the left/middle/right offsets and generate offsets and sizes 
        // for each entire level.
        std::vector<int> sizes, offsets;
        totalAllReduceSize = 0;
        for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
        {
            offsets[teamLevel] = totalAllReduceSize;
            sizes[teamLevel] = r*r*(2*numTargetFHH[teamLevel]+
                                      numSourceFHH[teamLevel]);

            leftOffsets[teamLevel] = totalAllReduceSize;
            totalAllReduceSize += numTargetFHH[teamLevel]*r*r;
            middleOffsets[teamLevel] = totalAllReduceSize;
            totalAllReduceSize += numTargetFHH[teamLevel]*r*r;
            rightOffsets[teamLevel] = totalAllReduceSize;
            totalAllReduceSize += numSourceFHH[teamLevel]*r*r;
        }

        A._teams->TreeSums( allReduceBuffer, sizes, offsets );
    }

    // Finish forming the low-rank approximation
    std::vector<Scalar> U( r*r ), VH( r*r ),
                        svdWork( lapack::SVDWorkSize(r,r) );
    std::vector<Real> singularValues( r ), 
                      svdRealWork( lapack::SVDRealWorkSize(r,r) ); 
    A.MultiplyHMatFHHFinalizeFormLowRank
    ( B, C, allReduceBuffer, leftOffsets, middleOffsets, rightOffsets,
      singularValues, U, VH, svdWork, svdRealWork );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeCounts
( std::vector<int>& numQRs, 
  std::vector<int>& numTargetFHH, std::vector<int>& numSourceFHH )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeCounts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatFHHFinalizeCounts
                ( numQRs, numTargetFHH, numSourceFHH );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
        if( _inTargetTeam )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            numQRs[teamLevel] += _colXMap.Size();
            ++numTargetFHH[teamLevel];
        }
        if( _inSourceTeam )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            numQRs[teamLevel] += _rowXMap.Size();
            ++numSourceFHH[teamLevel];
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeMiddleUpdates
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<Scalar>& allReduceBuffer,
        std::vector<int>& middleOffsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeMiddleUpdates");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int r = SampleRank( C.MaxRank() );

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
            if( C.Admissible() )
            {
                if( C._inTargetTeam ) 
                {
                    // Handle the middle update, Omega2' (alpha A B Omega1)
                    const Dense<Scalar>& X = *C._colXMap[A._sourceOffset];
                    const Dense<Scalar>& Omega2 = A._rowOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* middleUpdate = 
                        &allReduceBuffer[middleOffsets[teamLevel]];
                    middleOffsets[teamLevel] += r*r;

                    blas::Gemm
                    ( 'C', 'N', r, r, A.LocalHeight(),
                      (Scalar)1, Omega2.LockedBuffer(), Omega2.LDim(),
                                 X.LockedBuffer(),      X.LDim(),
                      (Scalar)0, middleUpdate,          r );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        nodeA.Child(t,r).
                        MultiplyHMatFHHFinalizeMiddleUpdates
                        ( nodeB.Child(r,s), nodeC.Child(t,s), 
                          allReduceBuffer, middleOffsets );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeLocalQR
( std::vector<Dense<Scalar>*>& Xs, std::vector<int>& XOffsets,
  std::vector<Scalar>& qrBuffer,  std::vector<int>& qrOffsets,
  std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& work )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeLocalQR");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatFHHFinalizeLocalQR
                ( Xs, XOffsets, qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
                  work );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        const int teamSize = mpi::CommSize( team );
        const int log2TeamSize = Log2( teamSize );
        const int r = SampleRank( MaxRank() );

        if( _inTargetTeam )
        {
            const unsigned numEntries = _colXMap.Size();
            const unsigned teamLevel = _teams->TeamLevel(_level);
            _colXMap.ResetIterator();
            for( unsigned i=0; i<numEntries; ++i )
            {
                Dense<Scalar>& X = *_colXMap.NextEntry();
                Xs[XOffsets[teamLevel]++] = &X;
                lapack::QR
                ( X.Height(), X.Width(), X.Buffer(), X.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
                tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                if( log2TeamSize > 0 )
                {
                    std::memset
                    ( &qrBuffer[qrOffsets[teamLevel]], 0, 
                      (r*r+r)*sizeof(Scalar) );
                    const bool root = !(teamRank & 0x1);
                    if( root )
                    {
                        // Copy our R into the upper triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                    else
                    {
                        // Copy our R into the lower triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                }
                qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            }
        }
        if( _inSourceTeam )
        {
            const int numEntries = _rowXMap.Size();
            const unsigned teamLevel = _teams->TeamLevel(_level);
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                Dense<Scalar>& X = *_rowXMap.NextEntry();
                Xs[XOffsets[teamLevel]++] = &X;
                lapack::QR
                ( X.Height(), X.Width(), X.Buffer(), X.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
                tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                if( log2TeamSize > 0 )
                {
                    std::memset
                    ( &qrBuffer[qrOffsets[teamLevel]], 0, 
                      (r*r+r)*sizeof(Scalar) );
                    const bool root = !(teamRank & 0x1);
                    if( root )
                    {
                        // Copy our R into the upper triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                    else
                    {
                        // Copy our R into the lower triangle of the next
                        // matrix to factor (which is 2r x r)
                        for( int j=0; j<X.Width(); ++j )
                            std::memcpy
                            ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                              X.LockedBuffer(0,j), 
                              std::min(X.Height(),j+1)*sizeof(Scalar) );
                    }
                }
                qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeOuterUpdates
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<Scalar>& allReduceBuffer,
        std::vector<int>& leftOffsets,
        std::vector<int>& rightOffsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeOuterUpdates");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int r = SampleRank( C.MaxRank() );

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
            if( C.Admissible() )
            {
                if( C._inTargetTeam ) 
                {
                    // Handle the left update, Q1' Omega2
                    const Dense<Scalar>& Q1 = *C._colXMap[A._sourceOffset];
                    const Dense<Scalar>& Omega2 = A._rowOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* leftUpdate = 
                        &allReduceBuffer[leftOffsets[teamLevel]];
                    leftOffsets[teamLevel] += r*r;

                    blas::Gemm
                    ( 'C', 'N', r, r, A.LocalHeight(),
                      (Scalar)1, Q1.LockedBuffer(),     Q1.LDim(),
                                 Omega2.LockedBuffer(), Omega2.LDim(),
                      (Scalar)0, leftUpdate,            r );
                }
                if( C._inSourceTeam )
                {
                    // Handle the right update, Q2' Omega1
                    const Dense<Scalar>& Q2 = *C._rowXMap[A._sourceOffset];
                    const Dense<Scalar>& Omega1 = B._colOmega;
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Scalar* rightUpdate = 
                        &allReduceBuffer[rightOffsets[teamLevel]];
                    rightOffsets[teamLevel] += r*r;

                    blas::Gemm
                    ( 'C', 'N', r, r, B.LocalWidth(),
                      (Scalar)1, Q2.LockedBuffer(),     Q2.LDim(),
                                 Omega1.LockedBuffer(), Omega1.LDim(),
                      (Scalar)0, rightUpdate,           r );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        nodeA.Child(t,r).
                        MultiplyHMatFHHFinalizeOuterUpdates
                        ( nodeB.Child(r,s), nodeC.Child(t,s), allReduceBuffer,
                          leftOffsets, rightOffsets );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFHHFinalizeFormLowRank
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
        std::vector<Scalar>& allReduceBuffer,
        std::vector<int>& leftOffsets,
        std::vector<int>& middleOffsets,
        std::vector<int>& rightOffsets,
        std::vector<Real>& singularValues,
        std::vector<Scalar>& U,
        std::vector<Scalar>& VH,
        std::vector<Scalar>& svdWork,
        std::vector<Real>& svdRealWork ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHFinalizeFormLowRank");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    const int r = SampleRank( C.MaxRank() );

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
            if( C.Admissible() )
            {
                if( C._inTargetTeam ) 
                {
                    // Form Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                    // in the place of X.
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    Dense<Scalar>& X = *C._colXMap[A._sourceOffset];

                    Scalar* leftUpdate = 
                        &allReduceBuffer[leftOffsets[teamLevel]];
                    const Scalar* middleUpdate = 
                        &allReduceBuffer[middleOffsets[teamLevel]];
                    leftOffsets[teamLevel] += r*r;
                    middleOffsets[teamLevel] += r*r;

                    lapack::AdjointPseudoInverse
                    ( r, r, leftUpdate, r, &singularValues[0],
                      &U[0], r, &VH[0], r, &svdWork[0], svdWork.size(),
                      &svdRealWork[0] );

                    // We can use the VH space to hold the product 
                    // pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                    blas::Gemm
                    ( 'N', 'N', r, r, r, 
                      (Scalar)1, leftUpdate,   r, 
                                 middleUpdate, r, 
                      (Scalar)0, &VH[0],       r );

                    // Q1 := X.
                    Dense<Scalar> Q1;
                    Q1 = X;

                    // Form X := Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                    blas::Gemm
                    ( 'N', 'N', Q1.Height(), r, r,
                      (Scalar)1, Q1.LockedBuffer(), Q1.LDim(),
                                 &VH[0],            r, 
                      (Scalar)0, X.Buffer(),        X.LDim() );
                }
                if( C._inSourceTeam )
                {
                    // Form Q2 pinv(Q2' Omega1) or its conjugate
                    Dense<Scalar>& X = *C._rowXMap[A._sourceOffset];
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);

                    Scalar* rightUpdate = 
                        &allReduceBuffer[rightOffsets[teamLevel]];
                    rightOffsets[teamLevel] += r*r;

                    lapack::AdjointPseudoInverse
                    ( r, r, rightUpdate, r, &singularValues[0],
                      &U[0], r, &VH[0], r, &svdWork[0], svdWork.size(),
                      &svdRealWork[0] );

                    // Q2 := X
                    Dense<Scalar> Q2;
                    Q2 = X;

                    blas::Gemm
                    ( 'N', 'C', Q2.Height(), r, r,
                      (Scalar)1, Q2.LockedBuffer(), Q2.LDim(),
                                 rightUpdate,       r,
                      (Scalar)0, X.Buffer(),        X.LDim() );
                    if( !Conjugated )
                        hmat_tools::Conjugate( X );
                }
            }
            else
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        nodeA.Child(t,r).MultiplyHMatFHHFinalizeFormLowRank
                        ( nodeB.Child(r,s), nodeC.Child(t,s), allReduceBuffer,
                          leftOffsets, middleOffsets, rightOffsets, 
                          singularValues, U, VH, svdWork, svdRealWork );
            }
            break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdates()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdates");
#endif
    const unsigned numTeamLevels = _teams->NumLevels();

    // Count the number of QRs we'll need to perform
    std::vector<int> numQRs(numTeamLevels,0);
    MultiplyHMatUpdatesCountQRs( numQRs );

    // Count the ranks of all of the low-rank updates that we will have to 
    // perform a QR on and also make space for their aggregations.
    int numTotalQRs=0;
    std::vector<int> rankOffsets(numTeamLevels);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        rankOffsets[teamLevel] = numTotalQRs;        
        numTotalQRs += numQRs[teamLevel];
    }
    std::vector<int> ranks(numTotalQRs);
    MultiplyHMatUpdatesLowRankCountAndResize( ranks, rankOffsets, 0 );

    // Carry the low-rank updates down from nodes into the low-rank and dense
    // blocks.
    MultiplyHMatUpdatesLowRankImport( 0 );

    // Allocate space for packed storage of the various components in our
    // distributed QR factorizations.
    numTotalQRs = 0;
    int qrTotalSize=0, tauTotalSize=0, maxRank=0;
    std::vector<int> XOffsets(numTeamLevels+1),
                     qrOffsets(numTeamLevels+1), 
                     tauOffsets(numTeamLevels+1);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        MPI_Comm team = _teams->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        XOffsets[teamLevel] = numTotalQRs;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        for( int i=0; i<numQRs[teamLevel]; ++i )
        {
            const int r = ranks[rankOffsets[teamLevel]+i];
            maxRank = std::max(maxRank,r);

            qrTotalSize += log2TeamSize*(r*r+r);
            tauTotalSize += (log2TeamSize+1)*r;
        }
        numTotalQRs += numQRs[teamLevel];
    }
    XOffsets[numTeamLevels] = numTotalQRs;
    qrOffsets[numTeamLevels] = qrTotalSize;
    tauOffsets[numTeamLevels] = tauTotalSize;

    std::vector<Dense<Scalar>*> Xs( numTotalQRs );
    std::vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize );
    std::vector<Scalar> work( lapack::QRWorkSize(maxRank) );

    MultiplyHMatUpdatesLocalQR
    ( Xs, XOffsets, qrBuffer, qrOffsets, tauBuffer, tauOffsets, work );
    
    // Reset the offset vectors
    numTotalQRs = qrTotalSize = tauTotalSize = 0;
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        XOffsets[teamLevel] = numTotalQRs;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;
    }

    // Perform the combined distributed TSQR factorizations.
    // This could almost certainly be simplified...
    const int numSteps = numTeamLevels-1;
    for( int step=0; step<numSteps; ++step )
    {
        MPI_Comm team = _teams->Team( step );
        const int teamSize = mpi::CommSize( team );
        const unsigned teamRank = mpi::CommRank( team );
        const bool haveAnotherComm = ( step < numSteps-1 );
        // only valid result if we have a next step...
        const bool rootOfNextStep = !(teamRank & 0x100);
        const int passes = 2*step;

        // Compute the total message size for this step
        int msgSize = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            for( int i=0; i<numQRs[l]; ++i )
            {
                const int r = ranks[rankOffsets[l]+i];     
                msgSize += (r*r+r)/2;
            }
        }
        std::vector<Scalar> sendBuffer( msgSize ), recvBuffer( msgSize );

        // Flip the first bit of our rank in this team to get our partner,
        // and then check if our bit is 0 to see if we're the root
        const unsigned firstPartner = teamRank ^ 0x1;
        const unsigned firstRoot = !(teamRank & 0x1);

        // Pack the messages for the firstPartner
        int sendOffset = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            int qrOffset = qrOffsets[l];
            MPI_Comm thisTeam = _teams->Team(l);
            const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

            for( int k=0; k<numQRs[l]; ++k )
            {
                const int r = ranks[rankOffsets[l]+k];
                if( firstRoot )
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
                else
                {
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( &sendBuffer[sendOffset],
                          &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)+(j+1)],
                          (j+1)*sizeof(Scalar) );
                        sendOffset += j+1;
                    }
                }
                qrOffset += log2ThisTeamSize*(r*r+r);
            }
        }

        // Exchange with our first partner
        mpi::SendRecv
        ( &sendBuffer[0], msgSize, firstPartner, 0,
          &recvBuffer[0], msgSize, firstPartner, 0, team );

        if( teamSize == 4 )
        {
            // Flip the second bit of our rank in this team to get our partner,
            // and then check if our bit is 0 to see if we're the root
            const unsigned secondPartner = teamRank ^ 0x10;
            const bool secondRoot = !(teamRank & 0x10);

            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step and into the next 
            // send buffer in a single sweep.
            sendOffset = 0;
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                int qrOffset = qrOffsets[l];
                int tauOffset = tauOffsets[l];
                MPI_Comm thisTeam = _teams->Team(l);
                const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

                for( int k=0; k<numQRs[l]; ++k )
                {
                    const int r = ranks[rankOffsets[l]+k];

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[qrOffset+passes*(r*r+r)], 
                      &tauBuffer[tauOffset+(passes+1)*r], &work[0] );
                    if( secondRoot )
                    {
                        // Copy into the upper triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
                    else
                    {
                        // Copy into the lower triangle of the next block
                        // and into the send buffer
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)+
                                        (j+1)],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            std::memcpy
                            ( &sendBuffer[sendOffset],
                              &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              (j+1)*sizeof(Scalar) );
                            sendOffset += j+1;
                        }
                    }
                    qrOffset += log2ThisTeamSize*(r*r+r);
                    tauOffset += (log2ThisTeamSize+1)*r;
                }
            }
            
            // Exchange with our second partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, secondPartner, 0,
              &recvBuffer[0], msgSize, secondPartner, 0, team );
            
            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step when necessary.
            recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                int qrOffset = qrOffsets[l];
                int tauOffset = tauOffsets[l];
                MPI_Comm thisTeam = _teams->Team(l);
                const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

                for( int k=0; k<numQRs[l]; ++k )
                {
                    const int r = ranks[rankOffsets[l]+k];

                    if( secondRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)+
                                        (j+1)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+(passes+1)*(r*r+r)+(j*j+j)],
                              &recvBuffer[recvOffset],
                              (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[qrOffset+(passes+1)*(r*r+r)], 
                      &tauBuffer[tauOffset+(passes+2)*r], &work[0] );
                    if( haveAnotherComm )
                    {
                        if( rootOfNextStep )
                        {
                            // Copy into the upper triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[qrOffset+(passes+2)*(r*r+r)+
                                            (j*j+j)],
                                  &qrBuffer[qrOffset+(passes+1)*(r*r+r)+
                                            (j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                        else
                        {
                            // Copy into the lower triangle of the next block
                            for( int j=0; j<r; ++j )
                                std::memcpy
                                ( &qrBuffer[qrOffset+(passes+2)*(r*r+r)+
                                            (j*j+j)+(j+1)],
                                  &qrBuffer[qrOffset+(passes+1)*(r*r+r)+
                                            (j*j+j)],
                                  (j+1)*sizeof(Scalar) );
                        }
                    }
                    qrOffset += log2ThisTeamSize*(r*r+r);
                    tauOffset += (log2ThisTeamSize+1)*r;
                }
            }
        }
        else // teamSize == 2
        {
            // Unpack the recv messages and perform the QR factorizations
            int recvOffset = 0;
            for( int l=0; l<numSteps-step; ++l )
            {
                int qrOffset = qrOffsets[l];
                int tauOffset = tauOffsets[l];
                MPI_Comm thisTeam = _teams->Team(l);
                const int log2ThisTeamSize = Log2(mpi::CommSize(thisTeam));

                for( int k=0; k<numQRs[l]; ++k )
                {
                    const int r = ranks[rankOffsets[l]+k];

                    if( firstRoot )
                    {
                        // Unpack into the bottom since our data was in the top
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)+(j+1)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    else
                    {
                        // Unpack into the top since our data was in the bottom
                        for( int j=0; j<r; ++j )
                        {
                            std::memcpy
                            ( &qrBuffer[qrOffset+passes*(r*r+r)+(j*j+j)],
                              &recvBuffer[recvOffset], (j+1)*sizeof(Scalar) );
                            recvOffset += j+1;
                        }
                    }
                    hmat_tools::PackedQR
                    ( r, &qrBuffer[qrOffset+passes*(r*r+r)], 
                      &tauBuffer[tauOffset+(passes+1)*r], &work[0] );

                    qrOffset += log2ThisTeamSize*(r*r+r);
                    tauOffset += (log2ThisTeamSize+1)*r;
                }
            }
        }
    }

    // HERE: Need to exchange the R's and send target factors to source teams
    //       when there are dense matrices involved.
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::
MultiplyHMatUpdatesCountQRs( std::vector<int>& numQRs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesCountQRs");
#endif
    const unsigned teamLevel = _teams->TeamLevel( _level );
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesCountQRs( numQRs );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
        if( _inTargetTeam && _DMap.Size() == 0 )
            ++numQRs[teamLevel];
        if( _inSourceTeam && _DMap.Size() == 0 )
            ++numQRs[teamLevel];
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::
MultiplyHMatUpdatesLowRankCountAndResize
( std::vector<int>& ranks, std::vector<int>& rankOffsets, int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLowRankCountAndResize");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }
        else if( _inSourceTeam )
        {
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _VMap.NextEntry()->Width();
        }

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankCountAndResize
                ( ranks, rankOffsets, rank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        // Compute the total update rank
        if( _inTargetTeam )
        {
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _colXMap.NextEntry()->Width();

            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }
        else if( _inSourceTeam )
        {
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _rowXMap.NextEntry()->Width();

            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _VMap.NextEntry()->Width();
        }

        // Store the rank and create the space
        const unsigned teamLevel = _teams->TeamLevel( _level );
        if( _inTargetTeam )
        {
            ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = DF.ULocal.Width();
            Dense<Scalar> ULocalCopy;
            hmat_tools::Copy( DF.ULocal, ULocalCopy );
            DF.ULocal.Resize( LocalHeight(), rank );
            std::memcpy
            ( DF.ULocal.Buffer(0,rank-oldRank), ULocalCopy.LockedBuffer(), 
              LocalHeight()*rank*sizeof(Scalar) );
        }
        if( _inSourceTeam )
        {
            ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = DF.VLocal.Width();
            Dense<Scalar> VLocalCopy;
            hmat_tools::Copy( DF.VLocal, VLocalCopy );
            DF.VLocal.Resize( LocalWidth(), rank );
            std::memcpy
            ( DF.VLocal.Buffer(0,rank-oldRank), VLocalCopy.LockedBuffer(),
              LocalWidth()*rank*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;

        const unsigned numDenseUpdates = _DMap.Size();
        const unsigned teamLevel = _teams->TeamLevel( _level );

        // Compute the total update rank
        if( _inTargetTeam )
        {
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _colXMap.NextEntry()->Width();

            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }
        else 
        {
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _rowXMap.NextEntry()->Width();

            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _VMap.NextEntry()->Width();
        }

        // Create the space and store the rank if we'll need to do a QR
        if( _inTargetTeam )
        {
            if( numDenseUpdates == 0 )
                ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = SF.D.Width();
            Dense<Scalar> UCopy;
            hmat_tools::Copy( SF.D, UCopy );
            SF.D.Resize( Height(), rank );
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              Height()*rank*sizeof(Scalar) );
        }
        else
        {
            if( numDenseUpdates == 0 )
                ranks[rankOffsets[teamLevel]++] = rank;
            const int oldRank = SF.D.Width();
            Dense<Scalar> VCopy;
            hmat_tools::Copy( SF.D, VCopy );
            SF.D.Resize( Width(), rank );
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              Width()*rank*sizeof(Scalar) );
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;

        const unsigned numDenseUpdates = _DMap.Size();
        const unsigned teamLevel = _teams->TeamLevel( _level );
        // Compute the total update rank
        {
            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _colXMap.NextEntry()->Width();

            numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
                rank += _UMap.NextEntry()->Width();
        }

        // Create the space and store the updates if there is no dense update
        {
            if( numDenseUpdates == 0 )
            {
                ranks[rankOffsets[teamLevel]++] = rank;
                ranks[rankOffsets[teamLevel]++] = rank;
            }
            const int oldRank = F.Rank();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( F.U, UCopy );
            F.U.Resize( Height(), rank );
            std::memcpy
            ( F.U.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              Height()*rank*sizeof(Scalar) );

            Dense<Scalar> VCopy;
            hmat_tools::Copy( F.V, VCopy );
            F.V.Resize( Width(), rank );
            std::memcpy
            ( F.V.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              Width()*rank*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            // Combine all of the U's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int m = Height();
            const int numLowRankUpdates = _UMap.Size();
            _UMap.ResetIterator();
            for( int update=0; update<numLowRankUpdates; ++update )
                rank += _UMap.NextEntry()->Width();

            if( numLowRankUpdates == 0 )
            {
                _UMap[0] = new Dense<Scalar>( m, rank );
            }
            else
            {
                _UMap.ResetIterator();
                Dense<Scalar> firstUCopy;
                Dense<Scalar> firstU = *_UMap.NextEntry();
                firstUCopy = firstU;
                firstU.Resize( m, rank );
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstUCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), 
                          firstUCopy.LockedBuffer(0,j), m*sizeof(Scalar) );
                }
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& U = *_UMap.NextEntry();
                    const int r = U.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), U.LockedBuffer(0,j),
                          m*sizeof(Scalar) );
                    _UMap.EraseLastEntry();
                }
            }
        }
        else
        {
            // Combine all of the U's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int n = Width();
            const int numLowRankUpdates = _VMap.Size();
            _VMap.ResetIterator();
            for( int update=0; update<numLowRankUpdates; ++update )
                rank += _VMap.NextEntry()->Width();

            if( numLowRankUpdates == 0 )
            {
                _VMap[0] = new Dense<Scalar>( n, rank );
            }
            else
            {
                _VMap.ResetIterator();
                Dense<Scalar> firstVCopy;
                Dense<Scalar> firstV = *_VMap.NextEntry();
                firstVCopy = firstV;
                firstV.Resize( n, rank );
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstVCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), 
                          firstVCopy.LockedBuffer(0,j), n*sizeof(Scalar) );
                }
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& V = *_VMap.NextEntry();
                    const int r = V.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), V.LockedBuffer(0,j),
                          n*sizeof(Scalar) );
                    _VMap.EraseLastEntry();
                }
            }
        }
        break;
    }
    case DENSE:
    {
        // Condense all of the U's and V's onto the dense matrix
        Dense<Scalar>& D = *_block.data.D;
        const int m = Height();
        const int n = Width();
        const int numLowRankUpdates = _UMap.Size();
        _UMap.ResetIterator();
        _VMap.ResetIterator();
        for( int update=0; update<numLowRankUpdates; ++update )
        {
            const Dense<Scalar>& U = *_UMap.NextEntry();
            const Dense<Scalar>& V = *_VMap.NextEntry();
            const int r = U.Width();
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( 'N', option, m, n, r,
              (Scalar)1, U.LockedBuffer(), U.LDim(),
                         V.LockedBuffer(), V.LDim(),
              (Scalar)1, D.Buffer(),       D.LDim() );
            _UMap.EraseLastEntry();
            _VMap.EraseLastEntry();
        }

        const int numDenseUpdates = _DMap.Size();
        _DMap.ResetIterator();
        for( int update=0; update<numDenseUpdates; ++update )
        {
            const Dense<Scalar>& DUpdate = *_DMap.NextEntry();
            for( int j=0; j<n; ++j )
            {
                const Scalar* DUpdateCol = DUpdate.LockedBuffer(0,j);
                Scalar* DCol = D.Buffer(0,j);
                for( int i=0; i<m; ++i )
                    DCol[i] += DUpdateCol[i];
            }
            _DMap.EraseLastEntry();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesLowRankImport
( int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLowRankImport");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& U = *_UMap.NextEntry();
                Dense<Scalar> ULocal; 

                for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                {
                    for( int s=0; s<4; ++s )
                    {
                        ULocal.LockedView
                        ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                        node.Child(t,s).MultiplyHMatUpdatesImportU
                        ( newRank, ULocal );
                    }
                }
                newRank += U.Width();
                _UMap.EraseLastEntry();
            }
        }
        if( _inSourceTeam )
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& V = *_VMap.NextEntry();
                Dense<Scalar> VLocal;

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    for( int t=0; t<4; ++t )
                    {
                        VLocal.LockedView
                        ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                        node.Child(t,s).MultiplyHMatUpdatesImportV
                        ( newRank, VLocal );
                    }
                }
                newRank += V.Width();
                _VMap.EraseLastEntry();
            }
        }

        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLowRankImport( newRank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF; 
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& U = *_UMap.NextEntry();
                const int m = U.Height();
                const int r = U.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( DF.ULocal.Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                _UMap.EraseLastEntry();
                newRank += r;
            }
        }
        if( _inSourceTeam )
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& V = *_VMap.NextEntry();
                const int n = V.Height();
                const int r = V.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( DF.VLocal.Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                _VMap.EraseLastEntry();
                newRank += r;
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF; 
        int newRank = rank;
        if( _inTargetTeam )
        {
            const int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& U = *_UMap.NextEntry();
                const int m = U.Height();
                const int r = U.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( SF.D.Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                _UMap.EraseLastEntry();
                newRank += r;
            }
        }
        else
        {
            newRank = rank;
            const int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& V = *_VMap.NextEntry();
                const int n = V.Height();
                const int r = V.Width();
                for( int j=0; j<r; ++j )
                    std::memcpy
                    ( SF.D.Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                _VMap.EraseLastEntry();
                newRank += r;
            }
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        int newRank = rank;
        const int numEntries = _UMap.Size();
        _UMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            Dense<Scalar>& U = *_UMap.NextEntry();
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.U.Buffer(0,newRank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
            _UMap.EraseLastEntry();
            newRank += r;
        }
        newRank = rank;
        _VMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            Dense<Scalar>& V = *_VMap.NextEntry();
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.V.Buffer(0,newRank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
            _VMap.EraseLastEntry();
            newRank += r;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesImportU
( int rank, const Dense<Scalar>& U )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesImportU");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> ULocal;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0; s<4; ++s )
            {
                ULocal.LockedView
                ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                node.Child(t,s).MultiplyHMatUpdatesImportU( rank, ULocal );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inTargetTeam )
        {
            DistLowRank& DF = *_block.data.DF;
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( DF.ULocal.Buffer(0,rank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 && _inTargetTeam )
        {
            SplitLowRank& SF = *_block.data.SF;
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( SF.D.Buffer(0,rank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
        }
        break;
    }
    case LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            LowRank<Scalar,Conjugated>& F = *_block.data.F;
            const int m = U.Height();
            const int r = U.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.U.Buffer(0,rank+j), U.LockedBuffer(0,j),
                  m*sizeof(Scalar) );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesImportV
( int rank, const Dense<Scalar>& V )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesImportV");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> VLocal;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            for( int t=0; t<4; ++t )
            {
                VLocal.LockedView
                ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                node.Child(t,s).MultiplyHMatUpdatesImportV( rank, VLocal );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam )
        {
            DistLowRank& DF = *_block.data.DF;
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( DF.VLocal.Buffer(0,rank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 && _inSourceTeam )
        {
            SplitLowRank& SF = *_block.data.SF;
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( SF.D.Buffer(0,rank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
        }
        break;
    }
    case LOW_RANK:
    {
        const unsigned numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            LowRank<Scalar,Conjugated>& F = *_block.data.F;
            const int n = V.Height();
            const int r = V.Width();
            for( int j=0; j<r; ++j )
                std::memcpy
                ( F.V.Buffer(0,rank+j), V.LockedBuffer(0,j),
                  n*sizeof(Scalar) );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatUpdatesLocalQR
( std::vector<Dense<Scalar>*>& Xs, std::vector<int>& XOffsets,
  std::vector<Scalar>& qrBuffer,  std::vector<int>& qrOffsets,
  std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
  std::vector<Scalar>& work )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatUpdatesLocalQR");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatUpdatesLocalQR
                ( Xs, XOffsets, qrBuffer, qrOffsets, tauBuffer, tauOffsets, 
                  work );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        const int teamSize = mpi::CommSize( team );
        const int log2TeamSize = Log2( teamSize );
        const unsigned teamLevel = _teams->TeamLevel(_level);

        if( _inTargetTeam )
        {
            Dense<Scalar>& ULocal = DF.ULocal;
            const int m = ULocal.Height();
            const int r = ULocal.Width();

            Xs[XOffsets[teamLevel]++] = &ULocal;
            lapack::QR
            ( m, r, ULocal.Buffer(), ULocal.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;
            if( log2TeamSize > 0 )
            {
                std::memset
                ( &qrBuffer[qrOffsets[teamLevel]], 0, (r*r+r)*sizeof(Scalar) );
                const bool root = !(teamRank & 0x1);
                if( root )
                {
                    // Copy our R into the upper triangle of the next
                    // matrix to factor (which is 2r x r)
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                          ULocal.LockedBuffer(0,j), 
                          std::min(m,j+1)*sizeof(Scalar) );
                }
                else
                {
                    // Copy our R into the lower triangle of the next
                    // matrix to factor (which is 2r x r)
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                          ULocal.LockedBuffer(0,j), 
                          std::min(m,j+1)*sizeof(Scalar) );
                }
            }
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
        }
        if( _inSourceTeam )
        {
            Dense<Scalar>& VLocal = DF.VLocal;
            const int n = VLocal.Height();
            const int r = VLocal.Width();

            Xs[XOffsets[teamLevel]++] = &VLocal;
            lapack::QR
            ( n, r, VLocal.Buffer(), VLocal.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
            tauOffsets[teamLevel] += (log2TeamSize+1)*r;
            if( log2TeamSize > 0 )
            {
                std::memset
                ( &qrBuffer[qrOffsets[teamLevel]], 0, (r*r+r)*sizeof(Scalar) );
                const bool root = !(teamRank & 0x1);
                if( root )
                {
                    // Copy our R into the upper triangle of the next
                    // matrix to factor (which is 2r x r)
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)],
                          VLocal.LockedBuffer(0,j), 
                          std::min(n,j+1)*sizeof(Scalar) );
                }
                else
                {
                    // Copy our R into the lower triangle of the next
                    // matrix to factor (which is 2r x r)
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( &qrBuffer[qrOffsets[teamLevel]+(j*j+j)+(j+1)],
                          VLocal.LockedBuffer(0,j), 
                          std::min(n,j+1)*sizeof(Scalar) );
                }
            }
            qrOffsets[teamLevel] += log2TeamSize*(r*r+r);
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;

        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            if( _inTargetTeam )
            {
                Dense<Scalar>& U = SF.D;
                const int m = U.Height();
                const int r = U.Width();

                Xs[XOffsets[teamLevel]++] = &U;
                lapack::QR
                ( m, r, U.Buffer(), U.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
                tauOffsets[teamLevel] += r;
            }
            else
            {
                Dense<Scalar>& V = SF.D;
                const int n = V.Height();
                const int r = V.Width();

                Xs[XOffsets[teamLevel]++] = &V;
                lapack::QR
                ( n, r, V.Buffer(), V.LDim(), 
                  &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
                tauOffsets[teamLevel] += r;
            }
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int numDenseUpdates = _DMap.Size();
        if( numDenseUpdates == 0 )
        {
            const unsigned teamLevel = _teams->TeamLevel(_level);
            Dense<Scalar>& U = F.U;
            Dense<Scalar>& V = F.V;
            const int m = U.Height();
            const int n = V.Height();
            const int r = U.Width();

            Xs[XOffsets[teamLevel]++] = &U;
            lapack::QR
            ( m, r, U.Buffer(), U.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
            tauOffsets[teamLevel] += r;

            Xs[XOffsets[teamLevel]++] = &V;
            lapack::QR
            ( n, r, V.Buffer(), V.LDim(), 
              &tauBuffer[tauOffsets[teamLevel]], &work[0], work.size() );
            tauOffsets[teamLevel] += r;
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

