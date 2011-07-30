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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSetUp
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSetUp");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    C._numLevels = A._numLevels;
    C._maxRank = A._maxRank;
    C._targetOffset = A._targetOffset;
    C._sourceOffset = B._sourceOffset;
    C._stronglyAdmissible = ( A._stronglyAdmissible || B._stronglyAdmissible );

    C._xSizeTarget = A._xSizeTarget;
    C._ySizeTarget = A._ySizeTarget;
    C._xSizeSource = B._xSizeSource;
    C._ySizeSource = B._ySizeSource;
    C._zSize = A._zSize;

    C._xTarget = A._xTarget;
    C._yTarget = A._yTarget;
    C._xSource = B._xSource;
    C._ySource = B._ySource;

    C._teams = A._teams;
    C._level = A._level;
    C._inTargetTeam = A._inTargetTeam;
    C._inSourceTeam = B._inSourceTeam;
    C._targetRoot = A._targetRoot;
    C._sourceRoot = B._sourceRoot;
    
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
        if( C._sourceRoot == C._targetRoot )
        {
            if( C._inSourceTeam || C._inTargetTeam )
            {
                C._block.type = DENSE;
                C._block.data.D = new Dense<Scalar>( A.Height(), B.Width() );
                hmat_tools::Scale( (Scalar)0, *C._block.data.D );
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
                if( C._inSourceTeam )
                {
                    C._block.data.SD->D.Resize( A.Height(), B.Width() );
                    hmat_tools::Scale( (Scalar)0, C._block.data.SD->D );
                }
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
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPrecompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( !A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam )
    {
        C._block.type = EMPTY;
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    if( C._block.type == EMPTY )
        A.MultiplyHMatMainSetUp( B, C );

    if( A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Handle all H H cases here
    const bool admissibleC = C.Admissible();
    const int key = A._sourceOffset;
    const int sampleRank = SampleRank( C.MaxRank() );
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
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
#ifndef RELEASE
                PopCallStack();
#endif
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
    else if( A._level >= startLevel && A._level < endLevel && 
             update >= startUpdate && update < endUpdate )
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
                    A._rowOmega.Resize( A.LocalHeight(), sampleRank );
                    ParallelGaussianRandomVectors( A._rowOmega );

                    A._rowT.Resize( A.LocalWidth(), sampleRank );
                    hmat_tools::Scale( (Scalar)0, A._rowT );

                    A.AdjointMultiplyDenseInitialize
                    ( A._rowContext, sampleRank );

                    A.AdjointMultiplyDensePrecompute
                    ( A._rowContext, (Scalar)1, A._rowOmega, A._rowT );

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
                {
                    B._colOmega.Resize( B.LocalWidth(), sampleRank ); 
                    ParallelGaussianRandomVectors( B._colOmega );

                    B._colT.Resize( B.LocalHeight(), sampleRank );
                    hmat_tools::Scale( (Scalar)0, B._colT );

                    B.MultiplyDenseInitialize( B._colContext, sampleRank );
                    B.MultiplyDensePrecompute
                    ( B._colContext, (Scalar)1, B._colOmega, B._colT );

                    B._beganColSpaceComp = true;
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
    }

    if( A._level < startLevel || A._level >= endLevel || 
        update < startUpdate  || update >= endUpdate )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
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
            C._UMap.Set( key, new Dense<Scalar>(C.LocalHeight(),DFB.rank) );
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get(key);

            hmat_tools::Scale( (Scalar)0, C._UMap.Get(key) );
            A.MultiplyDenseInitialize( context, DFB.rank );
            A.MultiplyDensePrecompute
            ( context, alpha, DFB.ULocal, C._UMap.Get(key) );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We must be in the left team
            const DistLowRankGhost& DFGB = *B._block.data.DFG;
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

            A.MultiplyDenseInitialize( context, DFGB.rank );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                C._mainContextMap.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C._mainContextMap.Get( key );

                A.MultiplyDenseInitialize( context, SFB.rank );
            }
            else
            {
                // We are the middle process
                C._mainContextMap.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C._mainContextMap.Get( key );

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
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

            A.MultiplyDenseInitialize( context, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We are the middle and right processes
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

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
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

            A.MultiplyDenseInitialize( context, FGB.rank );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._UMap.Set( key, new Dense<Scalar>( C.Height(), SFB.rank ) );
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

            hmat_tools::Scale( (Scalar)0, C._UMap.Get( key ) );
            A.MultiplyDenseInitialize( context, SFB.rank );
            A.MultiplyDensePrecompute
            ( context, alpha, SFB.D, C._UMap.Get( key ) );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._UMap.Set( key, new Dense<Scalar>( C.Height(), FB.Rank() ) );
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

            hmat_tools::Scale( (Scalar)0, C._UMap.Get( key ) );
            A.MultiplyDenseInitialize( context, FB.Rank() );
            A.MultiplyDensePrecompute
            ( context, alpha, FB.U, C._UMap.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._VMap.Set( key, new Dense<Scalar>( C.LocalWidth(), DFA.rank ) );
            hmat_tools::Scale( (Scalar)0, C._VMap.Get( key ) );
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
            if( Conjugated )
            {
                B.AdjointMultiplyDenseInitialize( context, DFA.rank );
                B.AdjointMultiplyDensePrecompute
                ( context, Conj(alpha), DFA.VLocal, C._VMap.Get( key ) );
            }
            else
            {
                B.TransposeMultiplyDenseInitialize( context, DFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( context, alpha, DFA.VLocal, C._VMap.Get( key ) );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            // Start H/F += F F
            if( A._inSourceTeam )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                const int kLocal = A.LocalWidth();
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    C._ZMap.Set( key, new Dense<Scalar>( DFA.rank, DFB.rank ) );
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    const char option = ( Conjugated ? 'C' : 'T' );
                    blas::Gemm
                    ( option, 'N', DFA.rank, DFB.rank, kLocal,
                      (Scalar)1, DFA.VLocal.LockedBuffer(), DFA.VLocal.LDim(),
                                 DFB.ULocal.LockedBuffer(), DFB.ULocal.LDim(),
                      (Scalar)0, ZC.Buffer(),               ZC.LDim() );
                }
            }
            break;
        }
        case DIST_NODE_GHOST:
        case DIST_LOW_RANK_GHOST:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
            if( Conjugated )
                B.AdjointMultiplyDenseInitialize( context, DFGA.rank );
            else
                B.TransposeMultiplyDenseInitialize( context, DFGA.rank );
            break;
        }
        case DIST_LOW_RANK:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                C._mainContextMap.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C._mainContextMap.Get( key );
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
                C._mainContextMap.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C._mainContextMap.Get( key );
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
            C._VMap.Set( key, new Dense<Scalar>( B.Width(), SFA.rank ) );
            Dense<Scalar>& CV = C._VMap.Get( key );
                
            hmat_tools::Scale( (Scalar)0, CV );
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
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
                const int k = A.Width();
                if( SFA.rank != 0 && SFB.rank != 0 )
                {
                    C._ZMap.Set( key, new Dense<Scalar>( SFA.rank, SFB.rank ) );
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    const char option = ( Conjugated ? 'C' : 'T' );
                    blas::Gemm
                    ( option, 'N', SFA.rank, SFB.rank, k,
                      (Scalar)1, SFA.D.LockedBuffer(), SFA.D.LDim(),
                                 SFB.D.LockedBuffer(), SFB.D.LDim(),
                      (Scalar)0, ZC.Buffer(),          ZC.LDim() );
                }
            }
            break;
        }
        case LOW_RANK:
        {
            // We must be the middle and right process
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            const int k = A.Width();
            if( SFA.rank != 0 && FB.Rank() != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( SFA.rank, FB.Rank() ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( option, 'N', SFA.rank, FB.Rank(), k,
                  (Scalar)1, SFA.D.LockedBuffer(), SFA.D.LDim(),
                             FB.U.LockedBuffer(),  FB.U.LDim(),
                  (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case DENSE:
        {
            // We must be both the middle and right process
            const int k = B.Height();
            const int n = B.Width();
            C._VMap.Set( key, new Dense<Scalar>( n, SFA.rank ) );
            const Dense<Scalar>& DB = *B._block.data.D;
            Dense<Scalar>& VC = C._VMap.Get( key );
            const char option = ( Conjugated ? 'C' : 'T' );
            const Scalar scale = ( Conjugated ? Conj(alpha) : alpha );
            blas::Gemm
            ( option, 'N', n, SFA.rank, k,
              scale,     DB.LockedBuffer(),    DB.LDim(),
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
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
            C._VMap.Set( key, new Dense<Scalar>( B.Width(), FA.Rank() ) );
            hmat_tools::Scale( (Scalar)0, C._VMap.Get( key ) );
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
            if( Conjugated )
            {
                B.AdjointMultiplyDenseInitialize( context, FA.Rank() );
                B.AdjointMultiplyDensePrecompute
                ( context, Conj(alpha), FA.V, C._VMap.Get( key ) );
            }
            else
            {
                B.TransposeMultiplyDenseInitialize( context, FA.Rank() );
                B.TransposeMultiplyDensePrecompute
                ( context, alpha, FA.V, C._VMap.Get( key ) );
            }
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We must be the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            const int k = B.Height();
            if( FA.Rank() != 0 && SFB.rank != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( FA.Rank(), SFB.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( option, 'N', FA.Rank(), SFB.rank, k,
                  (Scalar)1, FA.V.LockedBuffer(),  FA.V.LDim(),
                             SFB.D.LockedBuffer(), SFB.D.LDim(),
                  (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case LOW_RANK:
        {
            // We must own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            const int k = B.Height();
            if( FA.Rank() != 0 && FB.Rank() != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( FA.Rank(), FB.Rank() ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                const char option = ( Conjugated ? 'C' : 'T' );
                blas::Gemm
                ( option, 'N', FA.Rank(), FB.Rank(), k,
                  (Scalar)1, FA.V.LockedBuffer(), FA.V.LDim(),
                             FB.U.LockedBuffer(), FB.U.LDim(),
                  (Scalar)0, ZC.Buffer(),         ZC.LDim() );
            }
            break;
        }
        case SPLIT_DENSE:
            // We must be the left and middle process, but there is no
            // work to be done (split dense owned by right process)
            break;
        case DENSE:
        {
            // We must own all of A, B, and C
            const int k = B.Height();
            const int n = B.Width();
            C._VMap.Set( key, new Dense<Scalar>( n, FA.Rank() ) );
            const Dense<Scalar>& DB = *B._block.data.D;
            Dense<Scalar>& VC = C._VMap.Get( key );
            const char option = ( Conjugated ? 'C' : 'T' );
            const Scalar scale = ( Conjugated ? Conj(alpha) : alpha );
            blas::Gemm
            ( option, 'N', n, FA.Rank(), k,
              scale,     DB.LockedBuffer(),   DB.LDim(),
                         FA.V.LockedBuffer(), FA.V.LDim(),
              (Scalar)0, VC.Buffer(),         VC.LDim() );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._mainContextMap.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                const int m = A.Height();
                const int k = A.Width();
                if( m != 0 && SFB.rank != 0 )
                {
                    C._ZMap.Set( key, new Dense<Scalar>( m, SFB.rank ) );
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    blas::Gemm
                    ( 'N', 'N', m, SFB.rank, k,
                      alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                                 SFB.D.LockedBuffer(), SFB.D.LDim(),
                      (Scalar)0, ZC.Buffer(),          ZC.LDim() );
                }
            }
            break;
        }
        case LOW_RANK:
        {
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            const int m = A.Height();
            const int k = A.Width();
            if( m != 0 && FB.Rank() != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( m, FB.Rank() ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', m, FB.Rank(), k,
                  alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                             FB.U.LockedBuffer(),  FB.U.LDim(),
                  (Scalar)0, ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case DENSE:
        {
            const Dense<Scalar>& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                if( C._storedDenseUpdate )
                    hmat_tools::Multiply( alpha, SDA.D, DB, (Scalar)1, C._D );
                else
                {
                    hmat_tools::Multiply( alpha, SDA.D, DB, C._D );
                    C._haveDenseUpdate = true;
                    C._storedDenseUpdate = true;
                }
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
            break;
        case SPLIT_DENSE:
            if( admissibleC && C._inSourceTeam )
                C._haveDenseUpdate = true;
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE_GHOST:
            if( admissibleC )
                C._haveDenseUpdate = true;
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
        if( B._block.type == SPLIT_DENSE && admissibleC )
            C._haveDenseUpdate = true;
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
            const int m = A.Height();
            const int k = A.Width();
            C._UMap.Set( key, new Dense<Scalar>( m, SFB.rank ) );
            Dense<Scalar>& UC = C._UMap.Get( key );
            blas::Gemm
            ( 'N', 'N', m, SFB.rank, k,
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
                LowRank<Scalar,Conjugated> temp;
                hmat_tools::Multiply( alpha, DA, FB, temp );
                hmat_tools::RoundedUpdate
                ( C.MaxRank(), (Scalar)1, temp, (Scalar)1, FC );
            }
            else
            {
                Dense<Scalar>& DC = *C._block.data.D;
                hmat_tools::Multiply( alpha, DA, FB, (Scalar)1, DC );
            }
            break;
        }
        case SPLIT_DENSE:
            if( admissibleC )
                C._haveDenseUpdate = true;
            break;
        case DENSE:
        {
            const Dense<Scalar>& DB = *B._block.data.D;
            if( admissibleC )
            {
                // F += D D
                if( C._storedDenseUpdate )
                    hmat_tools::Multiply( alpha, DA, DB, (Scalar)1, C._D );
                else
                {
                    hmat_tools::Multiply( alpha, DA, DB, C._D );
                    C._haveDenseUpdate = true;
                    C._storedDenseUpdate = true;
                }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE_GHOST:
        if( B._block.type == SPLIT_DENSE && admissibleC )
            C._haveDenseUpdate = true;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSums
( DistQuasi2dHMat<Scalar,Conjugated>& B, DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
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
    A.MultiplyHMatMainSumsCountA( sizes, startLevel, endLevel );
    B.MultiplyHMatMainSumsCountB( sizes, startLevel, endLevel );
    A.MultiplyHMatMainSumsCountC
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatMainSumsPackA( buffer, offsetsCopy, startLevel, endLevel );
    B.MultiplyHMatMainSumsPackB( buffer, offsetsCopy, startLevel, endLevel );
    A.MultiplyHMatMainSumsPackC
    ( B, C, buffer, offsetsCopy, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Perform the reduces with log2(p) messages
    A._teams->TreeSumToRoots( buffer, sizes );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatMainSumsUnpackA( buffer, offsets, startLevel, endLevel );
    B.MultiplyHMatMainSumsUnpackB( buffer, offsets, startLevel, endLevel );
    A.MultiplyHMatMainSumsUnpackC
    ( B, C, buffer, offsets, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsCountA
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDenseSumsCount( sizes, _rowContext.numRhs );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsCountA
                    ( sizes, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel && 
            _beganRowSpaceComp && !_finishedRowSpaceComp )   
            TransposeMultiplyDenseSumsPack( _rowContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsPackA
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDenseSumsUnpack( _rowContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsUnpackA
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsCountB
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDenseSumsCount( sizes, _colContext.numRhs );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsCountB
                    ( sizes, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDenseSumsPack( _colContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsPackB
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainSumsUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDenseSumsUnpack( _colContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsUnpackB
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainSumsCountC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<int>& sizes, 
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
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
        {
            if( !admissibleC && A._level+1 < endLevel )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                const Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSumsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                A.MultiplyDenseSumsCount( sizes, DFB.rank );
            }
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
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                B.TransposeMultiplyDenseSumsCount( sizes, DFA.rank );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A._inSourceTeam )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRank& DFB = *B._block.data.DF;
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
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
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
        {
            if( !admissibleC && A._level+1 < endLevel )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSumsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseSumsPack
                ( C._mainContextMap.Get( key ), buffer, offsets );
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
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseSumsPack
                ( C._mainContextMap.Get( key ), buffer, offsets );
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A._inSourceTeam )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRank& DFB = *B._block.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A._teams->TeamLevel(A._level);
                    std::memcpy
                    ( &buffer[offsets[teamLevel]], 
                      C._ZMap.Get( key ).LockedBuffer(),
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
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
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const 
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
        {
            if( !admissibleC && A._level+1 < endLevel )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSumsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseSumsUnpack
                ( C._mainContextMap.Get( key ), buffer, offsets );
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
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseSumsUnpack
                ( C._mainContextMap.Get( key ), buffer, offsets );
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A._inSourceTeam )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRank& DFB = *B._block.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A._teams->TeamLevel(A._level);
                    std::memcpy
                    ( C._ZMap.Get( key ).Buffer(), &buffer[offsets[teamLevel]],
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
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
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel, int startUpdate, int endUpdate )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassData");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatMainPassDataCountA
    ( sendSizes, recvSizes, startLevel, endLevel );
    B.MultiplyHMatMainPassDataCountB
    ( sendSizes, recvSizes, startLevel, endLevel );
    A.MultiplyHMatMainPassDataCountC
    ( B, C, sendSizes, recvSizes, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Compute the offsets
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
    std::vector<Scalar> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatMainPassDataPackA
    ( sendBuffer, offsets, startLevel, endLevel );
    B.MultiplyHMatMainPassDataPackB
    ( sendBuffer, offsets, startLevel, endLevel );
    A.MultiplyHMatMainPassDataPackC
    ( B, C, sendBuffer, offsets, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Start the non-blocking sends
    MPI_Comm comm = _teams->Team( 0 );
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
    A.MultiplyHMatMainPassDataUnpackA
    ( recvBuffer, recvOffsets, startLevel, endLevel );
    B.MultiplyHMatMainPassDataUnpackB
    ( recvBuffer, recvOffsets, startLevel, endLevel );
    A.MultiplyHMatMainPassDataUnpackC
    ( B, C, recvBuffer, recvOffsets, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

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
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, _rowContext.numRhs );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataCountA
                    ( sendSizes, recvSizes, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataPackA
( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDensePassDataPack
            ( _rowContext, _rowOmega, sendBuffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataPackA
                    ( sendBuffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataUnpackA
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDensePassDataUnpack
            ( _rowContext, recvBuffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataUnpackA
                    ( recvBuffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataCountB
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDensePassDataCount
            ( sendSizes, recvSizes, _colContext.numRhs );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataCountB
                    ( sendSizes, recvSizes, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataPackB
( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDensePassDataPack( _colContext, sendBuffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataPackB
                    ( sendBuffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataUnpackB
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDensePassDataUnpack( _colContext, recvBuffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataUnpackB
                    ( recvBuffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataCountC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
  const DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataCountC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

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
                if( A._level+1 < endLevel )
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
                                  sendSizes, recvSizes, startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
#ifndef RELEASE
                PopCallStack();
#endif
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

    if( A._level < startLevel || A._level >= endLevel ||
        update < startUpdate  || update >= endUpdate )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
            AddToMap( recvSizes, B._targetRoot, A.Height()*A.Width() );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE_GHOST:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
            AddToMap( recvSizes, B._targetRoot, A.Height()*A.Width() );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPassDataPackC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
        DistQuasi2dHMat<Scalar,Conjugated>& C,
  std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataPackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

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
                if( A._level+1 < endLevel )
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
                                  sendBuffer, offsets, startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
#ifndef RELEASE
                PopCallStack();
#endif
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

    if( A._level < startLevel || A._level >= endLevel ||
        update < startUpdate  || update >= endUpdate )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
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
            ( C._mainContextMap.Get( key ), sendBuffer, offsets );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), sendBuffer, offsets );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                ( C._mainContextMap.Get( key ), 
                  DFA.VLocal, sendBuffer, offsets );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( teamRank == 0 && (A._inSourceTeam != A._inTargetTeam) )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                if( A._inSourceTeam && DFA.rank != 0 && DFB.rank != 0  )
                {
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    std::memcpy
                    ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[A._targetRoot] += DFA.rank*DFB.rank;
                    ZC.Clear();
                    C._ZMap.Erase( key );
                }
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F is only receiving here
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                ( C._mainContextMap.Get( key ), SFA.D, sendBuffer, offsets );
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
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                  SFA.rank*SFB.rank*sizeof(Scalar) );
                offsets[A._targetRoot] += SFA.rank*SFB.rank;
                ZC.Clear();
                C._ZMap.Erase( key );
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
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                  SFA.rank*FB.Rank()*sizeof(Scalar) );
                offsets[A._targetRoot] += SFA.rank*FB.Rank();
                ZC.Clear();
                C._ZMap.Erase( key );
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), FA.V, sendBuffer, offsets );
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(),
                  A.Height()*SFB.rank*sizeof(Scalar) );
                offsets[A._targetRoot] += A.Height()*SFB.rank;
                ZC.Clear();
                C._ZMap.Erase( key );
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
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                std::memcpy
                ( &sendBuffer[offsets[A._targetRoot]], ZC.LockedBuffer(), 
                  A.Height()*FB.Rank()*sizeof(Scalar) );
                offsets[A._targetRoot] += A.Height()*FB.Rank();
                ZC.Clear();
                C._ZMap.Erase( key );
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
  const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPassDataUnpackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

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
                if( A._level+1 < endLevel )
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
                                  recvBuffer, offsets, startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
#ifndef RELEASE
                PopCallStack();
#endif
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

    if( A._level < startLevel || A._level >= endLevel ||
        update < startUpdate  || update >= endUpdate )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( A._inTargetTeam && !A._inSourceTeam )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    C._ZMap.Set( key, new Dense<Scalar>( DFA.rank, DFB.rank ) );
                    if( teamRank == 0 )
                    {
                        Dense<Scalar>& ZC = C._ZMap.Get( key ); 
                        std::memcpy
                        ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                          DFA.rank*DFB.rank*sizeof(Scalar) );
                        offsets[A._sourceRoot] += DFA.rank*DFB.rank;
                    }
                }
            }
            break;
        case DIST_LOW_RANK_GHOST:
        {
            // Pass data for H/F += F F
            const DistLowRankGhost& DFGB = *B._block.data.DFG;
            if( DFA.rank != 0 && DFGB.rank != 0 )
            {
                C._ZMap.Set
                ( key, new Dense<Scalar>( DFA.rank, DFGB.rank ) );
                if( teamRank == 0 )
                {
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    std::memcpy
                    ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                      DFA.rank*DFGB.rank*sizeof(Scalar) );
                    offsets[A._sourceRoot] += DFA.rank*DFGB.rank;
                }
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
            break;
        case DIST_LOW_RANK:
            // Pass data for for H/F += F F is between other (two) team(s)
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
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
                C._ZMap.Set( key, new Dense<Scalar>( SFA.rank, SFB.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

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
                C._ZMap.Set( key, new Dense<Scalar>( SFA.rank, SFGB.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

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
                C._ZMap.Set( key, new Dense<Scalar>( SFA.rank, FGB.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

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
                C._ZMap.Set( key, new Dense<Scalar>( B.Height(), SFA.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is between other two processes
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            if( B._inSourceTeam && B.Height() != 0 && SFGA.rank != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( B.Height(), SFGA.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  B.Height()*SFGA.rank*sizeof(Scalar) );
                offsets[B._targetRoot] += B.Height()*SFGA.rank;
            }
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            ( C._mainContextMap.Get( key ), recvBuffer, offsets );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is in other process
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B.Height() != 0 && FGA.rank != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( B.Height(), FGA.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[B._targetRoot]],
                  B.Height()*FGA.rank*sizeof(Scalar) );
                offsets[B._targetRoot] += B.Height()*FGA.rank;
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                    C._ZMap.Set
                    ( key, new Dense<Scalar>( A.Height(), SFB.rank ) );
                    Dense<Scalar>& ZC = C._ZMap.Get( key );

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
                C._ZMap.Set( key, new Dense<Scalar>( A.Height(), SFGB.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

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
                C._ZMap.Set( key, new Dense<Scalar>( A.Height(), FGB.rank ) );
                Dense<Scalar>& ZC = C._ZMap.Get( key );

                std::memcpy
                ( ZC.Buffer(), &recvBuffer[offsets[A._sourceRoot]],
                  A.Height()*FGB.rank*sizeof(Scalar) );
                offsets[A._sourceRoot] += A.Height()*FGB.rank;
            }
            break;
        }
        case SPLIT_DENSE:
        {
            // Pass data for D/F += D D
            const int m = A.Height();    
            const int k = A.Width();
            if( m != 0 && k != 0 )
            {
                if( C._inSourceTeam )
                {
                    C._ZMap.Set( key, new Dense<Scalar>( m, k ) );
                    std::memcpy
                    ( C._ZMap.Get( key ).Buffer(), 
                      &recvBuffer[offsets[B._targetRoot]],
                      m*k*sizeof(Scalar) );
                    offsets[B._targetRoot] += m*k;
                }
            }
            break;
        }
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            const int m = A.Height();
            const int k = A.Width();
            if( m != 0 && k != 0 )
            {
                C._ZMap.Set( key, new Dense<Scalar>( m, k ) );
                std::memcpy
                ( C._ZMap.Get( key ).Buffer(), 
                  &recvBuffer[offsets[B._targetRoot]],
                  m*k*sizeof(Scalar) );
                offsets[B._targetRoot] += m*k; 
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
  DistQuasi2dHMat<Scalar,Conjugated>& C, 
  int startLevel, int endLevel, int startUpdate, int endUpdate )
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
    A.MultiplyHMatMainBroadcastsCountA( sizes, startLevel, endLevel );
    B.MultiplyHMatMainBroadcastsCountB( sizes, startLevel, endLevel );
    A.MultiplyHMatMainBroadcastsCountC
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( unsigned i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatMainBroadcastsPackA
    ( buffer, offsetsCopy, startLevel, endLevel );
    B.MultiplyHMatMainBroadcastsPackB
    ( buffer, offsetsCopy, startLevel, endLevel );
    A.MultiplyHMatMainBroadcastsPackC
    ( B, C, buffer, offsetsCopy, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Perform the broadcasts with log2(p) messages
    A._teams->TreeBroadcasts( buffer, sizes );

    // Unpack the broadcasted buffers
    A.MultiplyHMatMainBroadcastsUnpackA
    ( buffer, offsets, startLevel, endLevel );
    B.MultiplyHMatMainBroadcastsUnpackB
    ( buffer, offsets, startLevel, endLevel );
    A.MultiplyHMatMainBroadcastsUnpackC
    ( B, C, buffer, offsets, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountA
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDenseBroadcastsCount( sizes, _rowContext.numRhs );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsCountA
                    ( sizes, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsPackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDenseBroadcastsPack
            ( _rowContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsPackA
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsUnpackA");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganRowSpaceComp && !_finishedRowSpaceComp )
            TransposeMultiplyDenseBroadcastsUnpack
            ( _rowContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsUnpackA
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountB
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDenseBroadcastsCount( sizes, _colContext.numRhs );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsCountB
                    ( sizes, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsPackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDenseBroadcastsPack( _colContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            const Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsPackB
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsUnpackB");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( _level >= startLevel && _level < endLevel &&
            _beganColSpaceComp && !_finishedColSpaceComp )
            MultiplyDenseBroadcastsUnpack( _colContext, buffer, offsets );

        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsUnpackB
                    ( buffer, offsets, startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainBroadcastsCountC
( const DistQuasi2dHMat<Scalar,Conjugated>& B,
  const DistQuasi2dHMat<Scalar,Conjugated>& C, 
  std::vector<int>& sizes, 
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsCountC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A._level+1 < endLevel )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                const Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFB = *B._block.data.DF;
                A.MultiplyDenseBroadcastsCount( sizes, DFB.rank );
            }
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
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                B.TransposeMultiplyDenseBroadcastsCount( sizes, DFA.rank );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A._inTargetTeam )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRank& DFB = *B._block.data.DF;
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                sizes[teamLevel] += DFA.rank*DFB.rank;
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                const unsigned teamLevel = A._teams->TeamLevel(A._level);
                sizes[teamLevel] += DFA.rank*DFGB.rank;
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
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsPackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A._level+1 < endLevel )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseBroadcastsPack
                ( C._mainContextMap.Get( key ), buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseBroadcastsPack
                ( C._mainContextMap.Get( key ), buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A._inTargetTeam )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRank& DFB = *B._block.data.DF;
                MPI_Comm team = _teams->Team( _level );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 && DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A._teams->TeamLevel(A._level);
                    std::memcpy
                    ( &buffer[offsets[teamLevel]], 
                      C._ZMap.Get( key ).LockedBuffer(),
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                MPI_Comm team = _teams->Team( _level );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 && DFA.rank != 0 && DFGB.rank != 0 )
                {
                    const unsigned teamLevel = A._teams->TeamLevel(A._level);
                    std::memcpy
                    ( &buffer[offsets[teamLevel]],
                      C._ZMap.Get( key ).LockedBuffer(),
                      DFA.rank*DFGB.rank*sizeof(Scalar) );
                    offsets[teamLevel] += DFA.rank*DFGB.rank;
                }
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
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainBroadcastsUnpackC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int key = A._sourceOffset;
    const bool admissibleC = C.Admissible();
    switch( A._block.type )
    {
    case DIST_NODE:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( !admissibleC && A._level+1 < endLevel )
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        case DIST_LOW_RANK:
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseBroadcastsUnpack
                ( C._mainContextMap.Get( key ), buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B._block.type )
        {
        case DIST_NODE:
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseBroadcastsUnpack
                ( C._mainContextMap.Get( key ), buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A._inTargetTeam )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRank& DFB = *B._block.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A._teams->TeamLevel(A._level);
                    std::memcpy
                    ( C._ZMap.Get( key ).Buffer(), &buffer[offsets[teamLevel]],
                      DFA.rank*DFB.rank*sizeof(Scalar) );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            if( A._level >= startLevel && A._level < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A._block.data.DF;
                const DistLowRankGhost& DFGB = *B._block.data.DFG;
                if( DFA.rank != 0 && DFGB.rank != 0 )
                {
                    const unsigned teamLevel = A._teams->TeamLevel(A._level);
                    std::memcpy
                    ( C._ZMap.Get( key ).Buffer(), &buffer[offsets[teamLevel]],
                      DFA.rank*DFGB.rank*sizeof(Scalar) );
                    offsets[teamLevel] += DFA.rank*DFGB.rank;
                }
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
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel, int startUpdate, int endUpdate )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcompute");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A.MultiplyHMatMainPostcomputeA( startLevel, endLevel );
    B.MultiplyHMatMainPostcomputeB( startLevel, endLevel );
    A.MultiplyHMatMainPostcomputeC
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
    C.MultiplyHMatMainPostcomputeCCleanup( startLevel, endLevel );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeA
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeA");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    // Handle postcomputation of A's row space
    if( A._level >= startLevel && A._level < endLevel &&
        A._beganRowSpaceComp && !A._finishedRowSpaceComp )
    {
        A.AdjointMultiplyDensePostcompute
        ( A._rowContext, (Scalar)1, A._rowOmega, A._rowT );
        A._rowContext.Clear();
        A._finishedRowSpaceComp = true;
    }

    switch( A._block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( A._level+1 < endLevel )
        {
            Node& nodeA = *A._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeA.Child(t,s).MultiplyHMatMainPostcomputeA
                    ( startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeB
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeB");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& B = *this;

    // Handle postcomputation of B's column space
    if( B._level >= startLevel && B._level < endLevel &&
        B._beganColSpaceComp && !B._finishedColSpaceComp )
    {
        B.MultiplyDensePostcompute
        ( B._colContext, (Scalar)1, B._colOmega, B._colT );
        B._colContext.Clear();
        B._finishedColSpaceComp = true;
    }

    switch( B._block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( B._level+1 < endLevel )
        {
            Node& nodeB = *B._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeB.Child(t,s).MultiplyHMatMainPostcomputeB
                    ( startLevel, endLevel );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeC
( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                      DistQuasi2dHMat<Scalar,Conjugated>& C,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatMainPostcomputeC");
#endif
    const DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Handle all H H recursion here
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
                if( A._level+1 < endLevel )
                {
                    const Node& nodeA = *A._block.data.N;
                    const Node& nodeB = *B._block.data.N;
                    Node& nodeC = *C._block.data.N;
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            for( int r=0; r<4; ++r )
                                nodeA.Child(t,r).MultiplyHMatMainPostcomputeC
                                ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                                  startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
#ifndef RELEASE
                PopCallStack();
#endif
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

    if( A._level < startLevel || A._level >= endLevel ||
        update < startUpdate || update >= endUpdate )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
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
            ( C._mainContextMap.Get( key ), 
              alpha, DFB.ULocal, C._UMap.Get( key ) );
            C._mainContextMap.Get( key ).Clear();
            C._mainContextMap.Erase( key );

            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFB.VLocal, C._VMap.Get( key ) );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            const DistLowRankGhost& DFGB = *B._block.data.DFG; 
            Dense<Scalar> dummy( 0, DFGB.rank );
            C._UMap.Set( key, new Dense<Scalar>(LocalHeight(), DFGB.rank ) );
            hmat_tools::Scale( (Scalar)0, C._UMap.Get( key ) );
            A.MultiplyDensePostcompute
            ( C._mainContextMap.Get( key ), alpha, dummy, C._UMap.Get( key ) );
            C._mainContextMap.Get( key ).Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_NODE_GHOST:
    {
        switch( B._block.type )
        {
        case DIST_NODE:
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            C._VMap.Set( key, new Dense<Scalar>( B.LocalWidth(), DFB.rank ) );
            hmat_tools::Copy( DFB.VLocal, C._VMap.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                C._UMap.Set( key, new Dense<Scalar>( C.Height(), SFB.rank ) );
                hmat_tools::Scale( (Scalar)0, C._UMap.Get( key ) );
                Dense<Scalar> dummy( 0, SFB.rank );
                A.MultiplyDensePostcompute
                ( C._mainContextMap.Get( key ), 
                  alpha, dummy, C._UMap.Get( key ) );

                C._VMap.Set( key, new Dense<Scalar>( C.Width(), SFB.rank ) );
                hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            }
            C._mainContextMap.Get( key ).Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._UMap.Set( key, new Dense<Scalar>( C.Height(), SFGB.rank ) );
            hmat_tools::Scale( (Scalar)0, C._UMap.Get( key ) );
            Dense<Scalar> dummy( 0, SFGB.rank );
            A.MultiplyDensePostcompute
            ( C._mainContextMap.Get( key ), alpha, dummy, C._UMap.Get( key ) );
            C._mainContextMap.Get( key ).Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        case LOW_RANK:
        {
            // We are the middle and right process
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._VMap.Set( key, new Dense<Scalar>( C.Width(), FB.Rank() ) );
            hmat_tools::Copy( FB.V, C._VMap.Get( key ) );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We are the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._UMap.Set( key, new Dense<Scalar>( C.Height(), FGB.rank ) );
            hmat_tools::Scale( (Scalar)0, C._UMap.Get( key ) );
            Dense<Scalar> dummy( 0, FGB.rank );
            A.MultiplyDensePostcompute
            ( C._mainContextMap.Get( key ), alpha, dummy, C._UMap.Get( key ) );
            C._mainContextMap.Get( key ).Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case NODE:
    {
        switch( B._block.type )
        {
        case SPLIT_NODE:
        case NODE:
        case SPLIT_LOW_RANK:
            break;
        case LOW_RANK:
        {
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FB.V, C._VMap.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._VMap.Set( key, new Dense<Scalar>( B.LocalWidth(), SFB.rank ) );
            hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            MultiplyDenseContext& context = C._mainContextMap.Get( key );

            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFA.ULocal, C._UMap.Get( key ) );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), DFA.VLocal, C._VMap.Get( key ) );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, DFA.VLocal, C._VMap.Get( key ) );
            context.Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        case DIST_NODE_GHOST:
        {
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFA.ULocal, C._UMap.Get( key ) );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            C._UMap.Set( key, new Dense<Scalar>( A.LocalHeight(), DFB.rank ) );
            C._VMap.Set( key, new Dense<Scalar> );
            Dense<Scalar>& UC = C._UMap.Get( key );
            Dense<Scalar>& VC = C._VMap.Get( key );

            if( A._inTargetTeam && DFA.rank != 0 && DFB.rank != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', A.LocalHeight(), DFB.rank, DFA.rank,
                  alpha,     DFA.ULocal.LockedBuffer(), DFA.ULocal.LDim(),
                             ZC.LockedBuffer(),         ZC.LDim(),
                  (Scalar)0, UC.Buffer(),               UC.LDim() );
                ZC.Clear();
                C._ZMap.Erase( key );
            }
            else
                hmat_tools::Scale( (Scalar)0, UC );
            hmat_tools::Copy( DFB.VLocal, VC );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            const DistLowRankGhost& DFGB = *B._block.data.DFG; 
            C._UMap.Set( key, new Dense<Scalar>( A.LocalHeight(), DFGB.rank ) );
            Dense<Scalar>& UC = C._UMap.Get( key );
            
            if( DFA.rank != 0 && DFGB.rank != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', A.LocalHeight(), DFGB.rank, DFA.rank,
                  alpha,     DFA.ULocal.LockedBuffer(), DFA.ULocal.LDim(),
                             ZC.LockedBuffer(),         ZC.LDim(),
                  (Scalar)0, UC.Buffer(),               UC.LDim() );
                ZC.Clear();
                C._ZMap.Erase( key );
            }
            else
                hmat_tools::Scale( (Scalar)0, UC );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
            Dense<Scalar> dummy( 0, DFGA.rank );
            C._VMap.Set( key, new Dense<Scalar>(C.LocalWidth(),DFGA.rank) );
            hmat_tools::Scale( (Scalar)0, C._VMap.Get( key ) );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), dummy, C._VMap.Get( key ) );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, C._VMap.Get( key ) );
            context.Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B._block.data.DF;
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFB.VLocal, C._VMap.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
                MultiplyDenseContext& context = C._mainContextMap.Get( key );
                C._UMap.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFA.D, C._UMap.Get( key ) );
                Dense<Scalar> dummy( 0, SFA.rank );
                C._VMap.Set( key, new Dense<Scalar>(C.LocalWidth(),SFA.rank) );
                hmat_tools::Scale( (Scalar)0, C._VMap.Get( key ) );
                if( Conjugated )
                    B.AdjointMultiplyDensePostcompute
                    ( context, Conj(alpha), dummy, C._VMap.Get( key ) );
                else
                    B.TransposeMultiplyDensePostcompute
                    ( context, alpha, dummy, C._VMap.Get( key ) );
                context.Clear();
                C._mainContextMap.Erase( key );
            }
            break;
        case SPLIT_NODE_GHOST:
            // We are the left process
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C._UMap.Get( key ) );
            break;
        case NODE:
            // The precompute is not needed
            break;
        case NODE_GHOST:
            // We are the left process
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C._UMap.Get( key ) );
            break;
        case SPLIT_LOW_RANK:
            // We are either the middle process or both the left and right.
            // The middle process is done.
            if( A._inTargetTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._UMap.Set( key, new Dense<Scalar>( A.Height(), SFB.rank ) );
                Dense<Scalar>& UC = C._UMap.Get( key );

                if( SFA.rank != 0 && SFB.rank != 0 )
                {
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    blas::Gemm
                    ( 'N', 'N', A.Height(), SFB.rank, SFA.rank,
                      alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                                 ZC.LockedBuffer(),    ZC.LDim(),
                      (Scalar)0, UC.Buffer(),          UC.LDim() );
                    ZC.Clear();
                    C._ZMap.Erase( key );
                }
                else
                    hmat_tools::Scale( (Scalar)0, UC );

                C._VMap.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._UMap.Set( key, new Dense<Scalar>( A.Height(), SFGB.rank ) );
            Dense<Scalar>& UC = C._UMap.Get( key );
            if( SFA.rank != 0 && SFGB.rank != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', A.Height(), SFGB.rank, SFA.rank,
                  alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0, UC.Buffer(),          UC.LDim() );
                ZC.Clear();
                C._ZMap.Erase( key );
            }
            else
                hmat_tools::Scale( (Scalar)0, UC );
            break;
        }
        case LOW_RANK:
        {
            // We must be the middle and right process
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FB.V, C._VMap.Get( key ) );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We must be the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._UMap.Set( key, new Dense<Scalar>( A.Height(), FGB.rank ) );
            Dense<Scalar>& UC = C._UMap.Get( key );
            if( SFA.rank != 0 && FGB.rank != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', A.Height(), FGB.rank, SFA.rank,
                  alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  (Scalar)0, UC.Buffer(),          UC.LDim() );
                ZC.Clear();
                C._ZMap.Erase( key );
            }
            else
                hmat_tools::Scale( (Scalar)0, UC );
            break;
        }
        case SPLIT_DENSE:
            // We are either the middle process or both the left and right
            if( A._inTargetTeam )
            {
                C._UMap.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFA.D, C._UMap.Get( key ) );

                const SplitDense& SDB = *B._block.data.SD;
                C._VMap.Set( key, new Dense<Scalar>( C.Width(), SFA.rank ) );
                Dense<Scalar>& VC = C._VMap.Get( key );
                if( SFA.rank != 0 && B.Height() != 0 )
                {
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
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
                    C._ZMap.Erase( key );
                }
                else
                    hmat_tools::Scale( (Scalar)0, VC );
            }
            break;
        case SPLIT_DENSE_GHOST:
            // We are the left process
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C._UMap.Get( key ) );
            break;
        case DENSE:
            // We are the middle and right process, there is nothing left to do
            break;
        case DENSE_GHOST:
            // We are the left process
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C._UMap.Get( key ) );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
            Dense<Scalar> dummy( 0, SFGA.rank );
            C._VMap.Set( key, new Dense<Scalar>(C.LocalWidth(),SFGA.rank) );
            hmat_tools::Scale( (Scalar)0, C._VMap.Get( key ) );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), dummy, C._VMap.Get( key ) );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, C._VMap.Get( key ) );
            context.Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            C._VMap.Set( key, new Dense<Scalar>( C.Width(), SFGA.rank ) );
            Dense<Scalar>& VC = C._VMap.Get( key );
            if( SFGA.rank != 0 && B.Height() != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
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
                C._ZMap.Erase( key );
            }
            else
                hmat_tools::Scale( (Scalar)0, VC );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C._UMap.Get( key ) );
            break;
        case NODE:
            // We own all of A, B, and C
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C._UMap.Get( key ) );
            break;
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B._block.data.SF;
            const int m = A.Height();
            const int k = FA.Rank();
            C._UMap.Set( key, new Dense<Scalar>( m, SFB.rank ) );
            Dense<Scalar>& UC = C._UMap.Get( key );
            if( SFB.rank != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', m, SFB.rank, k,
                  alpha,     FA.U.LockedBuffer(), FA.U.LDim(),
                             ZC.LockedBuffer(),   ZC.LDim(),
                  (Scalar)0, UC.Buffer(),         UC.LDim() );
            }
            else
                hmat_tools::Scale( (Scalar)0, UC );
            break;
        }
        case LOW_RANK:
        {
            // We own all of A, B, and C
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;
            const int m = A.Height();
            const int k = FA.Rank();

            C._UMap.Set( key, new Dense<Scalar>( m, FB.Rank() ) );
            C._VMap.Set( key, new Dense<Scalar> );
            Dense<Scalar>& UC = C._UMap.Get( key );
            Dense<Scalar>& VC = C._VMap.Get( key );
            if( FB.Rank() != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                blas::Gemm
                ( 'N', 'N', m, FB.Rank(), k,
                  alpha,     FA.U.LockedBuffer(), FA.U.LDim(),
                             ZC.LockedBuffer(),   ZC.LDim(),
                  (Scalar)0, UC.Buffer(),         UC.LDim() );
            }
            else
                hmat_tools::Scale( (Scalar)0, UC );
            hmat_tools::Copy( FB.V, VC );
            break;
        }
        case SPLIT_DENSE:
        {
            // We are the left and middle process
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C._UMap.Get( key ) );
            break;
        }
        case DENSE:
        {
            // We own all of A, B, and C
            C._UMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C._UMap.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            MultiplyDenseContext& context = C._mainContextMap.Get( key );
            Dense<Scalar> dummy( 0, FGA.rank );
            C._VMap.Set( key, new Dense<Scalar>(C.LocalWidth(),FGA.rank) );
            hmat_tools::Scale( (Scalar)0, C._VMap.Get( key ) );
            if( Conjugated )
                B.AdjointMultiplyDensePostcompute
                ( context, Conj(alpha), dummy, C._VMap.Get( key ) );
            else
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, C._VMap.Get( key ) );
            context.Clear();
            C._mainContextMap.Erase( key );
            break;
        }
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B._block.data.SF;
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            C._VMap.Set( key, new Dense<Scalar>( C.Width(), FGA.rank ) );
            Dense<Scalar>& VC = C._VMap.Get( key );
            if( FGA.rank != 0 && B.Height() != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
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
                C._ZMap.Erase( key );
            }
            else
                hmat_tools::Scale( (Scalar)0, VC );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        switch( B._block.type )
        {
        case SPLIT_LOW_RANK:
            // We are either the middle process or the left and right

            // TODO: This could be removed by modifying the PassData
            //       unpacking routine to perform this step.
            if( A._inTargetTeam )
            {
                const SplitLowRank& SFB = *B._block.data.SF;
                C._UMap.Set( key, new Dense<Scalar> ); 
                Dense<Scalar>& UC = C._UMap.Get( key );
                if( A.Height() != 0 && SFB.rank != 0 )
                    hmat_tools::Copy( C._ZMap.Get( key ), UC );
                else
                    UC.Resize( A.Height(), SFB.rank ); 

                C._VMap.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B._block.data.SFG;
            C._UMap.Set( key, new Dense<Scalar> );
            Dense<Scalar>& UC = C._UMap.Get( key );
            if( A.Height() != 0 && SFGB.rank != 0 )
                hmat_tools::Copy( C._ZMap.Get( key ), UC );
            else
                UC.Resize( A.Height(), SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // We are the middle and right process
            const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;    
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FB.V, C._VMap.Get( key ) );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We are the left process
            const LowRankGhost& FGB = *B._block.data.FG;
            C._UMap.Set( key, new Dense<Scalar> ); 
            Dense<Scalar>& UC = C._UMap.Get( key );
            // TODO: This could be removed by modifying the PassData
            //       unpacking routine to perform this step.
            if( A.Height() != 0 && FGB.rank != 0 )
                hmat_tools::Copy( C._ZMap.Get( key ), UC );
            else
                UC.Resize( A.Height(), FGB.rank );
            break;
        }
        case SPLIT_DENSE:
            if( C._inSourceTeam )
            {
                const SplitDense& SDB = *B._block.data.SD;
                const int m = C.Height();
                const int n = C.Width();
                const int k = A.Width();
                if( admissibleC )
                {
                    if( C._storedDenseUpdate )
                    {
                        if( m != 0 && k != 0 )
                        {
                            Dense<Scalar>& ZC = C._ZMap.Get( key );
                            blas::Gemm
                            ( 'N', 'N', m, n, k,
                              alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                         SDB.D.LockedBuffer(), SDB.D.LDim(),
                              (Scalar)1, C._D.Buffer(),        C._D.LDim() );
                            ZC.Clear();
                            C._ZMap.Erase( key );
                        }
                    }
                    else
                    {
                        C._D.Resize( m, n );
                        if( m != 0 && k != 0 )
                        {
                            Dense<Scalar>& ZC = C._ZMap.Get( key );
                            blas::Gemm
                            ( 'N', 'N', m, n, k,
                              alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                         SDB.D.LockedBuffer(), SDB.D.LDim(),
                              (Scalar)0, C._D.Buffer(),        C._D.LDim() );
                            ZC.Clear();
                            C._ZMap.Erase( key );
                        }
                        else
                            hmat_tools::Scale( (Scalar)0, C._D );
                        C._storedDenseUpdate = true;
                    }
                }
                else if( m != 0 && k != 0 )
                {
                    Dense<Scalar>& ZC = C._ZMap.Get( key );
                    Dense<Scalar>& D = *C._block.data.D;
                    blas::Gemm
                    ( 'N', 'N', m, n, k,
                      alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                 SDB.D.LockedBuffer(), SDB.D.LDim(),
                      (Scalar)1, D.Buffer(),           D.LDim() );
                    ZC.Clear();
                    C._ZMap.Erase( key );
                }
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
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            const int m = C.Height();
            const int n = C.Width();
            const int k = A.Width();
            if( admissibleC )
            {
                if( C._storedDenseUpdate )
                {
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C._ZMap.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          (Scalar)1, C._D.Buffer(),        C._D.LDim() );
                        ZC.Clear();
                        C._ZMap.Erase( key );
                    }
                }
                else
                {
                    C._D.Resize( m, n );
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C._ZMap.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          (Scalar)0, C._D.Buffer(),        C._D.LDim() );
                        ZC.Clear();
                        C._ZMap.Erase( key );
                    }
                    else
                        hmat_tools::Scale( (Scalar)0, C._D );
                    C._storedDenseUpdate = true;
                }
            }
            else if( m != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                Dense<Scalar>& D = *C._block.data.D;
                blas::Gemm
                ( 'N', 'N', m, n, k,
                  alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  (Scalar)1, D.Buffer(),           D.LDim() );
                ZC.Clear();
                C._ZMap.Erase( key );
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
            C._VMap.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C._VMap.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B._block.data.SD;
            const int m = C.Height();
            const int n = C.Width();
            const int k = A.Width();
            if( admissibleC )
            {
                if( C._storedDenseUpdate )
                {
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C._ZMap.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          (Scalar)1, C._D.Buffer(),        C._D.LDim() );
                        ZC.Clear();
                        C._ZMap.Erase( key );
                    }
                }
                else
                {
                    C._D.Resize( m, n );
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C._ZMap.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          (Scalar)0, C._D.Buffer(),        C._D.LDim() );
                        ZC.Clear();
                        C._ZMap.Erase( key );
                    }
                    else
                        hmat_tools::Scale( (Scalar)0, C._D );
                    C._storedDenseUpdate = true;
                }
            }
            else if( m != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C._ZMap.Get( key );
                Dense<Scalar>& D = *C._block.data.D;
                blas::Gemm
                ( 'N', 'N', m, n, k,
                  alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  (Scalar)1, D.Buffer(),           D.LDim() );
                ZC.Clear();
                C._ZMap.Erase( key );
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A._block.type) << ", "
              << BlockTypeString(B._block.type) << ", "
              << BlockTypeString(C._block.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatMainPostcomputeCCleanup
( int startLevel, int endLevel )
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
        if( C._level+1 < endLevel )
        {
            Node& nodeC = *C._block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeC.Child(t,s).MultiplyHMatMainPostcomputeCCleanup
                    ( startLevel, endLevel );
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

