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
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
#endif
    YLocal.Resize( LocalHeight(), XLocal.Width() );
    Multiply( alpha, XLocal, (Scalar)0, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiply");
#endif
    YLocal.Resize( LocalWidth(), XLocal.Width() );
    TransposeMultiply( alpha, XLocal, (Scalar)0, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiply");
#endif
    YLocal.Resize( LocalWidth(), XLocal.Width() );
    AdjointMultiply( alpha, XLocal, (Scalar)0, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const Dense<Scalar>& XLocal, 
  Scalar beta,        Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
#endif
    RequireRoot();
    if( XLocal.Height() == 0 || YLocal.Height() == 0 || XLocal.Width() == 0 )
        return;
    hmat_tools::Scale( beta, YLocal );

    MultiplyDenseContext context;
    MultiplyDenseInitialize( context );
    MultiplyDensePrecompute( context, alpha, XLocal, YLocal );

    MultiplyDenseSummations( context, XLocal.Width() );
    //MultiplyDenseNaiveSummations( context, XLocal.Width() );

    MultiplyDensePassData( context, alpha, XLocal, YLocal );

    MultiplyDenseBroadcasts( context, XLocal.Width() );
    //MultiplyDenseNaiveBroadcasts( context, XLocal.Width() );

    MultiplyDensePostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Dense<Scalar>& XLocal, 
  Scalar beta,        Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiply");
#endif
    RequireRoot();
    if( XLocal.Height() == 0 || YLocal.Height() == 0 || XLocal.Width() == 0 )
        return;
    hmat_tools::Scale( beta, YLocal );

    TransposeMultiplyDenseContext context;
    TransposeMultiplyDenseInitialize( context );
    TransposeMultiplyDensePrecompute( context, alpha, XLocal, YLocal );

    TransposeMultiplyDenseSummations( context, XLocal.Width() );
    //TransposeMultiplyDenseNaiveSummations( context, XLocal.Width() );

    TransposeMultiplyDensePassData( context, alpha, XLocal, YLocal );

    TransposeMultiplyDenseBroadcasts( context, XLocal.Width() );
    //TransposeMultiplyDenseNaiveBroadcasts( context, XLocal.Width() );

    TransposeMultiplyDensePostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Dense<Scalar>& XLocal, 
  Scalar beta,        Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, YLocal );
    if( XLocal.Height() == 0 || YLocal.Height() == 0 || XLocal.Width() == 0 )
        return;

    AdjointMultiplyDenseContext context;
    AdjointMultiplyDenseInitialize( context );
    AdjointMultiplyDensePrecompute( context, alpha, XLocal, YLocal );

    AdjointMultiplyDenseSummations( context, XLocal.Width() );
    //AdjointMultiplyDenseNaiveSummations( context, XLocal.Width() );

    AdjointMultiplyDensePassData( context, alpha, XLocal, YLocal );

    AdjointMultiplyDenseBroadcasts( context, XLocal.Width() );
    //AdjointMultiplyDenseNaiveBroadcasts( context, XLocal.Width() );

    AdjointMultiplyDensePostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseInitialize
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseInitialize");
#endif
    context.Clear();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        context.block.type = DIST_NODE;
        context.block.data.DN = new typename MultiplyDenseContext::DistNode;

        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_NODE:
    {
        context.block.type = SPLIT_NODE;
        context.block.data.SN = new typename MultiplyDenseContext::SplitNode;

        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        context.block.type = DIST_LOW_RANK;
        context.block.data.Z = new Dense<Scalar>;
        break;
    case SPLIT_LOW_RANK:
        context.block.type = SPLIT_LOW_RANK;
        context.block.data.Z = new Dense<Scalar>;
        break;
    case SPLIT_DENSE:
        context.block.type = SPLIT_DENSE;
        context.block.data.Z = new Dense<Scalar>;
        break;

    default:
        context.block.type = EMPTY;
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseInitialize
( TransposeMultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseInitialize");
#endif
    // The non-transposed initialization is identical
    TransposeMultiplyDenseInitialize( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseInitialize
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyDenseInitialize( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePrecompute
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePrecompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePrecompute
                ( context, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // Form Z := alpha VLocal^[T/H] XLocal
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            Z.Resize( DF.rank, width );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', DF.rank, width, DF.VLocal.Height(), 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, Z.Buffer(),               Z.LDim() );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Dense<Scalar>& Z = *context.block.data.Z;

            Dense<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
                hmat_tools::AdjointMultiply( alpha, SF.D, XLocalSub, Z );
            else
                hmat_tools::TransposeMultiply( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRank<Scalar,Conjugated>& F = *_block.data.F;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, F.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, F.Height(), width );
        hmat_tools::Multiply( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            Dense<Scalar>& Z = *context.block.data.Z;

            Dense<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, Width(), width );
            hmat_tools::Multiply( alpha, SD.D, XLocalSub, Z );
        }
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, D.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, D.Height(), width );
        hmat_tools::Multiply( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePrecompute
( TransposeMultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePrecompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename TransposeMultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePrecompute
                ( context, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^T XLocal
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            Z.Resize( DF.rank, width );
            blas::Gemm
            ( 'T', 'N', DF.rank, width, DF.ULocal.Height(),
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, Z.Buffer(),               Z.LDim() );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Dense<Scalar>& Z = *context.block.data.Z;
            Dense<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmat_tools::TransposeMultiply( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRank<Scalar,Conjugated>& F = *_block.data.F;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmat_tools::TransposeMultiply
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmat_tools::TransposeMultiply
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDensePrecompute
( AdjointMultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDensePrecompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename AdjointMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyDensePrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename AdjointMultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyDensePrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyDensePrecompute
                ( context, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^H XLocal
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            Z.Resize( DF.rank, width );
            blas::Gemm
            ( 'C', 'N', DF.rank, width, DF.ULocal.Height(), 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, Z.Buffer(),               Z.LDim() );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Dense<Scalar>& Z = *context.block.data.Z;

            Dense<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmat_tools::AdjointMultiply( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRank<Scalar,Conjugated>& F = *_block.data.F;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmat_tools::AdjointMultiply
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmat_tools::AdjointMultiply
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSummations
( MultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MultiplyDenseSummationsCount( sizes, width );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    MultiplyDenseSummationsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    for( int i=0; i<numReduces; ++i )
    {
        if( sizes[i] != 0 )
        {
            MPI_Comm team = _subcomms->Subcomm( i+1 );
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

    // Unpack the reduced buffers (only roots of subcommunicators have data)
    MultiplyDenseSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSummations
( TransposeMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMultiplyDenseSummationsCount( sizes, width );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    TransposeMultiplyDenseSummationsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    for( int i=0; i<numReduces; ++i )
    {
        if( sizes[i] != 0 )
        {
            MPI_Comm team = _subcomms->Subcomm( i+1 );
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

    // Unpack the reduced buffers (only roots of subcommunicators have data)
    TransposeMultiplyDenseSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseSummations
( AdjointMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseSummations");
#endif
    // This unconjugated version is identical
    TransposeMultiplyDenseSummations( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSummationsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSummationsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseSummationsCount( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += _block.data.DF->rank*width;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSummationsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSummationsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseSummationsCount
                ( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += _block.data.DF->rank*width;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSummationsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Dense<Scalar>& Z = *context.block.data.Z;
            const int width = Z.Width();
            if( Z.Height() == Z.LDim() )
                std::memcpy
                ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
            else
                for( int j=0; j<width; ++j )
                    std::memcpy
                    ( &buffer[offsets[_level-1]+j*DF.rank], Z.LockedBuffer(0,j),
                      DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  TransposeMultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSummationsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Dense<Scalar>& Z = *context.block.data.Z;
            const int width = Z.Width();
            if( Z.Height() == Z.LDim() )
                std::memcpy
                ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
            else
                for( int j=0; j<width; ++j )
                    std::memcpy
                    ( &buffer[offsets[_level-1]+j*DF.rank], Z.LockedBuffer(0,j),
                      DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSummationsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            const int width = Z.Width();
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( Z.Height() == Z.LDim() )
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_level-1]], 
                      DF.rank*width*sizeof(Scalar) );
                else
                    for( int j=0; j<width; ++j )
                        std::memcpy
                        ( Z.Buffer(0,j), &buffer[offsets[_level-1]+j*DF.rank],
                          DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  TransposeMultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSummationsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            const int width = Z.Width();
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( Z.Height() == Z.LDim() )
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_level-1]], 
                      DF.rank*width*sizeof(Scalar) );
                else
                    for( int j=0; j<width; ++j )
                        std::memcpy
                        ( Z.Buffer(0,j), &buffer[offsets[_level-1]+j*DF.rank],
                          DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseNaiveSummations
( MultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseNaiveSummations");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseNaiveSummations
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                Dense<Scalar>& Z = *context.block.data.Z;
                MPI_Comm team = _subcomms->Subcomm( _level );
                int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                    mpi::Reduce
                    ( (const Scalar*)MPI_IN_PLACE, Z.Buffer(), 
                      DF.rank*width, 0, MPI_SUM, team );
                else
                    mpi::Reduce
                    ( Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseNaiveSummations
( TransposeMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseNaiveSummations");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseNaiveSummations
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                Dense<Scalar>& Z = *context.block.data.Z;
                MPI_Comm team = _subcomms->Subcomm( _level );
                int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                    mpi::Reduce
                    ( (const Scalar*)MPI_IN_PLACE, Z.Buffer(), 
                      DF.rank*width, 0, MPI_SUM, team );
                else
                    mpi::Reduce
                    ( Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseNaiveSummations
( AdjointMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyDenseNaiveSummations( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePassData
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal,
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassData");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                for( int t=0; t<2; ++t )
                    for( int s=0; s<2; ++s )
                        node.Child(t,s).MultiplyDensePassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                for( int t=2; t<4; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).MultiplyDensePassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    node.Child(t,s).MultiplyDensePassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            // Bottom-left quadrant
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    node.Child(t,s).MultiplyDensePassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MultiplyDensePassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).MultiplyDensePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MultiplyDensePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).MultiplyDensePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).MultiplyDensePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).MultiplyDensePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).MultiplyDensePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MultiplyDensePassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).MultiplyDensePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MultiplyDensePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).MultiplyDensePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).MultiplyDensePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).MultiplyDensePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).MultiplyDensePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MultiplyDensePassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).MultiplyDensePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MultiplyDensePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).MultiplyDensePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).MultiplyDensePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).MultiplyDensePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).MultiplyDensePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MultiplyDensePassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).MultiplyDensePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MultiplyDensePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).MultiplyDensePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).MultiplyDensePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).MultiplyDensePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).MultiplyDensePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            default:
                // This should be impossible
                break;
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;

        std::size_t bufferSize = 0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataSplitNodeCount
                ( bufferSize );
        bufferSize *= width;

        std::vector<byte> buffer( bufferSize );
        MPI_Comm comm = _subcomms->Subcomm(0);
        if( _inSourceTeam )
        {
            byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyDensePassDataSplitNodePack
                    ( head, nodeContext.Child(t,s) );
            if( bufferSize != 0 )
                mpi::Send( &buffer[0], bufferSize, _targetRoot, 0, comm );
        }
        else
        {
            if( bufferSize != 0 )
                mpi::Recv( &buffer[0], bufferSize, _sourceRoot, 0, comm );
            const byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyDensePassDataSplitNodeUnpack
                    ( head, nodeContext.Child(t,s), width );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        Dense<Scalar>& Z = *context.block.data.Z;
        if( DF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( _inSourceTeam )
                    mpi::Send
                    ( Z.LockedBuffer(), DF.rank*width, _targetRoot, 0, comm );
                else
                {
                    Z.Resize( DF.rank, width, DF.rank );
                    mpi::Recv
                    ( Z.Buffer(), DF.rank*width, _sourceRoot, 0, comm );
                }
            }
        }
        else
            Z.Resize( 0, width );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Dense<Scalar>& Z = *context.block.data.Z;
        if( SF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inSourceTeam )
                mpi::Send
                ( Z.LockedBuffer(), SF.rank*width, _targetRoot, 0, comm );
            else
            {
                Z.Resize( SF.rank, width, SF.rank );
                mpi::Recv( Z.Buffer(), SF.rank*width, _sourceRoot, 0, comm );
            }
        }
        else
            Z.Resize( 0, width );
        break;
    }
    case SPLIT_DENSE:
    {
        Dense<Scalar>& Z = *context.block.data.Z;
        if( Height() != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inSourceTeam )
                mpi::Send
                ( Z.LockedBuffer(), Z.Height()*width, _targetRoot, 0, comm );
            else
            {
                Z.Resize( Height(), width, Height() );
                mpi::Recv
                ( Z.Buffer(), Z.Height()*width, _sourceRoot, 0, comm );
            }
        }
        else
            Z.Resize( 0, width );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePassDataSplitNodePack
( byte*& head, const MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassDataSplitNodePack");
    if( !_inSourceTeam )
        throw std::logic_error("Calling process should be in source team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyDenseContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataSplitNodePack
                ( head, nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const Dense<Scalar>& Z = *context.block.data.Z;
        Write( head, Z.LockedBuffer(), Z.Height()*Z.Width() );
        break;
    }
    case SPLIT_DENSE:
    {
        const Dense<Scalar>& Z = *context.block.data.Z;
        Write( head, Z.LockedBuffer(), Z.Height()*Z.Width() );
        break;
    }
    default:
#ifndef RELEASE
        throw std::logic_error("This should be impossible");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::
MultiplyDensePassDataSplitNodeUnpack
( const byte*& head, MultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassDataSplitNodeUnpack");
    if( !_inTargetTeam )
        throw std::logic_error("Calling process should be in target team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataSplitNodeUnpack
                ( head, nodeContext.Child(t,s), width );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Dense<Scalar>& Z = *context.block.data.Z;
        if( SF.rank != 0 )
        {
            Z.Resize( SF.rank, width, SF.rank );
            Read( Z.Buffer(), head, Z.Height()*width );
        }
        else
            Z.Resize( 0, width );
        break;
    }
    case SPLIT_DENSE:
    {
        Dense<Scalar>& Z = *context.block.data.Z;
        if( Height() != 0 )
        {
            Z.Resize( Height(), width, Height() );
            Read( Z.Buffer(), head, Z.Height()*width );
        }
        else
            Z.Resize( 0, width );
        break;
    }
    default:
#ifndef RELEASE
        throw std::logic_error("This should be impossible");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePassData
( TransposeMultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal,
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePassData");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                for( int t=0; t<2; ++t )
                    for( int s=0; s<2; ++s )
                        node.Child(t,s).TransposeMultiplyDensePassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                for( int t=2; t<4; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).TransposeMultiplyDensePassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyDensePassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            // Bottom-left quadrant
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    node.Child(t,s).TransposeMultiplyDensePassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal ); 
                node.Child(3,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMultiplyDensePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMultiplyDensePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            default:
                // This should be impossible
                break;
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::SplitNode& nodeContext =
            *context.block.data.SN;

        std::size_t bufferSize = 0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodeCount
                ( bufferSize );
        bufferSize *= width;

        std::vector<byte> buffer( bufferSize );
        MPI_Comm comm = _subcomms->Subcomm(0);
        if( _inTargetTeam )
        {
            byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyDensePassDataSplitNodePack
                    ( head, nodeContext.Child(t,s), XLocal );
            if( bufferSize != 0 )
                mpi::Send( &buffer[0], bufferSize, _sourceRoot, 0, comm );
        }
        else
        {
            if( bufferSize != 0 )
                mpi::Recv( &buffer[0], bufferSize, _targetRoot, 0, comm );
            const byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).
                    TransposeMultiplyDensePassDataSplitNodeUnpack
                    ( head, nodeContext.Child(t,s), width );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        Dense<Scalar>& Z = *context.block.data.Z;
        if( DF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( _inTargetTeam )
                    mpi::Send
                    ( Z.LockedBuffer(), DF.rank*width, _sourceRoot, 0, comm );
                else
                {
                    Z.Resize( DF.rank, width, DF.rank );
                    mpi::Recv
                    ( Z.Buffer(), DF.rank*width, _targetRoot, 0, comm );
                }
            }
        }
        else
            Z.Resize( 0, width );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Dense<Scalar>& Z = *context.block.data.Z;
        if( SF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inTargetTeam )
                mpi::Send
                ( Z.LockedBuffer(), SF.rank*width, _sourceRoot, 0, comm );
            else
            {
                Z.Resize( SF.rank, width, SF.rank );
                mpi::Recv( Z.Buffer(), SF.rank*width, _targetRoot, 0, comm );
            }
        }
        else
            Z.Resize( 0, width );
        break;
    }
    case SPLIT_DENSE:
    {
        const int height = Height();
        Dense<Scalar>& Z = *context.block.data.Z;
        if( height != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inTargetTeam )
            {
                Dense<Scalar> XLocalSub;
                XLocalSub.LockedView
                ( XLocal, _localTargetOffset, 0, height, width );
                if( XLocalSub.LDim() != XLocalSub.Height() )
                {
                    // We must pack first
                    Z.Resize( height, width, height );
                    for( int j=0; j<width; ++j )
                        std::memcpy
                        ( Z.Buffer(0,j), XLocalSub.LockedBuffer(0,j), 
                          height*sizeof(Scalar) );
                    mpi::Send
                    ( Z.LockedBuffer(), height*width, _sourceRoot, 0, comm );
                }
                else
                {
                    mpi::Send
                    ( XLocalSub.LockedBuffer(), height*width, 
                      _sourceRoot, 0, comm );
                }
            }
            else
            {
                Z.Resize( height, width, height );
                mpi::Recv( Z.Buffer(), height*width, _targetRoot, 0, comm );
            }
        }
        else
            Z.Resize( 0, width );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::
TransposeMultiplyDensePassDataSplitNodePack
( byte*& head, const TransposeMultiplyDenseContext& context,
  const Dense<Scalar>& XLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::TransposeMultiplyDensePassDataSplitNodePack");
    if( !_inTargetTeam )
        throw std::logic_error("Calling process should be in target team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        const typename TransposeMultiplyDenseContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataSplitNodePack
                ( head, nodeContext.Child(t,s), XLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const Dense<Scalar>& Z = *context.block.data.Z;
        Write( head, Z.LockedBuffer(), Z.Height()*Z.Width() );
        break;
    }
    case SPLIT_DENSE:
    {
        const int height = Height();
        const int width = XLocal.Width();
        for( int j=0; j<width; ++j )
            Write( head, XLocal.LockedBuffer(_localTargetOffset,j), height );
        break;
    }
    default:
#ifndef RELEASE
        throw std::logic_error("This should be impossible");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::
TransposeMultiplyDensePassDataSplitNodeUnpack
( const byte*& head, TransposeMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::TransposeMultiplyDensePassDataSplitNodeUnpack");
    if( !_inSourceTeam )
        throw std::logic_error("Calling process should be in source team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataSplitNodeUnpack
                ( head, nodeContext.Child(t,s), width );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Dense<Scalar>& Z = *context.block.data.Z;
        if( SF.rank != 0 )
        {
            Z.Resize( SF.rank, width, SF.rank );
            Read( Z.Buffer(), head, Z.Height()*width );
        }
        else
            Z.Resize( 0, width );
        break;
    }
    case SPLIT_DENSE:
    {
        Dense<Scalar>& Z = *context.block.data.Z;
        if( Height() != 0 )
        {
            Z.Resize( Height(), width, Height() );
            Read( Z.Buffer(), head, Z.Height()*width );
        }
        else
            Z.Resize( 0, width );
        break;
    }
    default:
#ifndef RELEASE
        throw std::logic_error("This should be impossible");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDensePassData
( AdjointMultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal,
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::AdjointMultiplyDensePassData");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyDensePassData( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseBroadcasts
( MultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyDenseBroadcastsCount( sizes, width );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    MultiplyDenseBroadcastsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( sizes[i] != 0 )
        {
            MPI_Comm team = _subcomms->Subcomm( i+1 );
            mpi::Broadcast( &buffer[offsets[i]], sizes[i], 0, team );
        }
    }

    // Unpack the broadcasted buffers 
    MultiplyDenseBroadcastsUnpack( buffer, offsets, context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseBroadcasts
( TransposeMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMultiplyDenseBroadcastsCount( sizes, width );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    TransposeMultiplyDenseBroadcastsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( sizes[i] != 0 )
        {
            MPI_Comm team = _subcomms->Subcomm( i+1 );
            mpi::Broadcast( &buffer[offsets[i]], sizes[i], 0, team );
        }
    }

    // Unpack the broadcasted buffers 
    TransposeMultiplyDenseBroadcastsUnpack( buffer, offsets, context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseBroadcasts
( AdjointMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyDenseBroadcasts( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseBroadcastsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcastsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseBroadcastsCount( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += _block.data.DF->rank*width;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseBroadcastsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcastsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseBroadcastsCount
                ( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += _block.data.DF->rank*width;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseBroadcastsPack
                ( buffer, offsets, nodeContext.Child(t,s) ); 
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                const DistLowRank& DF = *_block.data.DF;
                const Dense<Scalar>& Z = *context.block.data.Z;
                const int width = Z.Width();
                std::memcpy
                ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::
TransposeMultiplyDenseBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  TransposeMultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseBroadcastsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                const DistLowRank& DF = *_block.data.DF;
                const Dense<Scalar>& Z = *context.block.data.Z;
                const int width = Z.Width();
                std::memcpy
                ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcastsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( DF.rank != 0 )
            {
                Z.Resize( DF.rank, width, DF.rank );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
            else
                Z.Resize( 0, width );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::
TransposeMultiplyDenseBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  TransposeMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( DF.rank != 0 )
            {
                Z.Resize( DF.rank, width, DF.rank );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
            else
                Z.Resize( 0, width );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseNaiveBroadcasts
( MultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseNaiveBroadcasts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseNaiveBroadcasts
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( DF.rank != 0 )
            {
                Z.Resize( DF.rank, width, DF.rank );
                MPI_Comm team = _subcomms->Subcomm( _level );
                mpi::Broadcast( Z.Buffer(), DF.rank*width, 0, team );
            }
            else
                Z.Resize( 0, width );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::
TransposeMultiplyDenseNaiveBroadcasts
( TransposeMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseNaiveBroadcasts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseNaiveBroadcasts
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( DF.rank != 0 )
            {
                Z.Resize( DF.rank, width, DF.rank );
                MPI_Comm team = _subcomms->Subcomm( _level );
                mpi::Broadcast( Z.Buffer(), DF.rank*width, 0, team );
            }
            else
                Z.Resize( 0, width );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseNaiveBroadcasts
( AdjointMultiplyDenseContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyDenseNaiveBroadcasts( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePostcompute
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePostcompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inTargetTeam )
        {
            const Node& node = *_block.data.N;
            typename MultiplyDenseContext::SplitNode& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyDensePostcompute
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // YLocal += ULocal Z 
            const DistLowRank& DF = *_block.data.DF;
            const Dense<Scalar>& Z = *context.block.data.Z;
            blas::Gemm
            ( 'N', 'N', DF.ULocal.Height(), width, DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         Z.LockedBuffer(),         Z.LDim(),
              (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            const Dense<Scalar>& Z = *context.block.data.Z;
            Dense<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmat_tools::Multiply( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const Dense<Scalar>& Z = *context.block.data.Z;
            const int localHeight = Height();
            for( int j=0; j<width; ++j )
            {
                const Scalar* ZCol = Z.LockedBuffer(0,j);
                Scalar* YCol = YLocal.Buffer(_localTargetOffset,j);
                for( int i=0; i<localHeight; ++i )
                    YCol[i] += ZCol[i];
            }
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePostcompute
( TransposeMultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePostcompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename TransposeMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inSourceTeam )
        {
            const Node& node = *_block.data.N;
            typename TransposeMultiplyDenseContext::SplitNode& nodeContext =
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyDensePostcompute
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^T Z 
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( Conjugated )
            {
                // YLocal += conj(VLocal) Z
                hmat_tools::Conjugate( Z );
                hmat_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmat_tools::Conjugate( YLocal );
            }
            else
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
            }
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Dense<Scalar>& Z = *context.block.data.Z;
            Dense<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                // YLocal += conj(V) Z
                hmat_tools::Conjugate( Z );
                hmat_tools::Conjugate( YLocalSub );
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
                hmat_tools::Conjugate( YLocalSub );
            }
            else
            {
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            const Dense<Scalar>& Z = *context.block.data.Z;
            Dense<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmat_tools::TransposeMultiply
            ( alpha, SD.D, Z, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDensePostcompute
( AdjointMultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal,
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDensePostcompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename AdjointMultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyDensePostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inSourceTeam )
        {
            const Node& node = *_block.data.N;
            typename AdjointMultiplyDenseContext::SplitNode& nodeContext =
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).AdjointMultiplyDensePostcompute
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^H Z
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( Conjugated )
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
            }
            else
            {
                // YLocal += conj(VLocal) Z
                hmat_tools::Conjugate( Z );
                hmat_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmat_tools::Conjugate( YLocal );
            }
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Dense<Scalar>& Z = *context.block.data.Z;
            Dense<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
            }
            else
            {
                // YLocal += conj(V) Z
                hmat_tools::Conjugate( Z );
                hmat_tools::Conjugate( YLocalSub );
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
                hmat_tools::Conjugate( YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            const Dense<Scalar>& Z = *context.block.data.Z;
            Dense<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmat_tools::AdjointMultiply
            ( alpha, SD.D, Z, (Scalar)1, YLocalSub );
        }
        break;

    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

