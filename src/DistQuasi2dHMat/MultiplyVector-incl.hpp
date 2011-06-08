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
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
#endif
    yLocal.Resize( LocalHeight() );
    Multiply( alpha, xLocal, (Scalar)0, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiply");
#endif
    yLocal.Resize( LocalWidth() );
    TransposeMultiply( alpha, xLocal, (Scalar)0, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiply");
#endif
    yLocal.Resize( LocalWidth() );
    AdjointMultiply( alpha, xLocal, (Scalar)0, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta,        Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, yLocal );

    MultiplyVectorContext context;
    MultiplyVectorInitialize( context );
    MultiplyVectorPrecompute( context, alpha, xLocal, yLocal );

    MultiplyVectorSummations( context );
    //MultiplyVectorNaiveSummations( context );

    MultiplyVectorPassData( context, alpha, xLocal, yLocal );

    MultiplyVectorBroadcasts( context );
    //MultiplyVectorNaiveBroadcasts( context );

    MultiplyVectorPostcompute( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, yLocal );

    MultiplyVectorContext context;
    TransposeMultiplyVectorInitialize( context );
    TransposeMultiplyVectorPrecompute( context, alpha, xLocal, yLocal );

    TransposeMultiplyVectorSummations( context );
    //TransposeMultiplyVectorNaiveSummations( context );

    TransposeMultiplyVectorPassData( context, alpha, xLocal, yLocal );

    TransposeMultiplyVectorBroadcasts( context );
    //TransposeMultiplyVectorNaiveBroadcasts( context );

    TransposeMultiplyVectorPostcompute( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta,        Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, yLocal );

    MultiplyVectorContext context;
    AdjointMultiplyVectorInitialize( context );
    AdjointMultiplyVectorPrecompute( context, alpha, xLocal, yLocal );

    AdjointMultiplyVectorSummations( context );
    //AdjointMultiplyVectorNaiveSummations( context );

    AdjointMultiplyVectorPassData( context, alpha, xLocal, yLocal );

    AdjointMultiplyVectorBroadcasts( context );
    //AdjointMultiplyVectorNaiveBroadcasts( context );

    AdjointMultiplyVectorPostcompute( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorInitialize
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorInitialize");
#endif
    context.Clear();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        context.block.type = DIST_NODE;
        context.block.data.DN = 
            new typename MultiplyVectorContext::DistNodeContext;

        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorInitialize( nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_NODE:
    {
        context.block.type = SPLIT_NODE;
        context.block.data.SN = 
            new typename MultiplyVectorContext::SplitNodeContext;

        typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorInitialize( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        context.block.type = DIST_LOW_RANK;
        context.block.data.z = new Vector<Scalar>;
        break;
    case SPLIT_LOW_RANK:
        context.block.type = SPLIT_LOW_RANK;
        context.block.data.z = new Vector<Scalar>;
        break;
    case SPLIT_DENSE:
        context.block.type = SPLIT_DENSE;
        context.block.data.z = new Vector<Scalar>;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorInitialize
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyVectorInitialize( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorInitialize
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyVectorInitialize( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPrecompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPrecompute");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case SPLIT_NODE:
    {
       typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPrecompute
                ( context, alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // Form z := alpha VLocal^[T/H] xLocal
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemv
            ( option, DF.VLocal.Height(), DF.rank, 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, z.Buffer(),               1 );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Vector<Scalar>& z = *context.block.data.z;

            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
                hmat_tools::AdjointMultiply( alpha, SF.D, xLocalSub, z );
            else
                hmat_tools::TransposeMultiply( alpha, SF.D, xLocalSub, z );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, F.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, F.Height() );
        hmat_tools::Multiply( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            Vector<Scalar>& z = *context.block.data.z;

            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, Width() );
            hmat_tools::Multiply( alpha, SD.D, xLocalSub, z );
        }
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, D.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, D.Height() );
        hmat_tools::Multiply( alpha, D, xLocalSub, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPrecompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPrecompute");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPrecompute
                ( context, alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form z := alpha ULocal^T xLocal
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            blas::Gemv
            ( 'T', DF.ULocal.Height(), DF.rank, 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, z.Buffer(),               1 );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Vector<Scalar>& z = *context.block.data.z;
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SF.D.Height() );
            hmat_tools::TransposeMultiply( alpha, SF.D, xLocalSub, z );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, F.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, F.Width() );
        hmat_tools::TransposeMultiply
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, D.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, D.Width() );
        hmat_tools::TransposeMultiply
        ( alpha, D, xLocalSub, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorPrecompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorPrecompute");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyVectorPrecompute
                ( context, alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form z := alpha ULocal^H xLocal
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            blas::Gemv
            ( 'C', DF.ULocal.Height(), DF.rank, 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, z.Buffer(),               1 );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Vector<Scalar>& z = *context.block.data.z;

            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView
            ( xLocal, _localTargetOffset, SF.D.Height() );
            hmat_tools::AdjointMultiply( alpha, SF.D, xLocalSub, z );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, F.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, F.Width() );
        hmat_tools::AdjointMultiply
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, D.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, D.Width() );
        hmat_tools::AdjointMultiply
        ( alpha, D, xLocalSub, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSummations
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MultiplyVectorSummationsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    MultiplyVectorSummationsPack( buffer, offsets, context );

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
    MultiplyVectorSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSummations
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMultiplyVectorSummationsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    TransposeMultiplyVectorSummationsPack( buffer, offsets, context );

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
    TransposeMultiplyVectorSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorSummations
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorSummations");
#endif
    // This unconjugated version is identical
    TransposeMultiplyVectorSummations( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSummationsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSummationsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSummationsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += _block.data.DF->rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSummationsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSummationsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSummationsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += _block.data.DF->rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSummationsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            std::memcpy
            ( &buffer[offsets[_level-1]], z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSummationsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            std::memcpy
            ( &buffer[offsets[_level-1]], z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSummationsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSummationsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorNaiveSummations
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorNaiveSummations");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorNaiveSummations
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                MPI_Comm team = _subcomms->Subcomm( _level );
                int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                {
                    mpi::Reduce
                    ( (const Scalar*)MPI_IN_PLACE, z.Buffer(), 
                      DF.rank, 0, MPI_SUM, team );
                }
                else
                    mpi::Reduce
                    ( z.LockedBuffer(), 0, DF.rank, 0, MPI_SUM, team );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorNaiveSummations
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorNaiveSummations");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorNaiveSummations
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                MPI_Comm team = _subcomms->Subcomm( _level );
                int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                {
                    mpi::Reduce
                    ( (const Scalar*)MPI_IN_PLACE, z.Buffer(), 
                      DF.rank, 0, MPI_SUM, team );
                }
                else
                    mpi::Reduce
                    ( z.LockedBuffer(), 0, DF.rank, 0, MPI_SUM, team );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorNaiveSummations
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorNaiveSummations( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassData
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassData");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
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
                        node.Child(t,s).MultiplyVectorPassData
                        ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                for( int t=2; t<4; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).MultiplyVectorPassData
                        ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
            }
            // Top-right quadrant
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    node.Child(t,s).MultiplyVectorPassData
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );

            // Bottom-left quadrant
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    node.Child(t,s).MultiplyVectorPassData
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MultiplyVectorPassData
                ( nodeContext.Child(0,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(0,1).MultiplyVectorPassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).MultiplyVectorPassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(0,2).MultiplyVectorPassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).MultiplyVectorPassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(0,3).MultiplyVectorPassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).MultiplyVectorPassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MultiplyVectorPassData
                ( nodeContext.Child(1,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,1).MultiplyVectorPassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).MultiplyVectorPassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(1,3).MultiplyVectorPassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal );
                node.Child(3,1).MultiplyVectorPassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(1,2).MultiplyVectorPassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).MultiplyVectorPassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MultiplyVectorPassData
                ( nodeContext.Child(2,2), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(2,3).MultiplyVectorPassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).MultiplyVectorPassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,2).MultiplyVectorPassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).MultiplyVectorPassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,2).MultiplyVectorPassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).MultiplyVectorPassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MultiplyVectorPassData
                ( nodeContext.Child(3,3), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(2,3).MultiplyVectorPassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).MultiplyVectorPassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,3).MultiplyVectorPassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal );
                node.Child(3,1).MultiplyVectorPassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,3).MultiplyVectorPassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).MultiplyVectorPassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
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
        typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;

        std::size_t bufferSize = 0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataSplitNodeCount( bufferSize );
        std::vector<byte> buffer( bufferSize );

        MPI_Comm comm = _subcomms->Subcomm(0);
        if( _inSourceTeam )
        {
            byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyVectorPassDataSplitNodePack
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
                    node.Child(t,s).MultiplyVectorPassDataSplitNodeUnpack
                    ( head, nodeContext.Child(t,s) );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;

        const DistLowRank& DF = *_block.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        if( DF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( _inSourceTeam )
                    mpi::Send
                    ( z.LockedBuffer(), DF.rank, _targetRoot, 0, comm );
                else
                    mpi::Recv( z.Buffer(), DF.rank, _sourceRoot, 0, comm );
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( SF.rank );
        if( SF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inSourceTeam )
                mpi::Send( z.LockedBuffer(), SF.rank, _targetRoot, 0, comm );
            else
                mpi::Recv( z.Buffer(), SF.rank, _sourceRoot, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( Height() );
        if( Height() != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inSourceTeam )
                mpi::Send( z.LockedBuffer(), Height(), _targetRoot, 0, comm );
            else
                mpi::Recv( z.Buffer(), Height(), _sourceRoot, 0, comm );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassDataSplitNodeCount
( std::size_t& bufferSize ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassDataSplitNodeCount");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataSplitNodeCount( bufferSize );
        break;
    }
    case SPLIT_LOW_RANK:
        bufferSize += _block.data.SF->rank*sizeof(Scalar);
        break;
    case SPLIT_DENSE:
        bufferSize += Height()*sizeof(Scalar);
        break;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassDataSplitNodePack
( byte*& head, const MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassDataSplitNodePack");
    if( !_inSourceTeam )
        throw std::logic_error("Calling process should be in source team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataSplitNodePack
                ( head, nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const Vector<Scalar>& z = *context.block.data.z;
        Write( head, z.LockedBuffer(), z.Height() );
        break;
    }
    case SPLIT_DENSE:
    {
        const Vector<Scalar>& z = *context.block.data.z;
        Write( head, z.LockedBuffer(), z.Height() );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassDataSplitNodeUnpack
( const byte*& head, MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassDataSplitNodeUnpack");
    if( !_inTargetTeam )
        throw std::logic_error("Calling process should be in target team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataSplitNodeUnpack
                ( head, nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( _block.data.SF->rank );
        Read( z.Buffer(), head, z.Height() );
        break;
    }
    case SPLIT_DENSE:
    {
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( Height() );
        Read( z.Buffer(), head, z.Height() );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPassData
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPassData");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
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
                        node.Child(t,s).TransposeMultiplyVectorPassData
                        ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                for( int t=2; t<4; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).TransposeMultiplyVectorPassData
                        ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
            }
            // Top-right quadrant
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPassData
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
            // Bottom-left quadrant
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPassData
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal ); 
                node.Child(3,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,2), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,3), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal );
                node.Child(3,1).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMultiplyVectorPassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).TransposeMultiplyVectorPassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
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
        typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;

        std::size_t bufferSize = 0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodeCount
                ( bufferSize );
        std::vector<byte> buffer( bufferSize );

        MPI_Comm comm = _subcomms->Subcomm(0);
        if( _inTargetTeam )
        {
            byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodePack
                    ( head, nodeContext.Child(t,s), xLocal );
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
                    node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodeUnpack
                    ( head, nodeContext.Child(t,s) );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;

        const DistLowRank& DF = *_block.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        if( DF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( _inTargetTeam )
                    mpi::Send
                    ( z.LockedBuffer(), DF.rank, _sourceRoot, 0, comm );
                else
                    mpi::Recv( z.Buffer(), DF.rank, _targetRoot, 0, comm );
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( SF.rank );
        if( SF.rank != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inTargetTeam )
                mpi::Send( z.LockedBuffer(), SF.rank, _sourceRoot, 0, comm );
            else
                mpi::Recv( z.Buffer(), SF.rank, _targetRoot, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( Height() );
        if( Height() != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inTargetTeam )
            {
                Vector<Scalar> xLocalSub;
                xLocalSub.LockedView( xLocal, _localTargetOffset, Height() );
                mpi::Send
                ( xLocalSub.LockedBuffer(), Height(), _sourceRoot, 0, comm );
            }
            else
                mpi::Recv( z.Buffer(), Height(), _targetRoot, 0, comm );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::
TransposeMultiplyVectorPassDataSplitNodeCount
( std::size_t& bufferSize ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::TransposeMultiplyVectorPassDataSplitNodeCount");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodeCount
                ( bufferSize );
        break;
    }
    case SPLIT_LOW_RANK:
        bufferSize += _block.data.SF->rank*sizeof(Scalar);
        break;
    case SPLIT_DENSE:
        bufferSize += Height()*sizeof(Scalar);
        break;
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
TransposeMultiplyVectorPassDataSplitNodePack
( byte*& head, const MultiplyVectorContext& context,
  const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::TransposeMultiplyVectorPassDataSplitNodePack");
    if( !_inTargetTeam )
        throw std::logic_error("Calling process should be in target team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodePack
                ( head, nodeContext.Child(t,s), xLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const Vector<Scalar>& z = *context.block.data.z;
        Write( head, z.LockedBuffer(), z.Height() );
        break;
    }
    case SPLIT_DENSE:
        Write( head, xLocal.LockedBuffer(_localTargetOffset), Height() );
        break;
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
TransposeMultiplyVectorPassDataSplitNodeUnpack
( const byte*& head, MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::TransposeMultiplyVectorPassDataSplitNodeUnpack");
    if( !_inSourceTeam )
        throw std::logic_error("Calling process should be in source team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataSplitNodeUnpack
                ( head, nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( _block.data.SF->rank );
        Read( z.Buffer(), head, z.Height() );
        break;
    }
    case SPLIT_DENSE:
    {
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( Height() );
        Read( z.Buffer(), head, z.Height() );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorPassData
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorPassData");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorPassData( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyVectorBroadcastsCount( sizes );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    MultiplyVectorBroadcastsPack( buffer, offsets, context );

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
    MultiplyVectorBroadcastsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMultiplyVectorBroadcastsCount( sizes );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    TransposeMultiplyVectorBroadcastsPack( buffer, offsets, context );

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
    TransposeMultiplyVectorBroadcastsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorBroadcasts( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorBroadcastsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += _block.data.DF->rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorBroadcastsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += _block.data.DF->rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], z.LockedBuffer(), 
                  DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], z.LockedBuffer(), 
                  DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            std::memcpy
            ( z.Buffer(), &buffer[offsets[_level-1]], DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            std::memcpy
            ( z.Buffer(), &buffer[offsets[_level-1]], DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorNaiveBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorNaiveBroadcasts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorNaiveBroadcasts
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            if( DF.rank != 0 )
            {
                MPI_Comm team = _subcomms->Subcomm( _level );
                mpi::Broadcast( z.Buffer(), DF.rank, 0, team );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorNaiveBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorNaiveBroadcasts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorNaiveBroadcasts
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            if( DF.rank != 0 )
            {
                MPI_Comm team = _subcomms->Subcomm( _level );
                mpi::Broadcast( z.Buffer(), DF.rank, 0, team );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorNaiveBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorNaiveBroadcasts( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPostcompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPostcompute");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inTargetTeam )
        {
            const Node& node = *_block.data.N;
            typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyVectorPostcompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // yLocal += ULocal z
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            blas::Gemv
            ( 'N', DF.ULocal.Height(), DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         z.LockedBuffer(),         1,
              (Scalar)1, yLocal.Buffer(),          1 );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            const Vector<Scalar>& z = *context.block.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localTargetOffset, SF.D.Height() );
            hmat_tools::Multiply( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const Vector<Scalar>& z = *context.block.data.z;
            const int localHeight = Height();
            const Scalar* zBuffer = z.LockedBuffer();
            Scalar* yLocalBuffer = yLocal.Buffer(_localTargetOffset);
            for( int i=0; i<localHeight; ++i )
                yLocalBuffer[i] += zBuffer[i];
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPostcompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPostcompute");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inSourceTeam )
        {
            const Node& node = *_block.data.N;
            typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPostcompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // yLocal += (VLocal^[T/H])^T z
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            if( Conjugated )
            {
                // yLocal += conj(VLocal) z
                hmat_tools::Conjugate( z );
                hmat_tools::Conjugate( yLocal );
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             z.LockedBuffer(),         1,
                  (Scalar)1, yLocal.Buffer(),          1 );
                hmat_tools::Conjugate( yLocal );
            }
            else
            {
                // yLocal += VLocal z
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             z.LockedBuffer(),         1,
                  (Scalar)1, yLocal.Buffer(),          1 );
            }
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Vector<Scalar>& z = *context.block.data.z;

            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                // yLocal += conj(V) z
                hmat_tools::Conjugate( z );
                hmat_tools::Conjugate( yLocalSub );
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
                hmat_tools::Conjugate( yLocalSub );
            }
            else
            {
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            const Vector<Scalar>& z = *context.block.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SD.D.Width() );
            hmat_tools::TransposeMultiply
            ( alpha, SD.D, z, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorPostcompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorPostcompute");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inSourceTeam )
        {
            const Node& node = *_block.data.N;
            typename MultiplyVectorContext::SplitNodeContext& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).AdjointMultiplyVectorPostcompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // yLocal += (VLocal^[T/H])^H z
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            if( Conjugated )
            {
                // yLocal += VLocal z
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             z.LockedBuffer(),         1,
                  (Scalar)1, yLocal.Buffer(),          1 );
            }
            else
            {
                // yLocal += conj(VLocal) z
                hmat_tools::Conjugate( z );
                hmat_tools::Conjugate( yLocal );
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             z.LockedBuffer(),         1,
                  (Scalar)1, yLocal.Buffer(),          1 );
                hmat_tools::Conjugate( yLocal );
            }
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Vector<Scalar>& z = *context.block.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
            }
            else
            {
                // yLocal += conj(V) z
                hmat_tools::Conjugate( z );
                hmat_tools::Conjugate( yLocalSub );
                hmat_tools::Multiply
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
                hmat_tools::Conjugate( yLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            const Vector<Scalar>& z = *context.block.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SD.D.Width() );
            hmat_tools::AdjointMultiply
            ( alpha, SD.D, z, (Scalar)1, yLocalSub );
        }
        break;

    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

