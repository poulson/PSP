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

    MultiplyVectorSums( context );
    MultiplyVectorPassData( context );
    MultiplyVectorBroadcasts( context );

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

    TransposeMultiplyVectorSums( context );
    TransposeMultiplyVectorPassData( context, xLocal );
    TransposeMultiplyVectorBroadcasts( context );

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

    AdjointMultiplyVectorSums( context );
    AdjointMultiplyVectorPassData( context, xLocal );
    AdjointMultiplyVectorBroadcasts( context );

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
        context.block.data.DN = new typename MultiplyVectorContext::DistNode;

        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_NODE:
    {
        context.block.type = SPLIT_NODE;
        context.block.data.SN = new typename MultiplyVectorContext::SplitNode;

        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorInitialize
                ( nodeContext.Child(t,s) );
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
        typename MultiplyVectorContext::DistNode& nodeContext = 
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
       typename MultiplyVectorContext::SplitNode& nodeContext = 
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
        typename MultiplyVectorContext::DistNode& nodeContext = 
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
        typename MultiplyVectorContext::SplitNode& nodeContext = 
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
        typename MultiplyVectorContext::DistNode& nodeContext = 
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
        typename MultiplyVectorContext::SplitNode& nodeContext = 
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSums
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSums");
#endif
    // Compute the message sizes for each reduce 
    const int numLevels = _teams->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MultiplyVectorSumsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    MultiplyVectorSumsPack( context, buffer, offsetsCopy );

    // Perform the reduces with log2(p) messages
    _teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of teamunicators have data)
    MultiplyVectorSumsUnpack( context, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSums
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSums");
#endif
    // Compute the message sizes for each reduce 
    const int numLevels = _teams->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMultiplyVectorSumsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    TransposeMultiplyVectorSumsPack( context, buffer, offsetsCopy );

    // Perform the reduces with log2(p) messages
    _teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of teamunicators have data)
    TransposeMultiplyVectorSumsUnpack( context, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorSums
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorSums");
#endif
    // This unconjugated version is identical
    TransposeMultiplyVectorSums( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSumsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSumsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSumsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level] += _block.data.DF->rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSumsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSumsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSumsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level] += _block.data.DF->rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSumsPack
( const MultiplyVectorContext& context, 
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSumsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSumsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            std::memcpy
            ( &buffer[offsets[_level]], z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSumsPack
( const MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSumsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSumsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            std::memcpy
            ( &buffer[offsets[_level]], z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorSumsUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorSumsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSumsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_level]], 
                  DF.rank*sizeof(Scalar) );
                offsets[_level] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorSumsUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorSumsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSumsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_level]], 
                  DF.rank*sizeof(Scalar) );
                offsets[_level] += DF.rank;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassData
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassData");
#endif
    // Constuct maps of the send/recv processes to the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyVectorPassDataCount( sendSizes, recvSizes );

    // Fill the offset vectors defined by the sizes
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
    MultiplyVectorPassDataPack( context, sendBuffer, offsets );

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
    MultiplyVectorPassDataUnpack( context, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassDataCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataCount
                ( sendSizes, recvSizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inSourceTeam )
                AddToMap( sendSizes, _targetRoot, DF.rank );
            else
                AddToMap( recvSizes, _sourceRoot, DF.rank );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        if( _inSourceTeam )
            AddToMap( sendSizes, _targetRoot, SF.rank );
        else
            AddToMap( recvSizes, _sourceRoot, SF.rank );
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
            AddToMap( sendSizes, _targetRoot, Height() );
        else
            AddToMap( recvSizes, _sourceRoot, Height() );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassDataPack
( MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassDataPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                MPI_Comm team = _teams->Team( _level );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                {
                    Vector<Scalar>& z = *context.block.data.z;
                    std::memcpy
                    ( &buffer[offsets[_targetRoot]], z.LockedBuffer(),
                      DF.rank*sizeof(Scalar) );
                    offsets[_targetRoot] += DF.rank;
                    z.Clear();
                }
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            if( SF.rank != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                std::memcpy
                ( &buffer[offsets[_targetRoot]], z.LockedBuffer(),
                  SF.rank*sizeof(Scalar) );
                offsets[_targetRoot] += SF.rank;
                z.Clear();
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
        {
            const int height = Height();
            if( height != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                std::memcpy
                ( &buffer[offsets[_targetRoot]], z.LockedBuffer(),
                  height*sizeof(Scalar) );
                offsets[_targetRoot] += height;
                z.Clear();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorPassDataUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorPassDataUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                MPI_Comm team = _teams->Team( _level );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                {
                    Vector<Scalar>& z = *context.block.data.z;
                    z.Resize( DF.rank );
                    std::memcpy
                    ( z.Buffer(), &buffer[offsets[_sourceRoot]],
                      DF.rank*sizeof(Scalar) );
                    offsets[_sourceRoot] += DF.rank;
                }
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            if( SF.rank != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                z.Resize( SF.rank );
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_sourceRoot]],
                  SF.rank*sizeof(Scalar) );
                offsets[_sourceRoot] += SF.rank;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            const int height = Height();
            if( height != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                z.Resize( height );
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_sourceRoot]],
                  height*sizeof(Scalar) );
                offsets[_sourceRoot] += height;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPassData
( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPassData");
#endif
    // Constuct maps of the send/recv processes to the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    TransposeMultiplyVectorPassDataCount( sendSizes, recvSizes );

    // Fill the offset vectors defined by the sizes
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
    TransposeMultiplyVectorPassDataPack( context, xLocal, sendBuffer, offsets );

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
    TransposeMultiplyVectorPassDataUnpack( context, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPassDataCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataCount
                ( sendSizes, recvSizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inTargetTeam )
                AddToMap( sendSizes, _sourceRoot, DF.rank );
            else
                AddToMap( recvSizes, _targetRoot, DF.rank );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        if( _inTargetTeam )
            AddToMap( sendSizes, _sourceRoot, SF.rank );
        else
            AddToMap( recvSizes, _targetRoot, SF.rank );
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
            AddToMap( sendSizes, _sourceRoot, Height() );
        else
            AddToMap( recvSizes, _targetRoot, Height() );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPassDataPack
( MultiplyVectorContext& context, const Vector<Scalar>& xLocal,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPassDataPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), xLocal, buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), xLocal, buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                MPI_Comm team = _teams->Team( _level );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                {
                    Vector<Scalar>& z = *context.block.data.z;
                    std::memcpy
                    ( &buffer[offsets[_sourceRoot]], z.LockedBuffer(),
                      DF.rank*sizeof(Scalar) );
                    offsets[_sourceRoot] += DF.rank;
                    z.Clear();
                }
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( _inTargetTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            if( SF.rank != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                std::memcpy
                ( &buffer[offsets[_sourceRoot]], z.LockedBuffer(),
                  SF.rank*sizeof(Scalar) );
                offsets[_sourceRoot] += SF.rank;
                z.Clear();
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            const int height = Height();
            if( height != 0 )
            {
                std::memcpy
                ( &buffer[offsets[_sourceRoot]],
                  xLocal.LockedBuffer(_localTargetOffset),
                  height*sizeof(Scalar) );
                offsets[_sourceRoot] += height;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyVectorPassDataUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorPassDataUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        if( _inSourceTeam )
       {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                MPI_Comm team = _teams->Team( _level );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                {
                    Vector<Scalar>& z = *context.block.data.z;
                    z.Resize( DF.rank );
                    std::memcpy
                    ( z.Buffer(), &buffer[offsets[_targetRoot]],
                      DF.rank*sizeof(Scalar) );
                    offsets[_targetRoot] += DF.rank;
                }
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            if( SF.rank != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                z.Resize( SF.rank );
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_targetRoot]],
                  SF.rank*sizeof(Scalar) );
                offsets[_targetRoot] += SF.rank;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
        {
            const int height = Height();
            if( height != 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                z.Resize( height );
                std::memcpy
                ( z.Buffer(), &buffer[offsets[_targetRoot]],
                  height*sizeof(Scalar) );
                offsets[_targetRoot] += height;
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

// HERE

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyVectorPassData
( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyVectorPassData");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorPassData( context, xLocal );
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
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyVectorBroadcastsCount( sizes );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of teamunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    MultiplyVectorBroadcastsPack( context, buffer, offsetsCopy );

    // Perform the broadcasts with log2(p) messages
    _teams->TreeBroadcasts( buffer, sizes, offsets );

    // Unpack the broadcasted buffers 
    MultiplyVectorBroadcastsUnpack( context, buffer, offsets );
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
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMultiplyVectorBroadcastsCount( sizes );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of teamunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    TransposeMultiplyVectorBroadcastsPack( context, buffer, offsetsCopy );

    // Perform the broadcasts with log2(p) messages
    _teams->TreeBroadcasts( buffer, sizes, offsets );

    // Unpack the broadcasted buffers 
    TransposeMultiplyVectorBroadcastsUnpack( context, buffer, offsets );
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
            sizes[_level] += _block.data.DF->rank;
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
            sizes[_level] += _block.data.DF->rank;
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
( const MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level]], z.LockedBuffer(), 
                  DF.rank*sizeof(Scalar) );
                offsets[_level] += DF.rank;
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
( const MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Vector<Scalar>& z = *context.block.data.z;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level]], z.LockedBuffer(), 
                  DF.rank*sizeof(Scalar) );
                offsets[_level] += DF.rank;
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
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            std::memcpy
            ( z.Buffer(), &buffer[offsets[_level]], DF.rank*sizeof(Scalar) );
            offsets[_level] += DF.rank;
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
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyVectorBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( DF.rank );
            std::memcpy
            ( z.Buffer(), &buffer[offsets[_level]], DF.rank*sizeof(Scalar) );
            offsets[_level] += DF.rank;
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
        typename MultiplyVectorContext::DistNode& nodeContext = 
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
            typename MultiplyVectorContext::SplitNode& nodeContext = 
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
        typename MultiplyVectorContext::DistNode& nodeContext = 
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
            typename MultiplyVectorContext::SplitNode& nodeContext = 
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
        typename MultiplyVectorContext::DistNode& nodeContext = 
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
            typename MultiplyVectorContext::SplitNode& nodeContext = 
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

