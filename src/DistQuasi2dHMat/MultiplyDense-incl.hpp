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
    MultiplyDenseInitialize( context, XLocal.Width() );
    MultiplyDensePrecompute( context, alpha, XLocal, YLocal );

    MultiplyDenseSums( context );
    MultiplyDensePassData( context );
    MultiplyDenseBroadcasts( context );

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

    MultiplyDenseContext context;
    TransposeMultiplyDenseInitialize( context, XLocal.Width() );
    TransposeMultiplyDensePrecompute( context, alpha, XLocal, YLocal );

    TransposeMultiplyDenseSums( context );
    TransposeMultiplyDensePassData( context, XLocal );
    TransposeMultiplyDenseBroadcasts( context );

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

    MultiplyDenseContext context;
    AdjointMultiplyDenseInitialize( context, XLocal.Width() );
    AdjointMultiplyDensePrecompute( context, alpha, XLocal, YLocal );

    AdjointMultiplyDenseSums( context );
    AdjointMultiplyDensePassData( context, XLocal );
    AdjointMultiplyDenseBroadcasts( context );

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
( MultiplyDenseContext& context, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseInitialize");
#endif
    context.Clear();
    context.numRhs = numRhs;
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
                ( nodeContext.Child(t,s), numRhs );
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
                ( nodeContext.Child(t,s), numRhs );
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
( MultiplyDenseContext& context, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyDenseInitialize( context, numRhs );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseInitialize
( MultiplyDenseContext& context, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyDenseInitialize( context, numRhs );
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
    const int numRhs = context.numRhs;
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
            Z.Resize( DF.rank, numRhs );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', DF.rank, numRhs, DF.VLocal.Height(), 
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
            ( XLocal, _localSourceOffset, 0, SF.D.Height(), numRhs );
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
        XLocalSub.LockedView
        ( XLocal, _localSourceOffset, 0, F.Width(), numRhs );
        YLocalSub.View( YLocal, _localTargetOffset, 0, F.Height(), numRhs );
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
            ( XLocal, _localSourceOffset, 0, Width(), numRhs );
            hmat_tools::Multiply( alpha, SD.D, XLocalSub, Z );
        }
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *_block.data.D;
        Dense<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localSourceOffset, 0, D.Width(), numRhs );
        YLocalSub.View( YLocal, _localTargetOffset, 0, D.Height(), numRhs );
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
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePrecompute");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MultiplyDenseContext::DistNode& nodeContext = 
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
        typename MultiplyDenseContext::SplitNode& nodeContext = 
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
            Z.Resize( DF.rank, numRhs );
            blas::Gemm
            ( 'T', 'N', DF.rank, numRhs, DF.ULocal.Height(),
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
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), numRhs );
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
        ( XLocal, _localTargetOffset, 0, F.Height(), numRhs );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), numRhs );
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
        ( XLocal, _localTargetOffset, 0, D.Height(), numRhs );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, D.Width(), numRhs );
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
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDensePrecompute");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MultiplyDenseContext::DistNode& nodeContext = 
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
        typename MultiplyDenseContext::SplitNode& nodeContext = 
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
            Z.Resize( DF.rank, numRhs );
            blas::Gemm
            ( 'C', 'N', DF.rank, numRhs, DF.ULocal.Height(), 
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
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), numRhs );
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
        ( XLocal, _localTargetOffset, 0, F.Height(), numRhs );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), numRhs );
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
        ( XLocal, _localTargetOffset, 0, D.Height(), numRhs );
        YLocalSub.View( YLocal, _localSourceOffset, 0, D.Width(), numRhs );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSums
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSums");
#endif
    // Compute the message sizes for each reduce 
    const int numLevels = _teams->NumLevels();
    const int numReduces = numLevels-1;
    if( numReduces == 0 )
        return;
    std::vector<int> sizes( numReduces, 0 );
    MultiplyDenseSumsCount( sizes, context.numRhs );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    if( totalSize == 0 )
        return;
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    MultiplyDenseSumsPack( context, buffer, offsets );

    // Reset the offsets vector and then perform the reduces. 
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    _teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of subcommunicators have data)
    MultiplyDenseSumsUnpack( context, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSums
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSums");
#endif
    // Compute the message sizes for each reduce 
    const int numLevels = _teams->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMultiplyDenseSumsCount( sizes, context.numRhs );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    TransposeMultiplyDenseSumsPack( context, buffer, offsets );

    // Reset the offsets vector and then perform the reduces. 
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    _teams->TreeSumToRoots( buffer, sizes, offsets );

    // Unpack the reduced buffers (only roots of subcommunicators have data)
    TransposeMultiplyDenseSumsUnpack( context, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseSums
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseSums");
#endif
    // This unconjugated version is identical
    TransposeMultiplyDenseSums( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSumsCount
( std::vector<int>& sizes, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSumsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        // We can avoid passing the child contexts because the data we 
        // want is invariant
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseSumsCount( sizes, numRhs );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level] += _block.data.DF->rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSumsCount
( std::vector<int>& sizes, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSumsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseSumsCount
                ( sizes, numRhs );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level] += _block.data.DF->rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSumsPack
( const MultiplyDenseContext& context, 
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSumsPack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseSumsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Dense<Scalar>& Z = *context.block.data.Z;
            if( Z.Height() == Z.LDim() )
                std::memcpy
                ( &buffer[offsets[_level]], Z.LockedBuffer(), 
                  DF.rank*numRhs*sizeof(Scalar) );
            else
                for( int j=0; j<numRhs; ++j )
                    std::memcpy
                    ( &buffer[offsets[_level]+j*DF.rank], Z.LockedBuffer(0,j),
                      DF.rank*sizeof(Scalar) );
            offsets[_level] += DF.rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSumsPack
( const MultiplyDenseContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSumsPack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseSumsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Dense<Scalar>& Z = *context.block.data.Z;
            if( Z.Height() == Z.LDim() )
                std::memcpy
                ( &buffer[offsets[_level]], Z.LockedBuffer(), 
                  DF.rank*numRhs*sizeof(Scalar) );
            else
                for( int j=0; j<numRhs; ++j )
                    std::memcpy
                    ( &buffer[offsets[_level]+j*DF.rank], Z.LockedBuffer(0,j),
                      DF.rank*sizeof(Scalar) );
            offsets[_level] += DF.rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseSumsUnpack
( MultiplyDenseContext& context, 
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseSumsUnpack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseSumsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( Z.Height() == Z.LDim() )
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_level]], 
                      DF.rank*numRhs*sizeof(Scalar) );
                else
                    for( int j=0; j<numRhs; ++j )
                        std::memcpy
                        ( Z.Buffer(0,j), &buffer[offsets[_level]+j*DF.rank],
                          DF.rank*sizeof(Scalar) );
                offsets[_level] += DF.rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseSumsUnpack
( MultiplyDenseContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseSumsUnpack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseSumsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                if( Z.Height() == Z.LDim() )
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_level]], 
                      DF.rank*numRhs*sizeof(Scalar) );
                else
                    for( int j=0; j<numRhs; ++j )
                        std::memcpy
                        ( Z.Buffer(0,j), &buffer[offsets[_level]+j*DF.rank],
                          DF.rank*sizeof(Scalar) );
                offsets[_level] += DF.rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePassData
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassData");
#endif
    // Constuct maps of the send/recv processes to the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyDensePassDataCount( sendSizes, recvSizes, context.numRhs );

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
    MultiplyDensePassDataPack( context, sendBuffer, offsets );

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
    MultiplyDensePassDataUnpack( context, recvBuffer, recvOffsets );
    
    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassDataCount");
#endif
    if( numRhs == 0 )
        return;

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataCount
                ( sendSizes, recvSizes, numRhs );
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
                AddToMap( sendSizes, _targetRoot, DF.rank*numRhs );
            else
                AddToMap( recvSizes, _sourceRoot, DF.rank*numRhs );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        if( _inSourceTeam )
            AddToMap( sendSizes, _targetRoot, SF.rank*numRhs );
        else
            AddToMap( recvSizes, _sourceRoot, SF.rank*numRhs );
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
            AddToMap( sendSizes, _targetRoot, Height()*numRhs );
        else
            AddToMap( recvSizes, _sourceRoot, Height()*numRhs );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePassDataPack
( MultiplyDenseContext& context,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassDataPack");
#endif
    if( context.numRhs == 0 )
        return;

    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataPack
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
                    Dense<Scalar>& Z = *context.block.data.Z;
                    std::memcpy
                    ( &buffer[offsets[_targetRoot]], Z.LockedBuffer(),
                      Z.Height()*Z.Width()*sizeof(Scalar) );
                    offsets[_targetRoot] += Z.Height()*Z.Width();
                    Z.Clear();
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
                Dense<Scalar>& Z = *context.block.data.Z;
                std::memcpy
                ( &buffer[offsets[_targetRoot]], Z.LockedBuffer(),
                  Z.Height()*Z.Width()*sizeof(Scalar) );
                offsets[_targetRoot] += Z.Height()*Z.Width();
                Z.Clear();
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inSourceTeam )
        {
            if( Height() != 0 )
            {
                Dense<Scalar>& Z = *context.block.data.Z;
                std::memcpy
                ( &buffer[offsets[_targetRoot]], Z.LockedBuffer(),
                  Z.Height()*Z.Width()*sizeof(Scalar) );
                offsets[_targetRoot] += Z.Height()*Z.Width();
                Z.Clear();
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePassDataUnpack
( MultiplyDenseContext& context,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePassDataUnpack");
#endif
    const int numRhs = context.numRhs;
    if( numRhs == 0 )
        return;

    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDensePassDataUnpack
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
                    Dense<Scalar>& Z = *context.block.data.Z;
                    Z.Resize( DF.rank, numRhs, DF.rank );
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_sourceRoot]],
                      DF.rank*numRhs*sizeof(Scalar) );
                    offsets[_sourceRoot] += DF.rank*numRhs;
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
                Dense<Scalar>& Z = *context.block.data.Z;
                Z.Resize( SF.rank, numRhs, SF.rank );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_sourceRoot]],
                  SF.rank*numRhs*sizeof(Scalar) );
                offsets[_sourceRoot] += SF.rank*numRhs;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
        {
            if( Height() != 0 )
            {
                Dense<Scalar>& Z = *context.block.data.Z;
                Z.Resize( Height(), numRhs, Height() );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_sourceRoot]],
                  Z.Height()*numRhs*sizeof(Scalar) );
                offsets[_sourceRoot] += Z.Height()*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePassData
( MultiplyDenseContext& context, const Dense<Scalar>& XLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePassData");
#endif
    // Constuct maps of the send/recv processes to the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    TransposeMultiplyDensePassDataCount( sendSizes, recvSizes, context.numRhs );

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
    TransposeMultiplyDensePassDataPack( context, XLocal, sendBuffer, offsets );

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
    TransposeMultiplyDensePassDataUnpack( context, recvBuffer, recvOffsets );
    
    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes, int numRhs ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePassDataCount");
#endif
    if( numRhs == 0 )
        return;

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataCount
                ( sendSizes, recvSizes, numRhs );
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
                AddToMap( sendSizes, _sourceRoot, DF.rank*numRhs );
            else
                AddToMap( recvSizes, _targetRoot, DF.rank*numRhs );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        if( _inTargetTeam )
            AddToMap( sendSizes, _sourceRoot, SF.rank*numRhs );
        else
            AddToMap( recvSizes, _targetRoot, SF.rank*numRhs );
        break;
    }
    case SPLIT_DENSE:
    {
        if( _inTargetTeam )
            AddToMap( sendSizes, _sourceRoot, Height()*numRhs );
        else
            AddToMap( recvSizes, _targetRoot, Height()*numRhs );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePassDataPack
( MultiplyDenseContext& context, const Dense<Scalar>& XLocal,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePassDataPack");
#endif
    const int numRhs = context.numRhs;
    if( numRhs == 0 )
        return;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataPack
                ( nodeContext.Child(t,s), XLocal, buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataPack
                ( nodeContext.Child(t,s), XLocal, buffer, offsets );
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
                    Dense<Scalar>& Z = *context.block.data.Z;
                    std::memcpy
                    ( &buffer[offsets[_sourceRoot]], Z.LockedBuffer(),
                      Z.Height()*Z.Width()*sizeof(Scalar) );
                    offsets[_sourceRoot] += Z.Height()*Z.Width();
                    Z.Clear();
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
                Dense<Scalar>& Z = *context.block.data.Z;
                std::memcpy
                ( &buffer[offsets[_sourceRoot]], Z.LockedBuffer(),
                  Z.Height()*Z.Width()*sizeof(Scalar) );
                offsets[_sourceRoot] += Z.Height()*Z.Width();
                Z.Clear();
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
                if( XLocal.LDim() != height )
                {
                    Scalar* start = &buffer[offsets[_sourceRoot]];
                    for( int j=0; j<numRhs; ++j )
                    {
                        std::memcpy
                        ( &start[height*j], 
                          XLocal.LockedBuffer(_localTargetOffset,j),
                          height*sizeof(Scalar) );
                    }
                }
                else
                {
                    std::memcpy
                    ( &buffer[offsets[_sourceRoot]],
                      XLocal.LockedBuffer(_localTargetOffset,0),
                      height*numRhs*sizeof(Scalar) );
                }
                offsets[_sourceRoot] += height*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDensePassDataUnpack
( MultiplyDenseContext& context, 
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePassDataUnpack");
#endif
    const int numRhs = context.numRhs;
    if( numRhs == 0 )
        return;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDensePassDataUnpack
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
                    Dense<Scalar>& Z = *context.block.data.Z;
                    Z.Resize( DF.rank, numRhs, DF.rank );
                    std::memcpy
                    ( Z.Buffer(), &buffer[offsets[_targetRoot]],
                      DF.rank*numRhs*sizeof(Scalar) );
                    offsets[_targetRoot] += DF.rank*numRhs;
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
                Dense<Scalar>& Z = *context.block.data.Z;
                Z.Resize( SF.rank, numRhs, SF.rank );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_targetRoot]],
                  SF.rank*numRhs*sizeof(Scalar) );
                offsets[_targetRoot] += SF.rank*numRhs;
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
                Dense<Scalar>& Z = *context.block.data.Z;
                Z.Resize( height, numRhs, height );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_targetRoot]],
                  height*numRhs*sizeof(Scalar) );
                offsets[_targetRoot] += height*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDensePassData
( MultiplyDenseContext& context, const Dense<Scalar>& XLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDensePassData");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyDensePassData( context, XLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseBroadcasts
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyDenseBroadcastsCount( sizes, context.numRhs );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    MultiplyDenseBroadcastsPack( context, buffer, offsets );

    // Reset the offsets vector and then perform the broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    _teams->TreeBroadcasts( buffer, sizes, offsets );

    // Unpack the broadcasted buffers 
    MultiplyDenseBroadcastsUnpack( context, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::TransposeMultiplyDenseBroadcasts
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMultiplyDenseBroadcastsCount( sizes, context.numRhs );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    TransposeMultiplyDenseBroadcastsPack( context, buffer, offsets );

    // Reset the offsets vector and then perform the broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    _teams->TreeBroadcasts( buffer, sizes, offsets );

    // Unpack the broadcasted buffers 
    TransposeMultiplyDenseBroadcastsUnpack( context, buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointMultiplyDenseBroadcasts
( MultiplyDenseContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDenseBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyDenseBroadcasts( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseBroadcastsCount
( std::vector<int>& sizes, int numRhs ) const
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
                node.Child(t,s).MultiplyDenseBroadcastsCount( sizes, numRhs );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level] += _block.data.DF->rank*numRhs;
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
( std::vector<int>& sizes, int numRhs ) const
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
                ( sizes, numRhs );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level] += _block.data.DF->rank*numRhs;
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
( const MultiplyDenseContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcastsPack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyDenseBroadcastsPack
                ( nodeContext.Child(t,s), buffer, offsets ); 
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                const DistLowRank& DF = *_block.data.DF;
                const Dense<Scalar>& Z = *context.block.data.Z;
                std::memcpy
                ( &buffer[offsets[_level]], Z.LockedBuffer(), 
                  DF.rank*numRhs*sizeof(Scalar) );
                offsets[_level] += DF.rank*numRhs;
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
( const MultiplyDenseContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcastsPack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseBroadcastsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            MPI_Comm team = _teams->Team( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                const DistLowRank& DF = *_block.data.DF;
                const Dense<Scalar>& Z = *context.block.data.Z;
                std::memcpy
                ( &buffer[offsets[_level]], Z.LockedBuffer(), 
                  DF.rank*numRhs*sizeof(Scalar) );
                offsets[_level] += DF.rank*numRhs;
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
( MultiplyDenseContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseBroadcastsUnpack");
#endif
    const int numRhs = context.numRhs;
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
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( DF.rank != 0 )
            {
                Z.Resize( DF.rank, numRhs, DF.rank );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_level]], 
                  DF.rank*numRhs*sizeof(Scalar) );
                offsets[_level] += DF.rank*numRhs;
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
TransposeMultiplyDenseBroadcastsUnpack
( MultiplyDenseContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDenseBroadcastsUnpack");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyDenseBroadcastsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense<Scalar>& Z = *context.block.data.Z;
            if( DF.rank != 0 )
            {
                Z.Resize( DF.rank, numRhs, DF.rank );
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_level]], 
                  DF.rank*numRhs*sizeof(Scalar) );
                offsets[_level] += DF.rank*numRhs;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDensePostcompute
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDensePostcompute");
#endif
    const int numRhs = context.numRhs;
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
            ( 'N', 'N', DF.ULocal.Height(), numRhs, DF.rank,
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
            ( YLocal, _localTargetOffset, 0, SF.D.Height(), numRhs );
            hmat_tools::Multiply( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const Dense<Scalar>& Z = *context.block.data.Z;
            const int localHeight = Height();
            for( int j=0; j<numRhs; ++j )
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
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal, 
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeMultiplyDensePostcompute");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
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
            typename MultiplyDenseContext::SplitNode& nodeContext =
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
                ( 'N', 'N', DF.VLocal.Height(), numRhs, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmat_tools::Conjugate( YLocal );
            }
            else
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), numRhs, DF.rank,
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
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), numRhs );
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
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), numRhs );
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
( MultiplyDenseContext& context,
  Scalar alpha, const Dense<Scalar>& XLocal,
                      Dense<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointMultiplyDensePostcompute");
#endif
    const int numRhs = context.numRhs;
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MultiplyDenseContext::DistNode& nodeContext = 
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
            typename MultiplyDenseContext::SplitNode& nodeContext =
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
                ( 'N', 'N', DF.VLocal.Height(), numRhs, DF.rank,
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
                ( 'N', 'N', DF.VLocal.Height(), numRhs, DF.rank,
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
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), numRhs );
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
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), numRhs );
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

