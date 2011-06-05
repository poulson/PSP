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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrix");
#endif
    YLocal.Resize( LocalHeight(), XLocal.Width() );
    MapMatrix( alpha, XLocal, (Scalar)0, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrix");
#endif
    YLocal.Resize( LocalWidth(), XLocal.Width() );
    TransposeMapMatrix( alpha, XLocal, (Scalar)0, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrix");
#endif
    YLocal.Resize( LocalWidth(), XLocal.Width() );
    HermitianTransposeMapMatrix( alpha, XLocal, (Scalar)0, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
  Scalar beta,        DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrix");
#endif
    RequireRoot();
    if( XLocal.Height() == 0 || YLocal.Height() == 0 || YLocal.Width() == 0 )
        return;
    hmatrix_tools::Scale( beta, YLocal );

    MapDenseMatrixContext context;
    MapDenseMatrixInitialize( context );
    MapDenseMatrixPrecompute( context, alpha, XLocal, YLocal );

    MapDenseMatrixSummations( context, XLocal.Width() );
    //MapDenseMatrixNaiveSummations( context, XLocal.Width() );

    MapDenseMatrixPassData( context, alpha, XLocal, YLocal );

    MapDenseMatrixBroadcasts( context, XLocal.Width() );
    //MapDenseMatrixNaiveBroadcasts( context, XLocal.Width() );

    MapDenseMatrixPostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
  Scalar beta,        DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrix");
#endif
    RequireRoot();
    if( XLocal.Height() == 0 || YLocal.Height() == 0 || YLocal.Width() == 0 )
        return;
    hmatrix_tools::Scale( beta, YLocal );

    MapDenseMatrixContext context;
    TransposeMapDenseMatrixInitialize( context );
    TransposeMapDenseMatrixPrecompute( context, alpha, XLocal, YLocal );

    TransposeMapDenseMatrixSummations( context, XLocal.Width() );
    //TransposeMapDenseMatrixNaiveSummations( context, XLocal.Width() );

    TransposeMapDenseMatrixPassData( context, alpha, XLocal, YLocal );

    TransposeMapDenseMatrixBroadcasts( context, XLocal.Width() );
    //TransposeMapDenseMatrixNaiveBroadcasts( context, XLocal.Width() );

    TransposeMapDenseMatrixPostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
  Scalar beta,        DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrix");
#endif
    RequireRoot();
    hmatrix_tools::Scale( beta, YLocal );
    if( XLocal.Height() == 0 || YLocal.Height() == 0 || YLocal.Width() == 0 )
        return;

    MapDenseMatrixContext context;
    HermitianTransposeMapDenseMatrixInitialize( context );
    HermitianTransposeMapDenseMatrixPrecompute
    ( context, alpha, XLocal, YLocal );

    HermitianTransposeMapDenseMatrixSummations( context, XLocal.Width() );
    //HermitianTransposeMapDenseMatrixNaiveSummations
    //( context, XLocal.Width() );

    HermitianTransposeMapDenseMatrixPassData( context, alpha, XLocal, YLocal );

    HermitianTransposeMapDenseMatrixBroadcasts( context, XLocal.Width() );
    //HermitianTransposeMapDenseMatrixNaiveBroadcasts
    //( context, XLocal.Width() );

    HermitianTransposeMapDenseMatrixPostcompute
    ( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixInitialize
( MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixInitialize");
#endif
    context.Clear();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        context.block.type = DIST_NODE;
        context.block.data.DN = 
            new typename MapDenseMatrixContext::DistNodeContext();

        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_NODE:
    {
        context.block.type = SPLIT_NODE;
        context.block.data.SN = 
            new typename MapDenseMatrixContext::SplitNodeContext();

        typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        context.block.type = DIST_LOW_RANK;
        context.block.data.Z = new Dense;
        break;
    case SPLIT_LOW_RANK:
        context.block.type = SPLIT_LOW_RANK;
        context.block.data.Z = new Dense;
        break;
    case SPLIT_DENSE:
        context.block.type = SPLIT_DENSE;
        context.block.data.Z = new Dense;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapDenseMatrixInitialize
( MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixInitialize");
#endif
    // The non-transposed initialization is identical
    MapDenseMatrixInitialize( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixInitialize
( MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixInitialize");
#endif
    // The non-transposed initialization is identical
    MapDenseMatrixInitialize( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixPrecompute
                ( context, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // Form Z := alpha VLocal^[T/H] XLocal
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
            Dense& Z = *context.block.data.Z;

            Dense XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( alpha, SF.D, XLocalSub, Z );
            else
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRank& F = *_block.data.F;
        Dense XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, F.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, F.Height(), width );
        hmatrix_tools::MatrixMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            Dense& Z = *context.block.data.Z;

            Dense XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, Width(), width );
            hmatrix_tools::MatrixMatrix( alpha, SD.D, XLocalSub, Z );
        }
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense& D = *_block.data.D;
        Dense XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, D.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, D.Height(), width );
        hmatrix_tools::MatrixMatrix
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapDenseMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixPrecompute
                ( context, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^T XLocal
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
            Dense& Z = *context.block.data.Z;
            Dense XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixTransposeMatrix( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRank& F = *_block.data.F;
        Dense XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense& D = *_block.data.D;
        Dense XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmatrix_tools::MatrixTransposeMatrix
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapDenseMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
    {
        typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapDenseMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapDenseMatrixPrecompute
                ( context, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^H XLocal
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
            Dense& Z = *context.block.data.Z;

            Dense XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRank& F = *_block.data.F;
        Dense XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense& D = *_block.data.D;
        Dense XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmatrix_tools::MatrixHermitianTransposeMatrix
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MapDenseMatrixSummationsCount( sizes, width );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapDenseMatrixSummationsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
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
    MapDenseMatrixSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapDenseMatrixSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMapDenseMatrixSummationsCount( sizes, width );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    TransposeMapDenseMatrixSummationsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the reduces. There should be
    // at most log_4(p) reduces.
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
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
    TransposeMapDenseMatrixSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixSummations");
#endif
    // This unconjugated version is identical
    TransposeMapDenseMatrixSummations( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixSummationsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixSummationsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixSummationsCount
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixSummationsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixSummationsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixSummationsCount
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixSummationsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixSummationsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixSummationsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::TransposeMapDenseMatrixSummationsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixNaiveSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixNaiveSummations");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixNaiveSummations
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixNaiveSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixNaiveSummations");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixNaiveSummations
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            if( DF.rank != 0 )
            {
                Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixNaiveSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMapDenseMatrixNaiveSummations( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixPassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixPassData");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
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
                        node.Child(t,s).MapDenseMatrixPassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                for( int t=2; t<4; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).MapDenseMatrixPassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    node.Child(t,s).MapDenseMatrixPassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            // Bottom-left quadrant
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    node.Child(t,s).MapDenseMatrixPassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapDenseMatrixPassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).MapDenseMatrixPassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MapDenseMatrixPassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).MapDenseMatrixPassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).MapDenseMatrixPassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).MapDenseMatrixPassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).MapDenseMatrixPassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapDenseMatrixPassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).MapDenseMatrixPassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MapDenseMatrixPassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).MapDenseMatrixPassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).MapDenseMatrixPassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).MapDenseMatrixPassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).MapDenseMatrixPassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapDenseMatrixPassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).MapDenseMatrixPassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MapDenseMatrixPassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).MapDenseMatrixPassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).MapDenseMatrixPassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).MapDenseMatrixPassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).MapDenseMatrixPassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapDenseMatrixPassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).MapDenseMatrixPassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MapDenseMatrixPassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).MapDenseMatrixPassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).MapDenseMatrixPassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).MapDenseMatrixPassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).MapDenseMatrixPassData
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
        typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;

        std::size_t bufferSize = 0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPassDataSplitNodeCount( bufferSize );
        bufferSize *= width;

        std::vector<byte> buffer( bufferSize );
        MPI_Comm comm = _subcomms->Subcomm(0);
        if( _inSourceTeam )
        {
            byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MapDenseMatrixPassDataSplitNodePack
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
                    node.Child(t,s).MapDenseMatrixPassDataSplitNodeUnpack
                    ( head, nodeContext.Child(t,s), width );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        Dense& Z = *context.block.data.Z;
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
        Dense& Z = *context.block.data.Z;
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
        Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixPassDataSplitNodePack
( byte*& head, const MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixPassDataSplitNodePack");
    if( !_inSourceTeam )
        throw std::logic_error("Calling process should be in source team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MapDenseMatrixContext::DistNodeContext& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixPassDataSplitNodePack
                ( head, nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const Dense& Z = *context.block.data.Z;
        Write( head, Z.LockedBuffer(), Z.Height()*Z.Width() );
        break;
    }
    case SPLIT_DENSE:
    {
        const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
MapDenseMatrixPassDataSplitNodeUnpack
( const byte*& head, MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixPassDataSplitNodeUnpack");
    if( !_inTargetTeam )
        throw std::logic_error("Calling process should be in target team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixPassDataSplitNodeUnpack
                ( head, nodeContext.Child(t,s), width );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Dense& Z = *context.block.data.Z;
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
        Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapDenseMatrixPassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixPassData");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
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
                        node.Child(t,s).TransposeMapDenseMatrixPassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                for( int t=2; t<4; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).TransposeMapDenseMatrixPassData
                        ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    node.Child(t,s).TransposeMapDenseMatrixPassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
            // Bottom-left quadrant
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    node.Child(t,s).TransposeMapDenseMatrixPassData
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal ); 
                node.Child(3,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMapDenseMatrixPassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapDenseMatrixPassData
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
        typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
            *context.block.data.SN;

        std::size_t bufferSize = 0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorPassDataSplitNodeCount
                ( bufferSize );
        bufferSize *= width;

        std::vector<byte> buffer( bufferSize );
        MPI_Comm comm = _subcomms->Subcomm(0);
        if( _inTargetTeam )
        {
            byte* head = &buffer[0];
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMapDenseMatrixPassDataSplitNodePack
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
                    TransposeMapDenseMatrixPassDataSplitNodeUnpack
                    ( head, nodeContext.Child(t,s), width );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;
        const DistLowRank& DF = *_block.data.DF;
        Dense& Z = *context.block.data.Z;
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
        Dense& Z = *context.block.data.Z;
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
        Dense& Z = *context.block.data.Z;
        if( height != 0 )
        {
            MPI_Comm comm = _subcomms->Subcomm( 0 );
            if( _inTargetTeam )
            {
                Dense XLocalSub;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixPassDataSplitNodePack
( byte*& head, const MapDenseMatrixContext& context,
  const DenseMatrix<Scalar>& XLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::TransposeMapDenseMatrixPassDataSplitNodePack");
    if( !_inTargetTeam )
        throw std::logic_error("Calling process should be in target team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        const typename MapDenseMatrixContext::DistNodeContext& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixPassDataSplitNodePack
                ( head, nodeContext.Child(t,s), XLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixPassDataSplitNodeUnpack
( const byte*& head, MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::TransposeMapDenseMatrixPassDataSplitNodeUnpack");
    if( !_inSourceTeam )
        throw std::logic_error("Calling process should be in source team");
#endif
    switch( _block.type )
    {
    case SPLIT_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixPassDataSplitNodeUnpack
                ( head, nodeContext.Child(t,s), width );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *_block.data.SF;
        Dense& Z = *context.block.data.Z;
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
        Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixPassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixPassData");
#endif
    // The unconjugated version should be identical
    TransposeMapDenseMatrixPassData( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MapDenseMatrixBroadcastsCount( sizes, width );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numBroadcasts*sizeof(int) );
    MapDenseMatrixBroadcastsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=offsets[i],++i )
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
    MapDenseMatrixBroadcastsUnpack( buffer, offsets, context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapDenseMatrixBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMapDenseMatrixBroadcastsCount( sizes, width );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of subcommunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numBroadcasts*sizeof(int) );
    TransposeMapDenseMatrixBroadcastsPack( buffer, offsets, context );

    // Reset the offsets vector and then perform the broadcasts. There should be
    // at most log_4(p) broadcasts.
    for( int i=0,offset=0; i<numBroadcasts; offset+=offsets[i],++i )
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
    TransposeMapDenseMatrixBroadcastsUnpack( buffer, offsets, context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapDenseMatrixBroadcasts( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixBroadcastsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixBroadcastsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixBroadcastsCount( sizes, width );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixBroadcastsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixBroadcastsCount");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixBroadcastsCount
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixBroadcastsPack
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
                const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixBroadcastsPack
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
                const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixBroadcastsUnpack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixBroadcastsPack");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixNaiveBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixNaiveBroadcasts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixNaiveBroadcasts
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
TransposeMapDenseMatrixNaiveBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixNaiveBroadcasts");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixNaiveBroadcasts
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixNaiveBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapDenseMatrixNaiveBroadcasts( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapDenseMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapDenseMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inTargetTeam )
        {
            const Node& node = *_block.data.N;
            typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MapDenseMatrixPostcompute
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // YLocal += ULocal Z 
            const DistLowRank& DF = *_block.data.DF;
            const Dense& Z = *context.block.data.Z;
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
            const Dense& Z = *context.block.data.Z;
            Dense YLocalSub;
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixMatrix
            ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const Dense& Z = *context.block.data.Z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapDenseMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapDenseMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapDenseMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inSourceTeam )
        {
            const Node& node = *_block.data.N;
            typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMapDenseMatrixPostcompute
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^T Z 
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
            if( Conjugated )
            {
                // YLocal += conj(VLocal) Z
                hmatrix_tools::Conjugate( Z );
                hmatrix_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmatrix_tools::Conjugate( YLocal );
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
            Dense& Z = *context.block.data.Z;
            Dense YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( Z );
                hmatrix_tools::Conjugate( YLocalSub );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
                hmatrix_tools::Conjugate( YLocalSub );
            }
            else
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            const Dense& Z = *context.block.data.Z;
            Dense YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixTransposeMatrix
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapDenseMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapDenseMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    switch( _block.type )
    {
    case DIST_NODE:
    {
        const Node& node = *_block.data.N;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapDenseMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case SPLIT_NODE:
        if( _inSourceTeam )
        {
            const Node& node = *_block.data.N;
            typename MapDenseMatrixContext::SplitNodeContext& nodeContext = 
                *context.block.data.SN;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).HermitianTransposeMapDenseMatrixPostcompute
                    ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        }
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^H Z
            const DistLowRank& DF = *_block.data.DF;
            Dense& Z = *context.block.data.Z;
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
                hmatrix_tools::Conjugate( Z );
                hmatrix_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             Z.LockedBuffer(),         Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmatrix_tools::Conjugate( YLocal );
            }
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRank& SF = *_block.data.SF;
            Dense& Z = *context.block.data.Z;
            Dense YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
            }
            else
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( Z );
                hmatrix_tools::Conjugate( YLocalSub );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
                hmatrix_tools::Conjugate( YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDense& SD = *_block.data.SD;
            const Dense& Z = *context.block.data.Z;
            Dense YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
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

