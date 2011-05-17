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
    hmatrix_tools::Scale( beta, YLocal );

    MapDenseMatrixContext context;
    MapMatrixPrecompute( context, alpha, XLocal, YLocal );

    MapMatrixSummations( context, XLocal.Width() );
    //MapMatrixNaiveSummations( context, XLocal.Width() );

    //MapMatrixPassData( context, alpha, XLocal, YLocal );
    MapMatrixNaivePassData( context, alpha, XLocal, YLocal );

    MapMatrixBroadcasts( context, XLocal.Width() );
    //MapMatrixNaiveBroadcasts( context, XLocal.Width() );

    MapMatrixPostcompute( context, alpha, XLocal, YLocal );
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
    hmatrix_tools::Scale( beta, YLocal );

    MapDenseMatrixContext context;
    TransposeMapMatrixPrecompute( context, alpha, XLocal, YLocal );

    TransposeMapMatrixSummations( context, XLocal.Width() );
    //TransposeMapMatrixNaiveSummations( context, XLocal.Width() );

    //TransposeMapMatrixPassData( context, alpha, XLocal, YLocal );
    TransposeMapMatrixNaivePassData( context, alpha, XLocal, YLocal );

    TransposeMapMatrixBroadcasts( context, XLocal.Width() );
    //TransposeMapMatrixNaiveBroadcasts( context, XLocal.Width() );

    TransposeMapMatrixPostcompute( context, alpha, XLocal, YLocal );
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
    hmatrix_tools::Scale( beta, YLocal );

    MapDenseMatrixContext context;
    HermitianTransposeMapMatrixPrecompute( context, alpha, XLocal, YLocal );

    HermitianTransposeMapMatrixSummations( context, XLocal.Width() );
    //HermitianTransposeMapMatrixNaiveSummations( context, XLocal.Width() );

    //HermitianTransposeMapMatrixPassData( context, alpha, XLocal, YLocal );
    HermitianTransposeMapMatrixNaivePassData( context, alpha, XLocal, YLocal );

    HermitianTransposeMapMatrixBroadcasts( context, XLocal.Width() );
    //HermitianTransposeMapMatrixNaiveBroadcasts( context, XLocal.Width() );

    HermitianTransposeMapMatrixPostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case DIST_NODE:
        delete context._shell.data.DN; break;

    case SPLIT_QUASI2D:
        delete context._shell.data.SH; break;

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete context._shell.data.Z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }

    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        context._shell.type = DIST_NODE;
        context._shell.data.DN = 
            new typename MapDenseMatrixContext::DistNodeContext();
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        context._shell.type = DIST_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inSourceTeam )
        {
            // Form Z := alpha VLocal^[T/H] XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', DF.rank, width, DF.VLocal.Height(), 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, Z.Buffer(),               Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
    {
        context._shell.type = SPLIT_QUASI2D;
        context._shell.data.SH = 
            new typename SplitQuasi2d::MapDenseMatrixContext();

        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
            *context._shell.data.SH;

        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localSourceOffset, 0, SH._width, width );
        YLocalSub.View
        ( YLocal, _localTargetOffset, 0, SH._height, width );
        SH.MapMatrixPrecompute( splitContext, alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;

            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( alpha, SF.D, XLocalSub, Z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, SF.D, XLocalSub, Z );
            }
        }
        break;
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;

            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, this->_width, width );
            hmatrix_tools::MatrixMatrix( alpha, SD.D, XLocalSub, Z );
        }
        break;
    case QUASI2D:
    {
        context._shell.type = QUASI2D;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2d& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, H.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, H.Height(), width );
        H.MapMatrix( alpha, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case LOW_RANK:
    {
        context._shell.type = LOW_RANK;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRankMatrix<Scalar,Conjugated>& F = *shell.data.F;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, F.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, F.Height(), width );
        hmatrix_tools::MatrixMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case DENSE:
    {
        context._shell.type = DENSE;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, D.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, D.Height(), width );
        hmatrix_tools::MatrixMatrix
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case EMPTY:
        context._shell.type = EMPTY;
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case DIST_NODE:
        delete context._shell.data.DN; break;

    case SPLIT_QUASI2D:
        delete context._shell.data.SH; break;

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete context._shell.data.Z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }

    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        context._shell.type = DIST_NODE;
        context._shell.data.DN = 
            new typename MapDenseMatrixContext::DistNodeContext();
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        context._shell.type = DIST_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^T XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );
            blas::Gemm
            ( 'T', 'N', DF.rank, width, DF.ULocal.Height(),
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, Z.Buffer(),               Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
    {
        context._shell.type = SPLIT_QUASI2D;
        context._shell.data.SH = 
            new typename SplitQuasi2d::MapDenseMatrixContext();

        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
            *context._shell.data.SH;

        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, SH._height, width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, SH._width, width );
        SH.TransposeMapMatrixPrecompute
        ( splitContext, alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixTransposeMatrix( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        break;
    case QUASI2D:
    {
        context._shell.type = QUASI2D;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2d& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, H.Height(), width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, H.Width(), width );
        H.TransposeMapMatrix( alpha, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case LOW_RANK:
    {
        context._shell.type = LOW_RANK;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRankMatrix<Scalar,Conjugated>& F = *shell.data.F;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case DENSE:
    {
        context._shell.type = DENSE;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case EMPTY:
        context._shell.type = EMPTY;
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case DIST_NODE:
        delete context._shell.data.DN; break;

    case SPLIT_QUASI2D:
        delete context._shell.data.SH; break;

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete context._shell.data.Z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }

    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        context._shell.type = DIST_NODE;
        context._shell.data.DN = 
            new typename MapDenseMatrixContext::DistNodeContext();
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        context._shell.type = DIST_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^H XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );
            blas::Gemm
            ( 'C', 'N', DF.rank, width, DF.ULocal.Height(), 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, Z.Buffer(),               Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
    {
        context._shell.type = SPLIT_QUASI2D;
        context._shell.data.SH = 
            new typename SplitQuasi2d::MapDenseMatrixContext();

        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
            *context._shell.data.SH;

        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, SH._height, width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, SH._width, width );
        SH.HermitianTransposeMapMatrixPrecompute
        ( splitContext, alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;

            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SF.D, XLocalSub, Z );
        }
        break;
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        break;
    case QUASI2D:
    {
        context._shell.type = QUASI2D;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2d& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, H.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, H.Width(), width );
        H.HermitianTransposeMapMatrix
        ( alpha, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case LOW_RANK:
    {
        context._shell.type = LOW_RANK;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        //
        // NOTE: I'm not sure this case will ever happen. It would require a
        //       diagonal block to be low-rank.
        const LowRankMatrix<Scalar,Conjugated>& F = *shell.data.F;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case DENSE:
    {
        context._shell.type = DENSE;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case EMPTY:
        context._shell.type = EMPTY;
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MapMatrixSummationsCount( sizes, width );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapMatrixSummationsPack( buffer, offsets, context );

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
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, &buffer[offsets[i]], sizes[i],
                  0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( &buffer[offsets[i]], 0, sizes[i], 0, MPI_SUM, team );
            }
        }
    }

    // Unpack the reduced buffers (only roots of subcommunicators have data)
    MapMatrixSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMapMatrixSummationsCount( sizes, width );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    TransposeMapMatrixSummationsPack( buffer, offsets, context );

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
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, &buffer[offsets[i]], sizes[i],
                  0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( &buffer[offsets[i]], 0, sizes[i], 0, MPI_SUM, team );
            }
        }
    }

    // Unpack the reduced buffers (only roots of subcommunicators have data)
    TransposeMapMatrixSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixSummations
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixSummations");
#endif
    // This unconjugated version is identical
    TransposeMapMatrixSummations( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsCount( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank*width;
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummationsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsCount
                ( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank*width;
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int width = Z.Width();
            std::memcpy
            ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int width = Z.Width();
            std::memcpy
            ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsUnpack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int width = Z.Width();
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsUnpack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int width = Z.Width();
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( Z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaiveSummations
( MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaiveSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixNaiveSummations
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int width = Z.Width();
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, Z.Buffer(), 
                  DF.rank*width, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
            }
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaiveSummations
( MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaiveSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixNaiveSummations
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int width = Z.Width();
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, Z.Buffer(), 
                  DF.rank*width, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
            }
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaiveSummations
( MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaiveSummations( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPassData
( MapDenseMatrixContext& context, 
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPassData");
#endif
    // TODO: Implement AllToAll redistribution
    throw std::logic_error("Non-naive version not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPassData");
#endif
    // TODO: Implement AllToAll redistribution
    throw std::logic_error("Non-naive version not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrixPassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPassData");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixPassData( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaivePassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).MapMatrixNaivePassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                node.Child(0,1).MapMatrixNaivePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                node.Child(1,1).MapMatrixNaivePassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).MapMatrixNaivePassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                node.Child(2,3).MapMatrixNaivePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                node.Child(3,3).MapMatrixNaivePassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            node.Child(0,2).MapMatrixNaivePassData
            ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
            node.Child(0,3).MapMatrixNaivePassData
            ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
            node.Child(1,2).MapMatrixNaivePassData
            ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
            node.Child(1,3).MapMatrixNaivePassData
            ( nodeContext.Child(1,3), alpha, XLocal, YLocal );

            // Bottom-left quadrant
            node.Child(2,0).MapMatrixNaivePassData
            ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
            node.Child(2,1).MapMatrixNaivePassData
            ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
            node.Child(3,0).MapMatrixNaivePassData
            ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
            node.Child(3,1).MapMatrixNaivePassData
            ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapMatrixNaivePassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).MapMatrixNaivePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).MapMatrixNaivePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).MapMatrixNaivePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).MapMatrixNaivePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).MapMatrixNaivePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapMatrixNaivePassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).MapMatrixNaivePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).MapMatrixNaivePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).MapMatrixNaivePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).MapMatrixNaivePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).MapMatrixNaivePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapMatrixNaivePassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).MapMatrixNaivePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).MapMatrixNaivePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).MapMatrixNaivePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).MapMatrixNaivePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).MapMatrixNaivePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapMatrixNaivePassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).MapMatrixNaivePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).MapMatrixNaivePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).MapMatrixNaivePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).MapMatrixNaivePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).MapMatrixNaivePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            default:
                // This should be impossible
                break;
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;

        const DistLowRankMatrix& DF = *shell.data.DF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inSourceTeam )
            {
                mpi::Send
                ( Z.LockedBuffer(), DF.rank*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                Z.Resize( DF.rank, width, DF.rank );
                mpi::Recv
                ( Z.Buffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
            *context._shell.data.SH;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, SH._width, width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, SH._height, width );
        SH.MapMatrixNaivePassData( splitContext, alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( Z.LockedBuffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv( Z.Buffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( Z.LockedBuffer(), this->_height*width, 
              _rootOfOtherTeam, 0, comm );
        }
        else
        {
            Z.Resize( this->_height, width, this->_height );
            mpi::Recv
            ( Z.Buffer(), this->_height*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaivePassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                node.Child(1,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                node.Child(3,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            node.Child(0,2).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
            node.Child(0,3).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
            node.Child(1,2).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
            node.Child(1,3).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(1,3), alpha, XLocal, YLocal );

            // Bottom-left quadrant
            node.Child(2,0).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
            node.Child(2,1).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
            node.Child(3,0).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
            node.Child(3,1).TransposeMapMatrixNaivePassData
            ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,1), alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,0), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal ); 
                node.Child(3,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,2), alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,2), alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,0), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,2), alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,1), alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,3), alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(2,3), alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,2), alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(1,3), alpha, XLocal, YLocal );
                node.Child(3,1).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,1), alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(0,3), alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(3,0), alpha, XLocal, YLocal );
                break;
            default:
                // This should be impossible
                break;
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _inSourceTeam && _inTargetTeam )
            break;

        const DistLowRankMatrix& DF = *shell.data.DF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inTargetTeam )
            {
                mpi::Send
                ( Z.LockedBuffer(), DF.rank*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                Z.Resize( DF.rank, width, DF.rank );
                mpi::Recv
                ( Z.Buffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
            *context._shell.data.SH;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, SH._height, width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, SH._width, width );
        SH.TransposeMapMatrixNaivePassData
        ( splitContext, alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            mpi::Send
            ( Z.LockedBuffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv( Z.Buffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            if( XLocalSub.LDim() != XLocalSub.Height() )
            {
                // We must pack first
                Z.Resize( this->_height, width, this->_height );
                for( int j=0; j<width; ++j )
                {
                    std::memcpy
                    ( Z.Buffer(0,j), XLocalSub.LockedBuffer(0,j), 
                      this->_height*sizeof(Scalar) );
                }
                mpi::Send
                ( Z.LockedBuffer(), this->_height*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                mpi::Send
                ( XLocalSub.LockedBuffer(), this->_height*width, 
                  _rootOfOtherTeam, 0, comm );
            }
        }
        else
        {
            Z.Resize( this->_height, width, this->_height );
            mpi::Recv
            ( Z.Buffer(), this->_height*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaivePassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaivePassData");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaivePassData( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MapMatrixBroadcastsCount( sizes, width );

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
    MapMatrixBroadcastsPack( buffer, offsets, context );

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
    MapMatrixBroadcastsUnpack( buffer, offsets, context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMapMatrixBroadcastsCount( sizes, width );

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
    TransposeMapMatrixBroadcastsPack( buffer, offsets, context );

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
    TransposeMapMatrixBroadcastsUnpack( buffer, offsets, context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixBroadcasts( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsCount( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank*width;
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcastsCount
( std::vector<int>& sizes, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsCount
                ( sizes, width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank*width;
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsPack
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
                const DistLowRankMatrix& DF = *shell.data.DF;
                const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
                const int width = Z.Width();
                std::memcpy
                ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsPack
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
                const DistLowRankMatrix& DF = *shell.data.DF;
                const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
                const int width = Z.Width();
                std::memcpy
                ( &buffer[offsets[_level-1]], Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
            }
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsUnpack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );
            std::memcpy
            ( Z.Buffer(), &buffer[offsets[_level-1]], 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );
            std::memcpy
            ( Z.Buffer(), &buffer[offsets[_level-1]], 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaiveBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaiveBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixNaiveBroadcasts
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );

            MPI_Comm team = _subcomms->Subcomm( _level );
            mpi::Broadcast( Z.Buffer(), DF.rank*width, 0, team );
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaiveBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaiveBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixNaiveBroadcasts
                ( nodeContext.Child(t,s), width );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            Z.Resize( DF.rank, width, DF.rank );

            MPI_Comm team = _subcomms->Subcomm( _level );
            mpi::Broadcast( Z.Buffer(), DF.rank*width, 0, team );
        }
        break;
    case SPLIT_QUASI2D:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaiveBroadcasts
( MapDenseMatrixContext& context, int width ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaiveBroadcasts( context, width );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // YLocal += ULocal Z 
            const DistLowRankMatrix& DF = *shell.data.DF;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            blas::Gemm
            ( 'N', 'N', DF.ULocal.Height(), width, DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         Z.LockedBuffer(),         Z.LDim(),
              (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2d& SH = *shell.data.SH;
            typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
                *context._shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, this->_width, width );
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, this->_height, width );
            SH.MapMatrixPostcompute
            ( splitContext, alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixMatrix
            ( (Scalar)1, SF.D, Z, (Scalar)1, YLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int localHeight = this->_height;
            for( int j=0; j<width; ++j )
            {
                const Scalar* ZCol = Z.LockedBuffer(0,j);
                Scalar* YCol = YLocal.Buffer(_localTargetOffset,j);
                for( int i=0; i<localHeight; ++i )
                    YCol[i] += ZCol[i];
            }
        }
        break;
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^T Z 
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
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
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2d& SH = *shell.data.SH;
            typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
                *context._shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, this->_width, width );
            SH.TransposeMapMatrixPostcompute
            ( splitContext, alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            DenseMatrix<Scalar> YLocalSub;
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
            const SplitDenseMatrix& SD = *shell.data.SD;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixTransposeMatrix
            ( alpha, SD.D, Z, (Scalar)1, YLocalSub );
        }
        break;
    case QUASI2D:
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapDenseMatrixContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^H Z
            const DistLowRankMatrix& DF = *shell.data.DF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
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
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2d& SH = *shell.data.SH;
            typename SplitQuasi2d::MapDenseMatrixContext& splitContext = 
                *context._shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, this->_width, width );
            SH.HermitianTransposeMapMatrixPostcompute
            ( splitContext, alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            DenseMatrix<Scalar> YLocalSub;
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
            const SplitDenseMatrix& SD = *shell.data.SD;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SD.D, Z, (Scalar)1, YLocalSub );
        }
        break;
    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

