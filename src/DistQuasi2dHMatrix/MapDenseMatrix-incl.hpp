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

    MapMatrixSummations( context );
    //MapMatrixNaiveSummations( context );

    //MapMatrixPassData( context, alpha, XLocal, YLocal );
    MapMatrixNaivePassData( context, alpha, XLocal, YLocal );

    MapMatrixBroadcasts( context );
    //MapMatrixNaiveBroadcasts( context );

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

    TransposeMapMatrixSummations( context );
    //TransposeMapMatrixNaiveSummations( context );

    //TransposeMapMatrixPassData( context, alpha, XLocal, YLocal );
    TransposeMapMatrixNaivePassData( context, alpha, XLocal, YLocal );

    TransposeMapMatrixBroadcasts( context );
    //TransposeMapMatrixNaiveBroadcasts( context );

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

    HermitianTransposeMapMatrixSummations( context );
    //HermitianTransposeMapMatrixNaiveSummations( context );

    //HermitianTransposeMapMatrixPassData( context, alpha, XLocal, YLocal );
    HermitianTransposeMapMatrixNaivePassData( context, alpha, XLocal, YLocal );

    HermitianTransposeMapMatrixBroadcasts( context );
    //HermitianTransposeMapMatrixNaiveBroadcasts( context );

    HermitianTransposeMapMatrixPostcompute( context, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

// HERE
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPrecompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixPrecompute( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // Form Z := alpha VLocal^[T/H] XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', DF.rank, width, DF.VLocal.Height(), 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, DF.Z.Buffer(),            DF.Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SH._width, width );
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SH._height, width );
            SH.MapMatrixPrecompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( alpha, SF.D, XLocalSub, SF.Z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, SF.D, XLocalSub, SF.Z );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, this->_width, width );
            hmatrix_tools::MatrixMatrix( alpha, SD.D, XLocalSub, SD.Z );
        }
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, H.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, H.Height(), width );
        H.MapMatrix( alpha, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case LOW_RANK:
    {
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
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPrecompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixPrecompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^T XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            blas::Gemm
            ( 'T', 'N', DF.rank, width, DF.ULocal.Height(),
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, DF.Z.Buffer(),            DF.Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SH._height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SH._width, width );
            SH.TransposeMapMatrixPrecompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixTransposeMatrix
            ( alpha, SF.D, XLocalSub, SF.Z );
        }
        break;
    case SPLIT_DENSE:
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapMatrixPrecompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^H XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            blas::Gemm
            ( 'C', 'N', DF.rank, width, DF.ULocal.Height(), 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, DF.Z.Buffer(),            DF.Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SH._height, width );
            YLocalSub.View( YLocal, _localSourceOffset, 0, SH._width, width );
            SH.HermitianTransposeMapMatrixPrecompute
            ( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SF.D, XLocalSub, SF.Z );
        }
        break;
    case SPLIT_DENSE:
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
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
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
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
    MapMatrixSummationsCount( sizes, alpha, XLocal, YLocal );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapMatrixSummationsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    MapMatrixSummationsUnpack( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
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
    TransposeMapMatrixSummationsCount( sizes, alpha, XLocal, YLocal );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    TransposeMapMatrixSummationsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    TransposeMapMatrixSummationsUnpack
    ( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixSummations");
#endif
    // This unconjugated version is identical
    TransposeMapMatrixSummations( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsCount
( std::vector<int>& sizes, 
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
( std::vector<int>& sizes,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaiveSummations");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixNaiveSummations
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, DF.Z.Buffer(), 
                  DF.rank*width, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( DF.Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaiveSummations");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixNaiveSummations
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, DF.Z.Buffer(), 
                  DF.rank*width, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( DF.Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaiveSummations( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPassData");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixPassData( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaivePassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(0,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            node.Child(0,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(0,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(1,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(1,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );

            // Bottom-left quadrant
            node.Child(2,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(2,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(3,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(3,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            default:
#ifndef RELEASE
                throw std::logic_error("Invalid subteam");
#endif
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
        MPI_Comm comm = _subcomms->Subcomm( 0 );
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inSourceTeam )
            {
                mpi::Send
                ( DF.Z.LockedBuffer(), DF.rank*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                DF.Z.Resize( DF.rank, width, DF.rank );
                mpi::Recv
                ( DF.Z.Buffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, SH._width, width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, SH._height, width );
        SH.MapMatrixNaivePassData( alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( SF.Z.LockedBuffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SF.Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv
            ( SF.Z.Buffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( SD.Z.LockedBuffer(), this->_height*width, 
              _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SD.Z.Resize( this->_height, width, this->_height );
            mpi::Recv
            ( SD.Z.Buffer(), this->_height*width, _rootOfOtherTeam, 0, comm );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            node.Child(0,2).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(0,3).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(1,2).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(1,3).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );

            // Bottom-left quadrant
            node.Child(2,0).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(2,1).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(3,0).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(3,1).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal ); 
                node.Child(3,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
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
        MPI_Comm comm = _subcomms->Subcomm( 0 );
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inTargetTeam )
            {
                mpi::Send
                ( DF.Z.LockedBuffer(), DF.rank*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                DF.Z.Resize( DF.rank, width, DF.rank );
                mpi::Recv
                ( DF.Z.Buffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, SH._height, width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, SH._width, width );
        SH.TransposeMapMatrixNaivePassData( alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            mpi::Send
            ( SF.Z.LockedBuffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SF.Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv
            ( SF.Z.Buffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            if( XLocalSub.LDim() != XLocalSub.Height() )
            {
                // We must pack first
                SD.Z.Resize( this->_height, width, this->_height );
                for( int j=0; j<width; ++j )
                {
                    std::memcpy
                    ( SD.Z.Buffer(0,j), XLocalSub.LockedBuffer(0,j), 
                      this->_height*sizeof(Scalar) );
                }
                mpi::Send
                ( SD.Z.LockedBuffer(), this->_height*width, 
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
            SD.Z.Resize( this->_height, width, this->_height );
            mpi::Recv
            ( SD.Z.Buffer(), this->_height*width, _rootOfOtherTeam, 0, comm );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaivePassData");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaivePassData( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
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
    MapMatrixBroadcastsCount( sizes, alpha, XLocal, YLocal );

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
    MapMatrixBroadcastsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    MapMatrixBroadcastsUnpack( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
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
    TransposeMapMatrixBroadcastsCount( sizes, alpha, XLocal, YLocal );

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
    TransposeMapMatrixBroadcastsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    TransposeMapMatrixBroadcastsUnpack
    ( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixBroadcasts( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsCount
( std::vector<int>& sizes,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
( std::vector<int>& sizes,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            std::memcpy
            ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
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
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            std::memcpy
            ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaiveBroadcasts");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixNaiveBroadcasts
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.Z.Resize( DF.rank, width, DF.rank );
            mpi::Broadcast( DF.Z.Buffer(), DF.rank*width, 0, team );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaiveBroadcasts");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixNaiveBroadcasts
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.Z.Resize( DF.rank, width, DF.rank );
            mpi::Broadcast( DF.Z.Buffer(), DF.rank*width, 0, team );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaiveBroadcasts( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPostcompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixPostcompute( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // YLocal += ULocal Z 
            const DistLowRankMatrix& DF = *shell.data.DF;
            blas::Gemm
            ( 'N', 'N', DF.ULocal.Height(), width, DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         DF.Z.LockedBuffer(),      DF.Z.LDim(),
              (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, this->_width, width );
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, this->_height, width );
            SH.MapMatrixPostcompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixMatrix
            ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const int localHeight = this->_height;
            for( int j=0; j<width; ++j )
            {
                const Scalar* ZCol = SD.Z.LockedBuffer(0,j);
                Scalar* YCol = YLocal.Buffer(0,j);
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixPostcompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^T Z 
            const DistLowRankMatrix& DF = *shell.data.DF;
            if( Conjugated )
            {
                // YLocal += conj(VLocal) Z
                hmatrix_tools::Conjugate( DF.Z );
                hmatrix_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmatrix_tools::Conjugate( YLocal );
            }
            else
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, this->_width, width );
            SH.TransposeMapMatrixPostcompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( SF.Z );
                hmatrix_tools::Conjugate( YLocalSub );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
                hmatrix_tools::Conjugate( YLocalSub );
            }
            else
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixTransposeMatrix
            ( alpha, SD.D, SD.Z, (Scalar)1, YLocalSub );
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
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapMatrixPostcompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^H Z
            const DistLowRankMatrix& DF = *shell.data.DF;
            if( Conjugated )
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
            }
            else
            {
                // YLocal += conj(VLocal) Z
                hmatrix_tools::Conjugate( DF.Z );
                hmatrix_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmatrix_tools::Conjugate( YLocal );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, this->_width, width );
            SH.HermitianTransposeMapMatrixPostcompute
            ( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
            }
            else
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( SF.Z );
                hmatrix_tools::Conjugate( YLocalSub );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
                hmatrix_tools::Conjugate( YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SD.D, SD.Z, (Scalar)1, YLocalSub );
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

