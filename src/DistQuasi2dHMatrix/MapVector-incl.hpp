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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVector");
#endif
    yLocal.Resize( LocalHeight() );
    MapVector( alpha, xLocal, (Scalar)0, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVector");
#endif
    yLocal.Resize( LocalWidth() );
    TransposeMapVector( alpha, xLocal, (Scalar)0, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVector");
#endif
    yLocal.Resize( LocalWidth() );
    HermitianTransposeMapVector( alpha, xLocal, (Scalar)0, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta,        Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVector");
#endif
    hmatrix_tools::Scale( beta, yLocal );

    MapVectorContext context;
    MapVectorPrecompute( context, alpha, xLocal, yLocal );

    MapVectorSummations( context );
    //MapVectorNaiveSummations( context );

    //MapVectorPassData( context, alpha, xLocal, yLocal );
    MapVectorNaivePassData( context, alpha, xLocal, yLocal );

    MapVectorBroadcasts( context );
    //MapVectorNaiveBroadcasts( context );

    MapVectorPostcompute( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVector");
#endif
    hmatrix_tools::Scale( beta, yLocal );

    MapVectorContext context;
    TransposeMapVectorPrecompute( context, alpha, xLocal, yLocal );

    TransposeMapVectorSummations( context );
    //TransposeMapVectorNaiveSummations( context );

    //TransposeMapVectorPassData( context, alpha, xLocal, yLocal );
    TransposeMapVectorNaivePassData( context, alpha, xLocal, yLocal );

    TransposeMapVectorBroadcasts( context );
    //TransposeMapVectorNaiveBroadcasts( context );

    TransposeMapVectorPostcompute( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta,        Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVector");
#endif
    hmatrix_tools::Scale( beta, yLocal );

    MapVectorContext context;
    HermitianTransposeMapVectorPrecompute( context, alpha, xLocal, yLocal );

    HermitianTransposeMapVectorSummations( context );
    //HermitianTransposeMapVectorNaiveSummations( context );

    //HermitianTransposeMapVectorPassData( context, alpha, xLocal, yLocal );
    HermitianTransposeMapVectorNaivePassData( context, alpha, xLocal, yLocal );

    HermitianTransposeMapVectorBroadcasts( context );
    //HermitianTransposeMapVectorNaiveBroadcasts( context );

    HermitianTransposeMapVectorPostcompute( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPrecompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPrecompute");
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
        delete context._shell.data.z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }

    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        context._shell.type = DIST_NODE;
        context._shell.data.DN = 
            new typename MapVectorContext::DistNodeContext();
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        context._shell.type = DIST_LOW_RANK;
        context._shell.data.z = new Vector<Scalar>();
        if( _inSourceTeam )
        {
            // Form z := alpha VLocal^[T/H] xLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            z.Resize( DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemv
            ( option, DF.VLocal.Height(), DF.rank, 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, z.Buffer(),               1 );
        }
        break;
    case SPLIT_QUASI2D:
    {
        context._shell.type = SPLIT_QUASI2D;
        context._shell.data.SH = new typename SplitQuasi2d::MapVectorContext();

        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapVectorContext& splitContext = 
            *context._shell.data.SH;

        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, SH._width );
        yLocalSub.View( yLocal, _localTargetOffset, SH._height );
        SH.MapVectorPrecompute( splitContext, alpha, xLocalSub, yLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.z = new Vector<Scalar>();
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar>& z = *context._shell.data.z;

            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeVector
                ( alpha, SF.D, xLocalSub, z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeVector
                ( alpha, SF.D, xLocalSub, z );
            }
        }
        break;
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.z = new Vector<Scalar>();
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            Vector<Scalar>& z = *context._shell.data.z;

            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, this->_width );
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocalSub, z );
        }
        break;
    case QUASI2D:
    {
        context._shell.type = QUASI2D;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2d& H = *shell.data.H;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, H.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, H.Height() );
        H.MapVector( alpha, xLocalSub, (Scalar)1, yLocalSub );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, F.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, F.Height() );
        hmatrix_tools::MatrixVector
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case DENSE:
    {
        context._shell.type = DENSE;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, D.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, D.Height() );
        hmatrix_tools::MatrixVector
        ( alpha, D, xLocalSub, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPrecompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorPrecompute");
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
        delete context._shell.data.z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }

    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        context._shell.type = DIST_NODE;
        context._shell.data.DN = 
            new typename MapVectorContext::DistNodeContext();
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        context._shell.type = DIST_LOW_RANK;
        context._shell.data.z = new Vector<Scalar>();
        if( _inTargetTeam )
        {
            // Form z := alpha ULocal^T xLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            z.Resize( DF.rank );
            blas::Gemv
            ( 'T', DF.ULocal.Height(), DF.rank, 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, z.Buffer(),               1 );
        }
        break;
    case SPLIT_QUASI2D:
    {
        context._shell.type = SPLIT_QUASI2D;
        context._shell.data.SH = new typename SplitQuasi2d::MapVectorContext();

        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapVectorContext& splitContext = 
            *context._shell.data.SH;

        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, SH._height );
        yLocalSub.View( yLocal, _localSourceOffset, SH._width );
        SH.TransposeMapVectorPrecompute
        ( splitContext, alpha, xLocalSub, yLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.z = new Vector<Scalar>();
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar>& z = *context._shell.data.z;
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SF.D.Height() );
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SF.D, xLocalSub, z );
        }
        break;
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.z = new Vector<Scalar>();
        break;
    case QUASI2D:
    {
        context._shell.type = QUASI2D;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2d& H = *shell.data.H;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, H.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, H.Width() );
        H.TransposeMapVector( alpha, xLocalSub, (Scalar)1, yLocalSub );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, F.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, F.Width() );
        hmatrix_tools::MatrixTransposeVector
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case DENSE:
    {
        context._shell.type = DENSE;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, D.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, D.Width() );
        hmatrix_tools::MatrixTransposeVector
        ( alpha, D, xLocalSub, (Scalar)1, yLocalSub );
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
HermitianTransposeMapVectorPrecompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorPrecompute");
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
        delete context._shell.data.z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }

    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        context._shell.type = DIST_NODE;
        context._shell.data.DN = 
            new typename MapVectorContext::DistNodeContext();
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        context._shell.type = DIST_LOW_RANK;
        context._shell.data.z = new Vector<Scalar>();
        if( _inTargetTeam )
        {
            // Form z := alpha ULocal^H xLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            z.Resize( DF.rank );
            blas::Gemv
            ( 'C', DF.ULocal.Height(), DF.rank, 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, z.Buffer(),               1 );
        }
        break;
    case SPLIT_QUASI2D:
    {
        context._shell.type = SPLIT_QUASI2D;
        context._shell.data.SH = new typename SplitQuasi2d::MapVectorContext();

        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapVectorContext& splitContext = 
            *context._shell.data.SH;

        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, SH._height );
        yLocalSub.View( yLocal, _localSourceOffset, SH._width );
        SH.HermitianTransposeMapVectorPrecompute
        ( splitContext, alpha, xLocalSub, yLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.z = new Vector<Scalar>();
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar>& z = *context._shell.data.z;

            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SF.D.Height() );
            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SF.D, xLocalSub, z );
        }
        break;
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.z = new Vector<Scalar>();
        break;
    case QUASI2D:
    {
        context._shell.type = QUASI2D;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2d& H = *shell.data.H;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, H.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, H.Width() );
        H.HermitianTransposeMapVector
        ( alpha, xLocalSub, (Scalar)1, yLocalSub );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, F.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, F.Width() );
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case DENSE:
    {
        context._shell.type = DENSE;
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, D.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, D.Width() );
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, D, xLocalSub, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSummations
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MapVectorSummationsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapVectorSummationsPack( buffer, offsets, context );

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
    MapVectorSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorSummations
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMapVectorSummationsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    TransposeMapVectorSummationsPack( buffer, offsets, context );

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
    TransposeMapVectorSummationsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorSummations
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorSummations");
#endif
    // This unconjugated version is identical
    TransposeMapVectorSummations( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSummationsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSummationsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorSummationsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorSummationsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            const Vector<Scalar>& z = *context._shell.data.z;
            std::memcpy
            ( &buffer[offsets[_level-1]], z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorSummationsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            const Vector<Scalar>& z = *context._shell.data.z;
            std::memcpy
            ( &buffer[offsets[_level-1]], z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummationsUnpack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummationsUnpack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorSummationsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaiveSummations
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaiveSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaiveSummations
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, z.Buffer(), 
                  DF.rank, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( z.LockedBuffer(), 0, DF.rank, 0, MPI_SUM, team );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorNaiveSummations
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorNaiveSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorNaiveSummations
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, z.Buffer(), 
                  DF.rank, 0, MPI_SUM, team );
            }
            else
                mpi::Reduce( z.LockedBuffer(), 0, DF.rank, 0, MPI_SUM, team );
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
HermitianTransposeMapVectorNaiveSummations
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapVectorNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorNaiveSummations( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPassData");
#endif
    // TODO: Implement AllToAll redistribution
    throw std::logic_error("Non-naive version not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorPassData");
#endif
    // TODO: Implement AllToAll redistribution
    throw std::logic_error("Non-naive version not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVectorPassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorPassData");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorPassData( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaivePassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).MapVectorNaivePassData
                ( nodeContext.Child(0,0), alpha, xLocal, yLocal );
                node.Child(0,1).MapVectorNaivePassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).MapVectorNaivePassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                node.Child(1,1).MapVectorNaivePassData
                ( nodeContext.Child(1,1), alpha, xLocal, yLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).MapVectorNaivePassData
                ( nodeContext.Child(2,2), alpha, xLocal, yLocal );
                node.Child(2,3).MapVectorNaivePassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).MapVectorNaivePassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                node.Child(3,3).MapVectorNaivePassData
                ( nodeContext.Child(3,3), alpha, xLocal, yLocal );
            }
            // Top-right quadrant
            node.Child(0,2).MapVectorNaivePassData
            ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
            node.Child(0,3).MapVectorNaivePassData
            ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
            node.Child(1,2).MapVectorNaivePassData
            ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
            node.Child(1,3).MapVectorNaivePassData
            ( nodeContext.Child(1,3), alpha, xLocal, yLocal );

            // Bottom-left quadrant
            node.Child(2,0).MapVectorNaivePassData
            ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
            node.Child(2,1).MapVectorNaivePassData
            ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
            node.Child(3,0).MapVectorNaivePassData
            ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
            node.Child(3,1).MapVectorNaivePassData
            ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapVectorNaivePassData
                ( nodeContext.Child(0,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(0,1).MapVectorNaivePassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).MapVectorNaivePassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(0,2).MapVectorNaivePassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).MapVectorNaivePassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(0,3).MapVectorNaivePassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).MapVectorNaivePassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapVectorNaivePassData
                ( nodeContext.Child(1,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,1).MapVectorNaivePassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).MapVectorNaivePassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(1,3).MapVectorNaivePassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal );
                node.Child(3,1).MapVectorNaivePassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(1,2).MapVectorNaivePassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).MapVectorNaivePassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapVectorNaivePassData
                ( nodeContext.Child(2,2), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(2,3).MapVectorNaivePassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).MapVectorNaivePassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,2).MapVectorNaivePassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).MapVectorNaivePassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,2).MapVectorNaivePassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).MapVectorNaivePassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapVectorNaivePassData
                ( nodeContext.Child(3,3), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(2,3).MapVectorNaivePassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).MapVectorNaivePassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,3).MapVectorNaivePassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal );
                node.Child(3,1).MapVectorNaivePassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,3).MapVectorNaivePassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).MapVectorNaivePassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
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
        Vector<Scalar>& z = *context._shell.data.z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inSourceTeam )
            {
                mpi::Send
                ( z.LockedBuffer(), DF.rank, _rootOfOtherTeam, 0, comm );
            }
            else
            {
                z.Resize( DF.rank );
                mpi::Recv( z.Buffer(), DF.rank, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapVectorContext& splitContext = 
            *context._shell.data.SH;

        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, SH._width );
        yLocalSub.View( yLocal, _localTargetOffset, SH._height );
        SH.MapVectorNaivePassData( splitContext, alpha, xLocalSub, yLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        Vector<Scalar>& z = *context._shell.data.z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( z.LockedBuffer(), SF.rank, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            z.Resize( SF.rank );
            mpi::Recv( z.Buffer(), SF.rank, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        Vector<Scalar>& z = *context._shell.data.z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( z.LockedBuffer(), this->_height, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            z.Resize( this->_height );
            mpi::Recv( z.Buffer(), this->_height, _rootOfOtherTeam, 0, comm );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorNaivePassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,0), alpha, xLocal, yLocal );
                node.Child(0,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                node.Child(1,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,1), alpha, xLocal, yLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,2), alpha, xLocal, yLocal );
                node.Child(2,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                node.Child(3,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,3), alpha, xLocal, yLocal );
            }
            // Top-right quadrant
            node.Child(0,2).TransposeMapVectorNaivePassData
            ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
            node.Child(0,3).TransposeMapVectorNaivePassData
            ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
            node.Child(1,2).TransposeMapVectorNaivePassData
            ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
            node.Child(1,3).TransposeMapVectorNaivePassData
            ( nodeContext.Child(1,3), alpha, xLocal, yLocal );

            // Bottom-left quadrant
            node.Child(2,0).TransposeMapVectorNaivePassData
            ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
            node.Child(2,1).TransposeMapVectorNaivePassData
            ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
            node.Child(3,0).TransposeMapVectorNaivePassData
            ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
            node.Child(3,1).TransposeMapVectorNaivePassData
            ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,1), alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,0), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal ); 
                node.Child(3,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,2), alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,2), alpha, xLocal, yLocal );
                node.Child(2,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,0), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,2), alpha, xLocal, yLocal );
                node.Child(2,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,1), alpha, xLocal, yLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,3), alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(2,3), alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,2), alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(1,3), alpha, xLocal, yLocal );
                node.Child(3,1).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,1), alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMapVectorNaivePassData
                ( nodeContext.Child(0,3), alpha, xLocal, yLocal );
                node.Child(3,0).TransposeMapVectorNaivePassData
                ( nodeContext.Child(3,0), alpha, xLocal, yLocal );
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
        Vector<Scalar>& z = *context._shell.data.z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );
        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _inTargetTeam )
            {
                mpi::Send
                ( z.LockedBuffer(), DF.rank, _rootOfOtherTeam, 0, comm );
            }
            else
            {
                z.Resize( DF.rank );
                mpi::Recv( z.Buffer(), DF.rank, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2d& SH = *shell.data.SH;
        typename SplitQuasi2d::MapVectorContext& splitContext = 
            *context._shell.data.SH;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, SH._height );
        yLocalSub.View( yLocal, _localSourceOffset, SH._width );
        SH.TransposeMapVectorNaivePassData
        ( splitContext, alpha, xLocalSub, yLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        Vector<Scalar>& z = *context._shell.data.z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
            mpi::Send( z.LockedBuffer(), SF.rank, _rootOfOtherTeam, 0, comm );
        else
        {
            z.Resize( SF.rank );
            mpi::Recv( z.Buffer(), SF.rank, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        Vector<Scalar>& z = *context._shell.data.z;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, this->_height );
            mpi::Send
            ( xLocalSub.LockedBuffer(), this->_height, 
              _rootOfOtherTeam, 0, comm );
        }
        else
        {
            z.Resize( this->_height );
            mpi::Recv( z.Buffer(), this->_height, _rootOfOtherTeam, 0, comm );
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
HermitianTransposeMapVectorNaivePassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapVectorNaivePassData");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorNaivePassData( context, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorBroadcasts
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MapVectorBroadcastsCount( sizes );

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
    MapVectorBroadcastsPack( buffer, offsets, context );

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
    MapVectorBroadcastsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorBroadcasts
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMapVectorBroadcastsCount( sizes );

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
    TransposeMapVectorBroadcastsPack( buffer, offsets, context );

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
    TransposeMapVectorBroadcastsUnpack( buffer, offsets, context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorBroadcasts
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorBroadcasts( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorBroadcastsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            const Vector<Scalar>& z = *context._shell.data.z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorBroadcastsPack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            const Vector<Scalar>& z = *context._shell.data.z;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            z.Resize( DF.rank );
            std::memcpy
            ( z.Buffer(), &buffer[offsets[_level-1]], DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorBroadcastsUnpack
                ( buffer, offsets, nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            z.Resize( DF.rank );
            std::memcpy
            ( z.Buffer(), &buffer[offsets[_level-1]], DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaiveBroadcasts
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaiveBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaiveBroadcasts
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            z.Resize( DF.rank );
            mpi::Broadcast( z.Buffer(), DF.rank, 0, team );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorNaiveBroadcasts
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorNaiveBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorNaiveBroadcasts
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            MPI_Comm team = _subcomms->Subcomm( _level );
            z.Resize( DF.rank );
            mpi::Broadcast( z.Buffer(), DF.rank, 0, team );
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
HermitianTransposeMapVectorNaiveBroadcasts
( MapVectorContext& context ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapVectorNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorNaiveBroadcasts( context );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPostcompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // yLocal += ULocal z
            const DistLowRankMatrix& DF = *shell.data.DF;
            const Vector<Scalar>& z = *context._shell.data.z;
            blas::Gemv
            ( 'N', DF.ULocal.Height(), DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         z.LockedBuffer(),         1,
              (Scalar)1, yLocal.Buffer(),          1 );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2d& SH = *shell.data.SH;
            typename SplitQuasi2d::MapVectorContext& splitContext = 
                *context._shell.data.SH;

            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, this->_width );
            yLocalSub.View( yLocal, _localTargetOffset, this->_height );
            SH.MapVectorPostcompute
            ( splitContext, alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            const Vector<Scalar>& z = *context._shell.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localTargetOffset, SF.D.Height() );
            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const Vector<Scalar>& z = *context._shell.data.z;
            const int localHeight = this->_height;
            const Scalar* zBuffer = z.LockedBuffer();
            Scalar* yLocalBuffer = yLocal.Buffer(_localTargetOffset);
            for( int i=0; i<localHeight; ++i )
                yLocalBuffer[i] += zBuffer[i];
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPostcompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // yLocal += (VLocal^[T/H])^T z
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
            if( Conjugated )
            {
                // yLocal += conj(VLocal) z
                hmatrix_tools::Conjugate( z );
                hmatrix_tools::Conjugate( yLocal );
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             z.LockedBuffer(),         1,
                  (Scalar)1, yLocal.Buffer(),          1 );
                hmatrix_tools::Conjugate( yLocal );
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
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2d& SH = *shell.data.SH;
            typename SplitQuasi2d::MapVectorContext& splitContext = 
                *context._shell.data.SH;

            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, this->_height );
            yLocalSub.View( yLocal, _localSourceOffset, this->_width );
            SH.TransposeMapVectorPostcompute
            ( splitContext, alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar>& z = *context._shell.data.z;

            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( z );
                hmatrix_tools::Conjugate( yLocalSub );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
                hmatrix_tools::Conjugate( yLocalSub );
            }
            else
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const Vector<Scalar>& z = *context._shell.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SD.D.Width() );
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SD.D, z, (Scalar)1, yLocalSub );
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
HermitianTransposeMapVectorPostcompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    {
        const DistNode& node = *shell.data.DN;
        typename MapVectorContext::DistNodeContext& nodeContext = 
            *context._shell.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        break;
    }
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // yLocal += (VLocal^[T/H])^H z
            const DistLowRankMatrix& DF = *shell.data.DF;
            Vector<Scalar>& z = *context._shell.data.z;
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
                hmatrix_tools::Conjugate( z );
                hmatrix_tools::Conjugate( yLocal );
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             z.LockedBuffer(),         1,
                  (Scalar)1, yLocal.Buffer(),          1 );
                hmatrix_tools::Conjugate( yLocal );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2d& SH = *shell.data.SH;
            typename SplitQuasi2d::MapVectorContext& splitContext = 
                *context._shell.data.SH;

            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, this->_height );
            yLocalSub.View( yLocal, _localSourceOffset, this->_width );
            SH.HermitianTransposeMapVectorPostcompute
            ( splitContext, alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar>& z = *context._shell.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
            }
            else
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( z );
                hmatrix_tools::Conjugate( yLocalSub );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocalSub );
                hmatrix_tools::Conjugate( yLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const Vector<Scalar>& z = *context._shell.data.z;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SD.D.Width() );
            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SD.D, z, (Scalar)1, yLocalSub );
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
