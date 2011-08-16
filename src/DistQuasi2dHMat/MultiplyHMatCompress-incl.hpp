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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompress()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompress");
#endif
    MultiplyHMatCompressRandomSetup( 0 );
    MultiplyHMatCompressRandomPrecompute( 0 );

    throw std::logic_error("This routine is in a state of flux.");

    // HERE
    /*
    MultiplyHMatCompressRandomSummations();
    MultiplyHMatCompressRandomExchangeCount();
    MultiplyHMatCompressRandomExchangePack();
    MultiplyHMatCompressRandomExchange();
    MultiplyHMatCompressRandomExchangeUnpack();
    MultiplyHMatCompressRandomBroadcasts();
    MultiplyHMatCompressRandomPostcompute();

    MultiplyHMatCompressLocalQR();
    MultiplyHMatCompressParallelQR();

    MultiplyHMatCompressLastExchangeCount();
    MultiplyHMatCompressLastExchangePack();
    MultiplyHMatCompressLastExchange();
    MultiplyHMatCompressFinalize();
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressRandomSetup
( int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressRandomSetup");
#endif
    if( Height() == 0 || Width() == 0 || !_inSourceTeam )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        const int numEntries = _VMap.Size();
        _VMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            rank += _VMap.CurrentEntry()->Width();

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressRandomSetup( rank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;

        // Add the F+=HH updates
        int numEntries = _rowXMap.Size();
        _rowXMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            rank += _rowXMap.CurrentEntry()->Width();

        // Add the low-rank updates
        numEntries = _VMap.Size();
        _VMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            rank += _VMap.CurrentEntry()->Width();

        // Add the rank of the original low-rank matrix
        rank += DF.VLocal.Width();

        // Create space for Omega
        const int localWidth = DF.VLocal.Height();
        const int sampleRank = SampleRank( rank );
        _colOmega.Resize( localWidth, sampleRank );
        ParallelGaussianRandomVectors( _colOmega );

        // Create space for the temporary product, V' Omega
        _ZMap.Set( 0, new Dense<Scalar>(rank,sampleRank) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;

        if( !_haveDenseUpdate )
        {
            // Add the F+=HH updates
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
                rank += _rowXMap.CurrentEntry()->Width();

            // Add the regular updates
            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
                rank += _VMap.CurrentEntry()->Width();

            // Add the original rank
            rank += SF.D.Width();

            // Create space for Omega
            const int width = SF.D.Height();
            const int sampleRank = SampleRank( rank );
            _colOmega.Resize( width, sampleRank );
            ParallelGaussianRandomVectors( _colOmega );

            // Create space for the temporary product, V' Omega
            _ZMap.Set( 0, new Dense<Scalar>(rank,sampleRank) );
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        
        if( !_haveDenseUpdate )
        {
            // Add the F+=HH updates
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
                rank += _rowXMap.CurrentEntry()->Width();

            // Add the regular updates
            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
                rank += _VMap.CurrentEntry()->Width();

            // Add the original low-rank matrix
            rank += F.Rank();

            // Create space for Omega
            const int width = F.V.Height();
            const int sampleRank = SampleRank( rank );
            _colOmega.Resize( width, sampleRank );
            ParallelGaussianRandomVectors( _colOmega );

            // Create space for the temporary product, V' Omega
            _ZMap.Set( 0, new Dense<Scalar>(rank,sampleRank) );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressRandomPrecompute
( int rank )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressRandomPrecompute");
#endif
    if( Height() == 0 || Width() == 0 || !_inSourceTeam )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        const int numEntries = _VMap.Size();
        _VMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_VMap.Increment() )
        {
            const Dense<Scalar>& V = *_VMap.CurrentEntry();
            MultiplyHMatCompressRandomPrecomputeImport( rank, V );

            rank += V.Width();
            _VMap.EraseCurrentEntry();
        }

        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressRandomPrecompute( rank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;
        Dense<Scalar>& Z = _ZMap.Get( 0 );
        const Dense<Scalar>& Omega = _colOmega;
        const char option = ( Conjugated ? 'C' : 'T' );

        // Add the F+=HH updates
        int numEntries = _rowXMap.Size();
        _rowXMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
        {
            const Dense<Scalar>& X = *_rowXMap.CurrentEntry();
            blas::Gemm
            ( option, 'N', X.Width(), Omega.Width(), X.Height(),
              (Scalar)1, X.LockedBuffer(),     X.LDim(),
                         Omega.LockedBuffer(), Omega.LDim(),
              (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );

            rank += X.Width();
            _rowXMap.EraseCurrentEntry();
        }

        // Add the low-rank updates
        numEntries = _VMap.Size();
        _VMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_VMap.Increment() )
        {
            const Dense<Scalar>& V = *_VMap.CurrentEntry();
            blas::Gemm
            ( option, 'N', V.Width(), Omega.Width(), V.Height(),
              (Scalar)1, V.LockedBuffer(),     V.LDim(),
                         Omega.LockedBuffer(), Omega.LDim(),
              (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );

            rank += V.Width();
            _VMap.EraseCurrentEntry();
        }

        // Add the rank of the original low-rank matrix
        blas::Gemm
        ( option, 'N', DF.VLocal.Width(), Omega.Width(), DF.VLocal.Height(),
          (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                     Omega.LockedBuffer(),     Omega.LDim(),
          (Scalar)0, Z.Buffer(rank,0),         Z.LDim() );
        rank += DF.VLocal.Width();
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;

        if( !_haveDenseUpdate )
        {
            Dense<Scalar>& Z = _ZMap.Get( 0 );
            const Dense<Scalar>& Omega = _colOmega;
            const char option = ( Conjugated ? 'C' : 'T' );

            // Add the F+=HH updates
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                const Dense<Scalar>& X = *_rowXMap.CurrentEntry();
                blas::Gemm
                ( option, 'N', X.Width(), Omega.Width(), X.Height(),
                  (Scalar)1, X.LockedBuffer(),     X.LDim(),
                             Omega.LockedBuffer(), Omega.LDim(),
                  (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );

                rank += X.Width();
                _rowXMap.EraseCurrentEntry();
            }

            // Add the regular updates
            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                const Dense<Scalar>& V = *_VMap.CurrentEntry();
                blas::Gemm
                ( option, 'N', V.Width(), Omega.Width(), V.Height(),
                  (Scalar)1, V.LockedBuffer(),     V.LDim(),
                             Omega.LockedBuffer(), Omega.LDim(),
                  (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );

                rank += V.Width();
                _VMap.EraseCurrentEntry();
            }

            // Add the original contribution
            blas::Gemm
            ( option, 'N', SF.D.Width(), Omega.Width(), SF.D.Height(),
              (Scalar)1, SF.D.LockedBuffer(),  SF.D.LDim(),
                         Omega.LockedBuffer(), Omega.LDim(),
              (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );
            rank += SF.D.Width();
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        
        if( !_haveDenseUpdate )
        {
            Dense<Scalar>& Z = _ZMap.Get( 0 );
            const Dense<Scalar>& Omega = _colOmega;
            const char option = ( Conjugated ? 'C' : 'T' );

            // Add the F+=HH updates
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                const Dense<Scalar>& X = *_rowXMap.CurrentEntry();
                blas::Gemm
                ( option, 'N', X.Width(), Omega.Width(), X.Height(),
                  (Scalar)1, X.LockedBuffer(),     X.LDim(),
                             Omega.LockedBuffer(), Omega.LDim(),
                  (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );

                rank += X.Width();
                _rowXMap.EraseCurrentEntry();
            }

            // Add the regular updates
            numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                const Dense<Scalar>& V = *_VMap.CurrentEntry();
                blas::Gemm
                ( option, 'N', V.Width(), Omega.Width(), V.Height(),
                  (Scalar)1, V.LockedBuffer(),     V.LDim(),
                             Omega.LockedBuffer(), Omega.LDim(),
                  (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );

                rank += V.Width();
                _VMap.EraseCurrentEntry();
            }

            // Add the original low-rank matrix
            blas::Gemm
            ( option, 'N', F.V.Width(), Omega.Width(), F.V.Height(),
              (Scalar)1, F.V.LockedBuffer(),   F.V.LDim(),
                         Omega.LockedBuffer(), Omega.LDim(),
              (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );
            rank += F.V.Width();
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
MultiplyHMatCompressRandomPrecomputeImport( int rank, const Dense<Scalar>& V )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMat::MultiplyHMatCompressRandomPrecomputeImport");
#endif
    if( Height() == 0 || Width() == 0 || !_inSourceTeam )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        MPI_Comm team = _teams->Team( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamSize == 2 )
        {
            const int sStart = (teamRank==0 ? 0 : 2);            
            const int sStop = (teamRank==0 ? 2 : 4);
            Dense<Scalar> VSub;
            for( int s=sStart,sOffset=0; s<sStop; 
                 sOffset+=node.sourceSizes[s],++s )
            {
                VSub.LockedView
                ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                for( int t=0; t<4; ++t )
                    node.Child(t,s).MultiplyHMatCompressRandomPrecomputeImport
                    ( rank, VSub );
            }
        }
        else  // teamSize >= 4
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressRandomPrecomputeImport
                    ( rank, V );
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        Dense<Scalar> VSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            VSub.LockedView( V, sOffset, 0, node.sourceSizes[s], V.Width() );
            for( int t=0; t<4; ++t )
                node.Child(t,s).MultiplyHMatCompressRandomPrecomputeImport
                ( rank, VSub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        Dense<Scalar>& Z = _ZMap.Get( 0 );
        const Dense<Scalar>& Omega = _colOmega;
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( option, 'N', V.Width(), Omega.Width(), V.Height(),
          (Scalar)1, V.LockedBuffer(),     V.LDim(),
                     Omega.LockedBuffer(), Omega.LDim(),
          (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );
        break;
    }
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( !_haveDenseUpdate )
        {
            Dense<Scalar>& Z = _ZMap.Get( 0 );
            const Dense<Scalar>& Omega = _colOmega;
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', V.Width(), Omega.Width(), V.Height(),
              (Scalar)1, V.LockedBuffer(),     V.LDim(),
                         Omega.LockedBuffer(), Omega.LDim(),
              (Scalar)0, Z.Buffer(rank,0),     Z.LDim() );
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

