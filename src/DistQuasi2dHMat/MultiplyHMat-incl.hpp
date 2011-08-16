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

#include "./MultiplyHMatFormGhostRanks-incl.hpp"
#include "./MultiplyHMatParallelQR-incl.hpp"

#include "./MultiplyHMatMain-incl.hpp"
#include "./MultiplyHMatFHH-incl.hpp"
#include "./MultiplyHMatCompress-incl.hpp"

// C := alpha A B
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C,
  int multType )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
    if( multType < 0 || multType > 2 )
        throw std::logic_error("Invalid multiplication type");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    if( multType == 0 )
        A.MultiplyHMatSingleUpdateAccumulate( alpha, B, C );
    else if( multType == 1 )
        A.MultiplyHMatSingleLevelAccumulate( alpha, B, C );
    else
        A.MultiplyHMatFullAccumulate( alpha, B, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFullAccumulate
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFullAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

#ifdef TIME_MULTIPLY
    Timer timer; 
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Start( 0 );
#endif
    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif
    A.MultiplyHMatFormGhostRanks( B );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 1 );
#endif

    const int startLevel = 0;
    const int endLevel = A.NumLevels();

    const int startUpdate = 0;
    const int endUpdate = 4;

#ifdef TIME_MULTIPLY
    timer.Start( 2 );
#endif
    A.MultiplyHMatMainPrecompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 2 );
    timer.Start( 3 );
#endif
    A.MultiplyHMatMainSums
    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 3 );
    timer.Start( 4 );
#endif
    A.MultiplyHMatMainPassData
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 4 );
    timer.Start( 5 );
#endif
    A.MultiplyHMatMainBroadcasts
    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 5 );
    timer.Start( 6 );
#endif
    A.MultiplyHMatMainPostcompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 6 );
#endif

#ifdef TIME_MULTIPLY
    timer.Start( 7 );
#endif
    A.MultiplyHMatFHHPrecompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 7 );
    timer.Start( 8 );
#endif
    A.MultiplyHMatFHHSums
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 8 );
    timer.Start( 9 );
#endif
    A.MultiplyHMatFHHPassData
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 9 );
    timer.Start( 10 );
#endif
    A.MultiplyHMatFHHBroadcasts
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 10 );
    timer.Start( 11 );
#endif
    A.MultiplyHMatFHHPostcompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 11 );
    timer.Start( 12 );
#endif
    A.MultiplyHMatFHHFinalize
    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 12 );
#endif

#ifdef TIME_MULTIPLY
    timer.Start( 13 );
#endif
    C.MultiplyHMatCompress();
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 13 );
#endif

#ifdef TIME_MULTIPLY
    const int commRank = mpi::CommRank( MPI_COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-full-" << commRank << ".log";
    std::ofstream file( os.str().c_str() );

    file << "Form ghost nodes: " << timer.GetTime( 0 ) << " seconds.\n"
         << "Form ghost ranks: " << timer.GetTime( 1 ) << " seconds.\n"
         << "Main precompute:  " << timer.GetTime( 2 ) << " seconds.\n"
         << "Main summations:  " << timer.GetTime( 3 ) << " seconds.\n"
         << "Main pass data:   " << timer.GetTime( 4 ) << " seconds.\n"
         << "Main broadcasts:  " << timer.GetTime( 5 ) << " seconds.\n"
         << "Main postcompute: " << timer.GetTime( 6 ) << " seconds.\n"
         << "FHH precompute:   " << timer.GetTime( 7 ) << " seconds.\n"
         << "FHH summations:   " << timer.GetTime( 8 ) << " seconds.\n"
         << "FHH pass data:    " << timer.GetTime( 9 ) << " seconds.\n"
         << "FHH broadcasts:   " << timer.GetTime( 10 ) << " seconds.\n"
         << "FHH postcompute:  " << timer.GetTime( 11 ) << " seconds.\n"
         << "FHH finalize:     " << timer.GetTime( 12 ) << " seconds.\n"
         << "Compress:         " << timer.GetTime( 13 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatSingleLevelAccumulate
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatSingleLevelAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

#ifdef TIME_MULTIPLY
    Timer timer; 
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Start( 0 );
#endif
    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif
    A.MultiplyHMatFormGhostRanks( B );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 1 );
#endif

    const int startUpdate = 0;
    const int endUpdate = 4;

    const int numLevels = A.NumLevels();
    for( int level=0; level<numLevels; ++level )
    {
        const int startLevel = level;
        const int endLevel = level+1;

#ifdef TIME_MULTIPLY
        timer.Start( 2 );
#endif
        A.MultiplyHMatMainPrecompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 2 );
        timer.Start( 3 );
#endif
        A.MultiplyHMatMainSums
        ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 3 );
        timer.Start( 4 );
#endif
        A.MultiplyHMatMainPassData
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 4 );
        timer.Start( 5 );
#endif
        A.MultiplyHMatMainBroadcasts
        ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 5 );
        timer.Start( 6 );
#endif
        A.MultiplyHMatMainPostcompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 6 );
#endif

#ifdef TIME_MULTIPLY
        timer.Start( 7 );
#endif
        A.MultiplyHMatFHHPrecompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 7 );
        timer.Start( 8 );
#endif
        A.MultiplyHMatFHHSums
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 8 );
        timer.Start( 9 );
#endif
        A.MultiplyHMatFHHPassData
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 9 );
        timer.Start( 10 );
#endif
        A.MultiplyHMatFHHBroadcasts
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 10 );
        timer.Start( 11 );
#endif
        A.MultiplyHMatFHHPostcompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 11 );
        timer.Start( 12 );
#endif
        A.MultiplyHMatFHHFinalize
        ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 12 );
#endif

#ifdef TIME_MULTIPLY
        timer.Start( 13 );
#endif
        C.MultiplyHMatCompress();
#ifdef TIME_MULTIPLY
        mpi::Barrier( MPI_COMM_WORLD );
        timer.Stop( 13 );
#endif
    }

#ifdef TIME_MULTIPLY
    const int commRank = mpi::CommRank( MPI_COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-singleLevel-" << commRank << ".log";
    std::ofstream file( os.str().c_str() );

    file << "Form ghost nodes: " << timer.GetTime( 0 ) << " seconds.\n"
         << "Form ghost ranks: " << timer.GetTime( 1 ) << " seconds.\n"
         << "Main precompute:  " << timer.GetTime( 2 ) << " seconds.\n"
         << "Main summations:  " << timer.GetTime( 3 ) << " seconds.\n"
         << "Main pass data:   " << timer.GetTime( 4 ) << " seconds.\n"
         << "Main broadcasts:  " << timer.GetTime( 5 ) << " seconds.\n"
         << "Main postcompute: " << timer.GetTime( 6 ) << " seconds.\n"
         << "FHH precompute:   " << timer.GetTime( 7 ) << " seconds.\n"
         << "FHH summations:   " << timer.GetTime( 8 ) << " seconds.\n"
         << "FHH pass data:    " << timer.GetTime( 9 ) << " seconds.\n"
         << "FHH broadcasts:   " << timer.GetTime( 10 ) << " seconds.\n"
         << "FHH postcompute:  " << timer.GetTime( 11 ) << " seconds.\n"
         << "FHH finalize:     " << timer.GetTime( 12 ) << " seconds.\n"
         << "Compress:         " << timer.GetTime( 13 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatSingleUpdateAccumulate
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatSingleUpdateAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

#ifdef TIME_MULTIPLY
    Timer timer; 
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Start( 0 );
#endif
    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif
    A.MultiplyHMatFormGhostRanks( B );
#ifdef TIME_MULTIPLY
    mpi::Barrier( MPI_COMM_WORLD );
    timer.Stop( 1 );
#endif

    const int numLevels = A.NumLevels();
    for( int level=0; level<numLevels; ++level )
    {
        const int startLevel = level;
        const int endLevel = level+1;

        for( int update=0; update<4; ++update )
        {
            const int startUpdate = update;
            const int endUpdate = update+1;

#ifdef TIME_MULTIPLY
            timer.Start( 2 );
#endif
            A.MultiplyHMatMainPrecompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 2 );
            timer.Start( 3 );
#endif
            A.MultiplyHMatMainSums
            ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 3 );
            timer.Start( 4 );
#endif
            A.MultiplyHMatMainPassData
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 4 );
            timer.Start( 5 );
#endif
            A.MultiplyHMatMainBroadcasts
            ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 5 );
            timer.Start( 6 );
#endif
            A.MultiplyHMatMainPostcompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 6 );
#endif

#ifdef TIME_MULTIPLY
            timer.Start( 7 );
#endif
            A.MultiplyHMatFHHPrecompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 7 );
            timer.Start( 8 );
#endif
            A.MultiplyHMatFHHSums
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 8 );
            timer.Start( 9 );
#endif
            A.MultiplyHMatFHHPassData
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 9 );
            timer.Start( 10 );
#endif
            A.MultiplyHMatFHHBroadcasts
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 10 );
            timer.Start( 11 );
#endif
            A.MultiplyHMatFHHPostcompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 11 );
            timer.Start( 12 );
#endif
            A.MultiplyHMatFHHFinalize
            ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 12 );
#endif

#ifdef TIME_MULTIPLY
            timer.Start( 13 );
#endif
            C.MultiplyHMatCompress();
#ifdef TIME_MULTIPLY
            mpi::Barrier( MPI_COMM_WORLD );
            timer.Stop( 13 );
#endif
        }
    }

#ifdef TIME_MULTIPLY
    const int commRank = mpi::CommRank( MPI_COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-singleUpdate-" << commRank << ".log";
    std::ofstream file( os.str().c_str() );

    file << "Form ghost nodes: " << timer.GetTime( 0 ) << " seconds.\n"
         << "Form ghost ranks: " << timer.GetTime( 1 ) << " seconds.\n"
         << "Main precompute:  " << timer.GetTime( 2 ) << " seconds.\n"
         << "Main summations:  " << timer.GetTime( 3 ) << " seconds.\n"
         << "Main pass data:   " << timer.GetTime( 4 ) << " seconds.\n"
         << "Main broadcasts:  " << timer.GetTime( 5 ) << " seconds.\n"
         << "Main postcompute: " << timer.GetTime( 6 ) << " seconds.\n"
         << "FHH precompute:   " << timer.GetTime( 7 ) << " seconds.\n"
         << "FHH summations:   " << timer.GetTime( 8 ) << " seconds.\n"
         << "FHH pass data:    " << timer.GetTime( 9 ) << " seconds.\n"
         << "FHH broadcasts:   " << timer.GetTime( 10 ) << " seconds.\n"
         << "FHH postcompute:  " << timer.GetTime( 11 ) << " seconds.\n"
         << "FHH finalize:     " << timer.GetTime( 12 ) << " seconds.\n"
         << "Compress:          " << timer.GetTime( 13 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif

#ifndef RELEASE
    PopCallStack();
#endif
}

