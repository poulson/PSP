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
#include "./MultiplyHMatMain-incl.hpp"
#include "./MultiplyHMatFHH-incl.hpp"
#include "./MultiplyHMatUpdates-incl.hpp"

// C := alpha A B
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
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

    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();

    MPI_Comm comm = A._teams->Team( 0 );
    const int rank = mpi::CommRank( comm );

    if( rank == 0 || rank == 16 )
    {
        A.LatexWriteLocalStructure("A");
        B.LatexWriteLocalStructure("B");
    }

    mpi::Barrier( comm );
    if( rank == 0 )
    {
        std::cout << "FormGhostRanks...";
        std::cout.flush();
    }
    A.MultiplyHMatFormGhostRanks( B );
    mpi::Barrier( comm );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;

    if( rank == 0 )
    {
        std::cout << "MainPrecompute...";
        std::cout.flush();
    }
    A.MultiplyHMatMainPrecompute( alpha, B, C );
    mpi::Barrier( comm );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;

    if( rank == 0 )
    {
        std::cout << "MainSums...";
        std::cout.flush();
    }
    A.MultiplyHMatMainSums( B, C );
    mpi::Barrier( comm );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;

    if( rank == 0 )
    {
        std::cout << "MainPassData...";
        std::cout.flush();
    }
    A.MultiplyHMatMainPassData( alpha, B, C );
    mpi::Barrier( comm );
    if( rank == 0 )
        std::cout << "DONE" << std::endl;

    A.MultiplyHMatMainBroadcasts( B, C );
    A.MultiplyHMatMainPostcompute( alpha, B, C );

    A.MultiplyHMatFHHPrecompute( alpha, B, C );
    A.MultiplyHMatFHHSums( alpha, B, C );
    A.MultiplyHMatFHHPassData( alpha, B, C );
    A.MultiplyHMatFHHBroadcasts( alpha, B, C );
    A.MultiplyHMatFHHPostcompute( alpha, B, C );
    A.MultiplyHMatFHHFinalize( B, C );

    C.MultiplyHMatUpdates();
#ifndef RELEASE
    PopCallStack();
#endif
}

