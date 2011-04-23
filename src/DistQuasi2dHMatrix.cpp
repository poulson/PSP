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
#include "psp.hpp"

//----------------------------------------------------------------------------//
// Public static routines                                                     //
//----------------------------------------------------------------------------//
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ScatteredSizes
( std::vector<std::size_t>& scatteredSizes, 
  const Quasi2dHMatrix<Scalar,Conjugated>& H, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ScatteredSizes");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    scatteredSizes.resize( p );
    std::memset( &scatteredSizes[0], 0, p*sizeof(std::size_t) );

    // Recurse on this shell to compute the packed sizes
    int sourceRankOffset=0, targetRankOffset=0;
    CountScatteredShellSizes
    ( scatteredSizes, rank, p, sourceRankOffset, targetRankOffset, H );

#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::CountScatteredShellSizes
( std::vector<std::size_t>& scatteredSizes,
  int rank, int p, int sourceRankOffset, int targetRankOffset,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;

    if( p == 1 )
    {
#ifndef RELEASE
        if( rank != sourceRankOffset && rank != targetRankOffset )
            throw std::logic_error("Mistake in logic");
#endif
        if( sourceRankOffset == targetRankOffset )
            scatteredSizes[rank] += H.PackedSize();
        else if( rank == sourceRankOffset )
            CountSizeOfSourceSideOfLeaf( scatteredSizes[rank], H );
        else
            CountSizeOfTargetSideOfLeaf( scatteredSizes[rank], H );
    }
    else if( p == 2 )
    {
        // Every process stores the basic H-matrix information
        const std::size_t headerSize = 
            15*sizeof(int) + 2*sizeof(bool) + 
            sizeof(typename Quasi2d::ShellType);
        for( int j=0; j<p; ++j )
            scatteredSizes[j] += headerSize;

        const typename Quasi2d::Shell& shell = H._shell;
        switch( shell.type )
        {
        case Quasi2d::NODE:
            // HERE: Branch based on your rank
            break;
        case Quasi2d::NODE_SYMMETRIC:
#ifndef RELEASE
            throw std::logic_error("Nonsymmetric case not yet supported");
#endif
            break;
        case Quasi2d::LOW_RANK:
            break;
        case Quasi2d::DENSE:
            break;
        }
    }
    else // p >= 4
    {
        // Every process stores the basic H-matrix information
        const std::size_t headerSize = 
            15*sizeof(int) + 2*sizeof(bool) + 
            sizeof(typename Quasi2d::ShellType);
        for( int j=0; j<p; ++j )
            scatteredSizes[j] += headerSize;

        const typename Quasi2d::Shell& shell = H._shell;
        switch( shell.type )
        {
        case Quasi2d::NODE:
            break;
        case Quasi2d::NODE_SYMMETRIC:
#ifndef RELEASE
            PushCallStack("Nonsymmetric case not yet supported");
#endif
            break;
        case Quasi2d::LOW_RANK:
            break;
        case Quasi2d::DENSE:
            break;
        }
    }
}

template class psp::DistQuasi2dHMatrix<float,false>;
template class psp::DistQuasi2dHMatrix<float,true>;
template class psp::DistQuasi2dHMatrix<double,false>;
template class psp::DistQuasi2dHMatrix<double,true>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,true>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,true>;
