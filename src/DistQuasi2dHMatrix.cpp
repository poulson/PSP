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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::PackedSizes
( std::vector<std::size_t>& packedSizes, 
  const Quasi2dHMatrix<Scalar,Conjugated>& H, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::PackedSizes");
#endif
    const int rank = mpi::CommRank( comm );
    const unsigned p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
    if( p > (1u<<(2*(H._numLevels-1))) )
        throw std::logic_error("More than 4^(numLevels-1) processes.");

    // Initialize for the recursion
    packedSizes.resize( p );
    std::memset( &packedSizes[0], 0, p*sizeof(std::size_t) );
    std::vector<int> localHeights( p );
    std::vector<int> localWidths( p );
    ComputeLocalSizes( localHeights, localWidths, H );

    // Recurse on this shell to compute the packed sizes
    int sourceRankOffset=0, targetRankOffset=0;
    PackedSizesRecursion
    ( packedSizes, localHeights, localWidths, rank, 
      sourceRankOffset, targetRankOffset, p, H );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalHeight
( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::ComputeLocalHeight");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localHeight;
    ComputeLocalDimensionRecursion
    ( localHeight, p, rank, H._xSizeTarget, H._ySizeTarget, H._zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localHeight;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalWidth
( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::ComputeLocalWidth");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localWidth;
    ComputeLocalDimensionRecursion
    ( localWidth, p, rank, H._xSizeSource, H._ySizeSource, H._zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidth;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalSizes
( std::vector<int>& localHeights, std::vector<int>& localWidths,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE    
    PushCallStack("DistQuasi2dHMatrix::ComputeLocalSizes");
    const int p = localHeights.size();
    const int q = localWidths.size();
    if( p != q )
        throw std::logic_error("Vectors are of different lengths");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    ComputeLocalSizesRecursion
    ( &localHeights[0], &localWidths[0], localHeights.size(), 
      H._xSizeSource, H._xSizeTarget, H._ySizeSource, H._ySizeTarget, 
      H._zSize );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::PackedSizesRecursion
( std::vector<std::size_t>& packedSizes, 
  const std::vector<int>& localHeights,
  const std::vector<int>& localWidths,
  int rank, int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SharedQuasi2dHMatrix<Scalar,Conjugated> SharedQuasi2d;

    // TODO: Count the header information
    /*
    const std::size_t headerSize = 
        15*sizeof(int) + 2*sizeof(bool) + 
        sizeof(typename Quasi2d::ShellType);
    for( int j=0; j<teamSize; ++j )
        packedSizes[j] += headerSize;
    */

    if( teamSize == 1 )
    {
#ifndef RELEASE
        if( rank != sourceRankOffset && rank != targetRankOffset )
            throw std::logic_error("Mistake in logic");
#endif
        if( sourceRankOffset == targetRankOffset )
            packedSizes[rank] += H.PackedSize();
        else if( rank == sourceRankOffset )
            packedSizes[rank] += SharedQuasi2d::PackedSourceSize( H );
        else
            packedSizes[rank] += SharedQuasi2d::PackedTargetSize( H );
    }
    else // teamSize >= 2
    {
        const typename Quasi2d::Shell& shell = H._shell;
        switch( shell.type )
        {
        case Quasi2d::NODE:
        {
            const int newTeamSize = teamSize/2;
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    PackedSizesRecursion
                    ( packedSizes, localHeights, localWidths, rank,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
            break;
        }
        case Quasi2d::NODE_SYMMETRIC:
#ifndef RELEASE
            PushCallStack("Nonsymmetric case not yet supported");
#endif
            break;
        case Quasi2d::LOW_RANK:
            // TODO: Count size of DistLowRankMatrix
            break;
        case Quasi2d::DENSE:
#ifndef RELEASE
            throw std::logic_error("Too many processes");
#endif
            break;
        }
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalDimensionRecursion
( int& localDim, int p, int rank, int xSize, int ySize, int zSize )
{
    if( p != 1 )
    {
        if( rank > p/2 )
        {
            ComputeLocalDimensionRecursion
            ( localDim, p/2, rank-p/2, 
              xSize-(xSize/2), ySize-(ySize/2), zSize );
        }
        else
        {
            ComputeLocalDimensionRecursion
            ( localDim, p/2, rank, xSize/2, ySize/2, zSize );
        }
    }
    else
    {
        localDim = xSize*ySize*zSize;
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalSizesRecursion
( int* localHeights, int* localWidths, int teamSize, 
  int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget, 
  int zSize ) 
{
    if( teamSize != 1 )
    {
        ComputeLocalSizesRecursion
        ( localHeights, localWidths, teamSize/2,
          xSizeSource/2, xSizeTarget/2, ySizeSource/2, ySizeTarget/2, zSize );
        ComputeLocalSizesRecursion
        ( &localHeights[teamSize/2], &localWidths[teamSize/2], teamSize/2,
          xSizeSource-(xSizeSource/2), xSizeTarget-(xSizeTarget/2),
          ySizeSource-(ySizeSource/2), ySizeTarget-(ySizeTarget/2),
          zSize );
    }
    else
    {
        localHeights[0] = xSizeTarget*ySizeTarget*zSize;
        localWidths[0] = xSizeSource*ySizeSource*zSize;
    }
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::LocalHeight() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::LocalHeight");
#endif
    int p = mpi::CommSize( _comm );
    int rank = mpi::CommRank( _comm );

    int localHeight;
    ComputeLocalDimensionRecursion
    ( localHeight, p, rank, _xSizeTarget, _ySizeTarget, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localHeight;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::LocalWidth() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::LocalWidth");
#endif
    int p = mpi::CommSize( _comm );
    int rank = mpi::CommRank( _comm );

    int localWidth;
    ComputeLocalDimensionRecursion
    ( localWidth, p, rank, _xSizeSource, _ySizeSource, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidth;
}

template class psp::DistQuasi2dHMatrix<float,false>;
template class psp::DistQuasi2dHMatrix<float,true>;
template class psp::DistQuasi2dHMatrix<double,false>;
template class psp::DistQuasi2dHMatrix<double,true>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,true>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,true>;
