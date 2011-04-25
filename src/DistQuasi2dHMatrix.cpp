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
    PackedSizesRecursion( packedSizes, localHeights, localWidths, 0, 0, p, H );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::Pack
( std::vector<byte*>& packedPieces, 
  const Quasi2dHMatrix<Scalar,Conjugated>& H, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Pack");
#endif
    const int p = mpi::CommSize( comm );
    std::vector<byte**> headPointers(p); 
    for( int i=0; i<p; ++i )
        headPointers[i] = &packedPieces[i];
    
    std::vector<int> localHeights( p );
    std::vector<int> localWidths( p );
    ComputeLocalSizes( localHeights, localWidths, H );
    PackRecursion( headPointers, localHeights, localWidths, 0, 0, p, H );
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
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SharedQuasi2dHMatrix<Scalar,Conjugated> SharedQuasi2d;

    const std::size_t headerSize = 
        15*sizeof(int) + 2*sizeof(bool) + sizeof(ShellType);
    // Add the header space to the source and target teams
    for( int i=0; i<teamSize; ++i )
        packedSizes[sourceRankOffset+i] += headerSize;
    for( int i=0; i<teamSize; ++i )
        packedSizes[targetRankOffset+i] += headerSize;

    const typename Quasi2d::Shell& shell = H._shell;
    const int m = H._height;
    const int n = H._width;
    switch( shell.type )
    {
    case NODE:
    {
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the entire H-matrix on one process
                packedSizes[sourceRankOffset] += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix
                std::size_t sourceSize, targetSize;
                SharedQuasi2d::PackedSizes( sourceSize, targetSize, H );
                packedSizes[sourceRankOffset] += sourceSize;
                packedSizes[targetRankOffset] += targetSize;
            }
        }
        else
        {
            // Recurse
            const int newTeamSize = teamSize/2;
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    PackedSizesRecursion
                    ( packedSizes, localHeights, localWidths,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
        }
    }
    case NODE_SYMMETRIC:
    {
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the entire H-matrix on one process
                packedSizes[sourceRankOffset] += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix
                std::size_t sourceSize, targetSize;
                SharedQuasi2d::PackedSizes( sourceSize, targetSize, H );
                packedSizes[sourceRankOffset] += sourceSize;
                packedSizes[targetRankOffset] += targetSize;
            }
        }
        else
        {
#ifndef RELEASE
            throw std::logic_error("Symmetric case not yet supported");
#endif
        }
    }
    case LOW_RANK:
    {
        if( teamSize == 1 )
        {
            // Store a shared low-rank matrix                
            const DenseMatrix<Scalar>& shell.data.F->U;
            const DenseMatrix<Scalar>& shell.data.F->V;
            const int r = U.Width();

            // The source and target processes store the matrix rank and their
            // factor's entries.
            packedSizes[sourceRankOffset] += sizeof(int) + n*r*sizeof(Scalar);
            packedSizes[targetRankOffset] += sizeof(int) + m*r*sizeof(Scalar);
        }
        else
        {
            // Store a distributed low-rank matrix
            const DenseMatrix<Scalar>& U = shell.data.F->U;
            const DenseMatrix<Scalar>& V = shell.data.F->V;
            const int r = U.Width();

            // Make room for: 
            //   rank, my team's root, my team's size, other team's root
            const std::size_t headerSize = 4*sizeof(int);

            // Write out the source information
            for( int i=0; i<teamSize; ++i )
            {
                const int sourceRank = sourceRankOffset + i;
                packedSizes[sourceRank] += 
                    headerSize + localWidths[sourceRank]*r*sizeof(Scalar);
            }

            // Write out the target information
            for( int i=0; i<teamSize; ++i )
            {
                const int targetRank = targetRankOffset + i;
                packedSizes[targetRank] += 
                    headerSize + localHeights[targetRank]*r*sizeof(Scalar);
            } 
        }
        break;
    }
    case DENSE:
    {
        const DenseMatrix<Scalar>& *shell.data.D;

        if( teamSize == 1 )
        {
            // We can just store the dense matrix at the source rank since, 
            // if there are two processes the source stores it, and if there
            // is one process the source and target ranks are the same.
            const DenseMatrix<Scalar>& *shell.data.D;
            const MatrixType type = D.Type();

            packedSizes[sourceRankOffset] += sizeof(MatrixType);
            if( type == GENERAL )
                packedSizes[sourceRankOffset] += m*n*sizeof(Scalar);
            else
                packedSizes[sourceRankOffset] += ((m*m+m)/2)*sizeof(Scalar);
        }
        else
        {
#ifndef RELEASE
            throw std::logic_error("Too many processes");
#endif
        }
        break;
    }
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::PackRecursion
( std::vector<byte**>& headPointers,
  const std::vector<int>& localHeights,
  const std::vector<int>& localWidths,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SharedQuasi2dHMatrix<Scalar,Conjugated> SharedQuasi2d;

    // Write the header information for every process in the source team
    for( int i=0; i<teamSize; ++i )
    {
        const int sourceRank = sourceRankOffset + i;
        byte** h = headPointers[sourceRank];

        *((int*)*h) = H._height;              *h += sizeof(int);
        *((int*)*h) = H._width;               *h += sizeof(int);
        *((int*)*h) = H._numLevels;           *h += sizeof(int);
        *((int*)*h) = H._maxRank;             *h += sizeof(int);
        *((int*)*h) = H._sourceOffset;        *h += sizeof(int);
        *((int*)*h) = H._targetOffset;        *h += sizeof(int);
        *((bool*)*h) = H._symmetric;          *h += sizeof(bool);
        *((bool*)*h) = H._stronglyAdmissible; *h += sizeof(bool);
        *((int*)*h) = H._xSizeSource;         *h += sizeof(int);
        *((int*)*h) = H._xSizeTarget;         *h += sizeof(int);
        *((int*)*h) = H._ySizeSource;         *h += sizeof(int);
        *((int*)*h) = H._ySizeTarget;         *h += sizeof(int);
        *((int*)*h) = H._zSize;               *h += sizeof(int);
        *((int*)*h) = H._xSource;             *h += sizeof(int);
        *((int*)*h) = H._xTarget;             *h += sizeof(int);
        *((int*)*h) = H._ySource;             *h += sizeof(int);
        *((int*)*h) = H._yTarget;             *h += sizeof(int);
    }

    // Write the header information for every process in the target team
    // (shamelessly copy and pasted from above...)
    for( int i=0; i<teamSize; ++i )
    {
        const int targetRank = targetRankOffset + i;
        byte** h = headPointers[targetRank];

        *((int*)*h) = H._height;              *h += sizeof(int);
        *((int*)*h) = H._width;               *h += sizeof(int);
        *((int*)*h) = H._numLevels;           *h += sizeof(int);
        *((int*)*h) = H._maxRank;             *h += sizeof(int);
        *((int*)*h) = H._sourceOffset;        *h += sizeof(int);
        *((int*)*h) = H._targetOffset;        *h += sizeof(int);
        *((bool*)*h) = H._symmetric;          *h += sizeof(bool);
        *((bool*)*h) = H._stronglyAdmissible; *h += sizeof(bool);
        *((int*)*h) = H._xSizeSource;         *h += sizeof(int);
        *((int*)*h) = H._xSizeTarget;         *h += sizeof(int);
        *((int*)*h) = H._ySizeSource;         *h += sizeof(int);
        *((int*)*h) = H._ySizeTarget;         *h += sizeof(int);
        *((int*)*h) = H._zSize;               *h += sizeof(int);
        *((int*)*h) = H._xSource;             *h += sizeof(int);
        *((int*)*h) = H._xTarget;             *h += sizeof(int);
        *((int*)*h) = H._ySource;             *h += sizeof(int);
        *((int*)*h) = H._yTarget;             *h += sizeof(int);
    }

    switch( shell.type )
    {
    case NODE:
    {
        break;
    }
    case NODE_SYMMETRIC:
    {
        break;
    }
    case LOW_RANK:
    {
        break;
    }
    case DENSE:
    {

        break;
    }
    }
    /*
    if( teamSize == 1 )
    {
        if( sourceRankOffset == targetRankOffset )
        {
            byte** h = headPointers[sourceRankOffset];

            *((ShellType*)*h) = QUASI2D;
            H.Pack( *h );        
            *h += H.PackedSize();
        }
        else
        {
            // HERE
            // if( admissible )
            // {
            // ...
            // }
            // else
            // {
            byte** hSource = headPointers[sourceRankOffset];
            byte** hTarget = headPointers[targetRankOffset];

            *((ShellType*)*hSource) = SHARED_QUASI2D;
            *((ShellType*)*hTarget) = SHARED_QUASI2D;
            SharedQuasi2d::Pack
            ( *hSource, *hTarget, sourceRankOffset, targetRankOffset, H );

            std::size_t sourceSize, targetSize;
            SharedQuasi2d::PackedSizes( sourceSize, targetSize, H );
            *hSource += sourceSize;
            *hTarget += targetSize;
            // }
        }
    }
    else // teamSize >= 2
    {
        const typename Quasi2d::Shell& shell = H._shell;
        switch( shell.type )
        {
        case Quasi2d::NODE:
        {
            // Write out the shell types for the source and target teams
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                *((ShellType*)*hSource) = NODE; *hSource += sizeof(ShellType);

                byte** hTarget = headPointers[targetRankOffset+i];
                *((ShellType*)*hTarget) = NODE; *hTarget += sizeof(ShellType);
            }

            const int newTeamSize = teamSize/2;
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    PackRecursion
                    ( headPointers, localHeights, localWidths,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
            break;
        }
        case Quasi2d::NODE_SYMMETRIC:
#ifndef RELEASE
            throw std::logic_error("Nonsymmetric case not yet supported");
#endif
            break;
        case Quasi2d::LOW_RANK:
#ifndef RELEASE
            if( sourceRankOffset == targetRankOffset )
                throw std::logic_error("Offsets were equal off the diagonal");
#endif
            // Write out the shell types for the source and target teams
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                *((ShellType*)*hSource) = DIST_LOW_RANK;
                *hSource += sizeof(ShellType);

                byte** hTarget = headPointers[targetRankOffset+i];
                *((ShellType*)*hTarget) = DIST_LOW_RANK; 
                *hTarget += sizeof(ShellType);
            }

            // TODO
            break;
        case Quasi2d::DENSE:
#ifndef RELEASE
            throw std::logic_error("Too many processes");
#endif
            break;
        }
    }
    */
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
