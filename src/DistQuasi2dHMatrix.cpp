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
std::size_t
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::PackedSizes
( std::vector<std::size_t>& packedSizes, 
  const Quasi2dHMatrix<Scalar,Conjugated>& H, MPI_Comm team )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::PackedSizes");
#endif
    const unsigned p = mpi::CommSize( team );
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
    std::size_t totalSize = 0;
    for( unsigned i=0; i<p; ++i )
        totalSize += packedSizes[i];
#ifndef RELEASE
    PopCallStack();
#endif
    return totalSize;
}

template<typename Scalar,bool Conjugated>
std::size_t
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

    std::size_t totalSize = 0;
    for( int i=0; i<p; ++i )
        totalSize += (*headPointers[i]-packedPieces[i]);
#ifndef RELEASE
    PopCallStack();
#endif
    return totalSize;
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
    case Quasi2d::NODE:
    {
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the entire H-matrix on one process
                const int rank = sourceRankOffset;
                packedSizes[rank] += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix
                const int sourceRank = sourceRankOffset;
                const int targetRank = targetRankOffset;

                std::pair<std::size_t,std::size_t> sizes = 
                    SharedQuasi2d::PackedSizes( H );

                packedSizes[sourceRank] += sizes.first;
                packedSizes[targetRank] += sizes.second;
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
        break;
    }
    case Quasi2d::NODE_SYMMETRIC:
    {
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the entire H-matrix on one process
                const int rank = sourceRankOffset;
                packedSizes[rank] += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix
                const int sourceRank = sourceRankOffset;
                const int targetRank = targetRankOffset;

                std::pair<std::size_t,std::size_t> sizes = 
                    SharedQuasi2d::PackedSizes( H );

                packedSizes[sourceRank] += sizes.first;
                packedSizes[targetRank] += sizes.second;
            }
        }
        else
        {
#ifndef RELEASE
            throw std::logic_error("Symmetric case not yet supported");
#endif
        }
        break;
    }
    case Quasi2d::LOW_RANK:
    {
        const int r = shell.data.F->Rank();
        if( teamSize == 1 )
        {
            // Store a shared low-rank matrix                
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;

            // The source and target processes store the matrix rank, 
            // whether they are on the source side, their partner's (MPI) rank,
            // and their factor's entries.
            packedSizes[sourceRank] += 
                2*sizeof(int) + sizeof(bool) + n*r*sizeof(Scalar);
            packedSizes[targetRank] += 
                2*sizeof(int) + sizeof(bool) + m*r*sizeof(Scalar);
        }
        else
        {
            // Store a distributed low-rank matrix

            // Make room for: 
            //   matrix rank, whether we're on the source side, and the
            //   other team's root.
            const std::size_t headerSize = 2*sizeof(int) + sizeof(bool);

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
    case Quasi2d::DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const MatrixType type = D.Type();

        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the type and entries
                const int rank = sourceRankOffset;
                packedSizes[rank] += sizeof(MatrixType);
                if( type == GENERAL )
                    packedSizes[rank] += m*n*sizeof(Scalar);
                else
                    packedSizes[rank] += ((m*m+m)/2)*sizeof(Scalar);
            }
            else
            {
                // Both teams store whether or not they are on the source side
                // and who their partner is
                const int sourceRank = sourceRankOffset;
                const int targetRank = targetRankOffset;
                packedSizes[sourceRank] += sizeof(bool) + sizeof(int);
                packedSizes[targetRank] += sizeof(bool) + sizeof(int);

                // The source side stores the matrix type and entries
                packedSizes[sourceRank] += sizeof(MatrixType);
                if( type == GENERAL )
                    packedSizes[sourceRank] += m*n*sizeof(Scalar);
                else
                    packedSizes[sourceRank] += ((m*m+m)/2)*sizeof(Scalar);
            }
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

    const typename Quasi2d::Shell& shell = H._shell;
    const int m = H._height;
    const int n = H._width;
    switch( shell.type )
    {
    case Quasi2d::NODE:
    {
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the entire H-matrix on one process
                const int rank = sourceRankOffset;
                byte** h = headPointers[rank];
                *((ShellType*)*h) = QUASI2D; *h += sizeof(ShellType);
                H.Pack( *h ); *h += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix 
                const int sourceRank = sourceRankOffset;
                const int targetRank = targetRankOffset;
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                *((ShellType*)*hSource) = SHARED_QUASI2D; 
                *((ShellType*)*hTarget) = SHARED_QUASI2D;
                *hSource += sizeof(ShellType);
                *hTarget += sizeof(ShellType);

                std::pair<std::size_t,std::size_t> sizes =
                    SharedQuasi2d::Pack
                    ( *hSource, *hTarget, sourceRank, targetRank, H );

                *hSource += sizes.first;
                *hTarget += sizes.second;
            }
        }
        else
        {
            // Recurse
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                byte** hTarget = headPointers[targetRankOffset+i];
                *((ShellType*)*hSource) = NODE; *hSource += sizeof(ShellType);
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
        }
        break;
    }
    case Quasi2d::NODE_SYMMETRIC:
    {
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store the entire H-matrix on one process
                const int rank = sourceRankOffset;
                byte** h = headPointers[rank];
                *((ShellType*)*h) = QUASI2D; *h += sizeof(ShellType);
                H.Pack( *h ); *h += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix 
                const int sourceRank = sourceRankOffset;
                const int targetRank = targetRankOffset;
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                *((ShellType*)*hSource) = SHARED_QUASI2D; 
                *((ShellType*)*hTarget) = SHARED_QUASI2D;
                *hSource += sizeof(ShellType);
                *hTarget += sizeof(ShellType);

                std::pair<std::size_t,std::size_t> sizes =
                    SharedQuasi2d::Pack
                    ( *hSource, *hTarget, sourceRank, targetRank, H );

                *hSource += sizes.first;
                *hTarget += sizes.second;
            }
        }
        else
        {
#ifndef RELEASE
            throw std::logic_error("Symmetric case not yet supported");
#endif
        }
        break;
    }
    case Quasi2d::LOW_RANK:
    {
        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int r = U.Width();
        if( teamSize == 1 )
        {
            // Store a shared low-rank representation
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            byte** hSource = headPointers[sourceRank];
            byte** hTarget = headPointers[targetRank];
            *((ShellType*)*hSource) = SHARED_LOW_RANK;
            *((ShellType*)*hTarget) = SHARED_LOW_RANK;
            *hSource += sizeof(ShellType);
            *hTarget += sizeof(ShellType);

            // Store the source data
            *((int*)*hSource) = r;          *hSource += sizeof(int);
            *((bool*)*hSource) = true;      *hSource += sizeof(bool);
            *((int*)*hSource) = targetRank; *hSource += sizeof(int);
            for( int j=0; j<r; ++j )
            {
                std::memcpy( *hSource, V.LockedBuffer(0,j), n*sizeof(Scalar) );
                *hSource += n*sizeof(Scalar);
            }

            // Store the target data
            *((int*)*hTarget) = r;          *hTarget += sizeof(int);
            *((bool*)*hTarget) = false;     *hTarget += sizeof(bool);
            *((int*)*hTarget) = sourceRank; *hTarget += sizeof(int);
            for( int j=0; j<r; ++j )
            {
                std::memcpy( *hTarget, U.LockedBuffer(0,j), m*sizeof(Scalar) );
                *hTarget += m*sizeof(Scalar);
            }
        }
        else
        {
            // Store a distributed low-rank representation
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                byte** hTarget = headPointers[targetRankOffset+i];
                *((ShellType*)*hSource) = DIST_LOW_RANK;
                *((ShellType*)*hTarget) = DIST_LOW_RANK;
                *hSource += sizeof(ShellType);
                *hTarget += sizeof(ShellType);
            }

            // Store the source data
            int rowOffset = 0;
            for( int i=0; i<teamSize; ++i )
            {
                const int sourceRank = sourceRankOffset + i;
                byte** hSource = headPointers[sourceRank];
                const int localWidth = localWidths[sourceRank];

                *((int*)*hSource) = r;                *hSource += sizeof(int);
                *((bool*)*hSource) = true;            *hSource += sizeof(bool);
                *((int*)*hSource) = targetRankOffset; *hSource += sizeof(int);

                for( int j=0; j<r; ++j )
                {
                    std::memcpy
                    ( *hSource, V.LockedBuffer(rowOffset,j), 
                      localWidth*sizeof(Scalar) );
                    *hSource += localWidth*sizeof(Scalar);
                }
                rowOffset += localWidth;
            }

            // Store the target data
            rowOffset = 0;
            for( int i=0; i<teamSize; ++i )
            {
                const int targetRank = targetRankOffset + i;
                byte** hTarget = headPointers[targetRank];
                const int localHeight = localHeights[targetRank];

                *((int*)*hTarget) = r;                *hTarget += sizeof(int);
                *((bool*)*hTarget) = false;           *hTarget += sizeof(bool);
                *((int*)*hTarget) = sourceRankOffset; *hTarget += sizeof(int);

                for( int j=0; j<r; ++j )
                {
                    std::memcpy
                    ( *hTarget, U.LockedBuffer(rowOffset,j),
                      localHeight*sizeof(Scalar) );
                    *hTarget += localHeight*sizeof(Scalar);
                }
                rowOffset += localHeight;
            }
        }
        break;
    }
    case Quasi2d::DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const MatrixType type = D.Type();
        if( teamSize == 1 )
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store a serial dense matrix
                const int rank = sourceRankOffset;
                byte** h = headPointers[rank];
                *((ShellType*)*h) = DENSE; *h += sizeof(ShellType);

                *((MatrixType*)*h) = type; *h += sizeof(MatrixType);
                if( type == GENERAL )
                {
                    for( int j=0; j<n; ++j )
                    {
                        std::memcpy
                        ( *h, D.LockedBuffer(0,j), m*sizeof(Scalar) );
                        *h += m*sizeof(Scalar);
                    }
                }
                else
                {
                    for( int j=0; j<n; ++j )
                    {
                        std::memcpy
                        ( *h, D.LockedBuffer(j,j), (m-j)*sizeof(Scalar) );
                        *h += (m-j)*sizeof(Scalar);
                    }
                }
            }
            else
            {
                // Store a shared dense matrix
                const int sourceRank = sourceRankOffset;
                const int targetRank = targetRankOffset;
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                *((ShellType*)*hSource) = SHARED_DENSE; 
                *((ShellType*)*hTarget) = SHARED_DENSE;
                *hSource += sizeof(ShellType);
                *hTarget += sizeof(ShellType);

                // Store the source data
                *((bool*)*hSource) = true;       *hSource += sizeof(bool);
                *((int*)*hSource) = targetRank;  *hSource += sizeof(int);
                *((MatrixType*)*hSource) = type; *hSource += sizeof(MatrixType);
                if( type == GENERAL )
                {
                    for( int j=0; j<n; ++j )
                    {
                        std::memcpy
                        ( *hSource, D.LockedBuffer(0,j), m*sizeof(Scalar) );
                        *hSource += m*sizeof(Scalar);
                    }
                }
                else
                {
                    for( int j=0; j<n; ++j )
                    {
                        std::memcpy
                        ( *hSource, D.LockedBuffer(j,j), (m-j)*sizeof(Scalar) );
                        *hSource += (m-j)*sizeof(Scalar);
                    }
                }

                // Store the target data
                *((bool*)*hTarget) = false;     *hTarget += sizeof(bool);
                *((int*)*hTarget) = sourceRank; *hTarget += sizeof(int);
            }
        }
#ifndef RELEASE
        else
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
        localDim = xSize*ySize*zSize;
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
    int teamSize = mpi::CommSize( _team );
    int teamRank = mpi::CommRank( _team );

    int localHeight;
    ComputeLocalDimensionRecursion
    ( localHeight, teamSize, teamRank, _xSizeTarget, _ySizeTarget, _zSize );
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
    int teamSize = mpi::CommSize( _team );
    int teamRank = mpi::CommRank( _team );

    int localWidth;
    ComputeLocalDimensionRecursion
    ( localWidth, teamSize, teamRank, _xSizeSource, _ySizeSource, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidth;
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::Unpack
( const byte* packedDistHMatrix, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Unpack");
#endif
    const byte* head = packedDistHMatrix;
    UnpackRecursion( head, *this, comm, comm );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedDistHMatrix);
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::UnpackRecursion
( const byte*& head, DistQuasi2dHMatrix<Scalar,Conjugated>& H, 
  MPI_Comm comm, MPI_Comm team )
{
    H._comm = comm;
    H._team = team;

    // Read in the header information
    H._height             = *((int*)head);  head += sizeof(int);
    H._width              = *((int*)head);  head += sizeof(int);
    H._numLevels          = *((int*)head);  head += sizeof(int);
    H._maxRank            = *((int*)head);  head += sizeof(int);
    H._sourceOffset       = *((int*)head);  head += sizeof(int);
    H._targetOffset       = *((int*)head);  head += sizeof(int);
    H._symmetric          = *((bool*)head); head += sizeof(bool);
    H._stronglyAdmissible = *((bool*)head); head += sizeof(bool);
    H._xSizeSource        = *((int*)head);  head += sizeof(int);
    H._xSizeTarget        = *((int*)head);  head += sizeof(int);
    H._ySizeSource        = *((int*)head);  head += sizeof(int);
    H._ySizeTarget        = *((int*)head);  head += sizeof(int);
    H._zSize              = *((int*)head);  head += sizeof(int);
    H._xSource            = *((int*)head);  head += sizeof(int);
    H._xTarget            = *((int*)head);  head += sizeof(int);
    H._ySource            = *((int*)head);  head += sizeof(int);
    H._yTarget            = *((int*)head);  head += sizeof(int);

    // Delete the old shell
    Shell& shell = H._shell;
    switch( shell.type ) 
    {
    case NODE:            delete shell.data.node; break;
    case NODE_SYMMETRIC:  delete shell.data.nodeSymmetric; break;
    case DIST_LOW_RANK:   delete shell.data.DF; break;
    case SHARED_QUASI2D:  delete shell.data.SH; break;
    case SHARED_LOW_RANK: delete shell.data.SF; break;
    case SHARED_DENSE:    delete shell.data.SD; break;
    case QUASI2D:         delete shell.data.H; break;
    case DENSE:           delete shell.data.D; break;
    case EMPTY: break;
    }

    // Read in the information for the new shell
    shell.type = *((ShellType*)head); head += sizeof(ShellType);
    const int m = H._height;
    const int n = H._width;
    switch( shell.type )
    {
    case NODE:
    { 
        shell.data.node = 
            new Node
            ( H._xSizeSource, H._xSizeTarget,
              H._ySizeSource, H._ySizeTarget, H._zSize, team );
        Node& node = *shell.data.node;

        if( node.inRightTeam )
        {
            // Our process owns the bottom-right block but participates in all 
            // but the top-left block.
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>;
                    if( s >= 2 || t >= 2 )
                    {
                        UnpackRecursion
                        ( head, node.Child(t,s), comm, node.childTeam );
                    }
                    else
                    {
                        node.Child(t,s)._shell.type = EMPTY;
                    }
                }
            }
        }
        else
        {
            // Our process owns the top-left block but participates in all but
            // the bottom-right block
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>;    
                    if( s < 2 || t < 2 )
                    {
                        UnpackRecursion
                        ( head, node.Child(t,s), comm, node.childTeam );
                    }
                    else
                    {
                        node.Child(t,s)._shell.type = EMPTY;
                    }
                }
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
    {
        shell.data.DF = new DistLowRankMatrix<Scalar,Conjugated>;
        DistLowRankMatrix<Scalar,Conjugated>& DF = *shell.data.DF;

        DF.height = m;
        DF.width = n;
        DF.comm = comm;
        DF.team = team;

        DF.rank            = *((int*)head);  head += sizeof(int);
        DF.inSourceTeam    = *((bool*)head); head += sizeof(bool);
        DF.rootOfOtherTeam = *((int*)head);  head += sizeof(int);

        DF.D.SetType( GENERAL );
        if( DF.inSourceTeam )
        {
            const int localWidth = this->LocalWidth();
            DF.D.Resize( localWidth, DF.rank );
            for( int j=0; j<DF.rank; ++j )
            {
                std::memcpy
                ( DF.D.Buffer(0,j), head, localWidth*sizeof(Scalar) );
                head += localWidth*sizeof(Scalar);
            }
        }
        else
        {
            const int localHeight = this->LocalHeight();
            DF.D.Resize( localHeight, DF.rank );
            for( int j=0; j<DF.rank; ++j )
            {
                std::memcpy
                ( DF.D.Buffer(0,j), head, localHeight*sizeof(Scalar) );
                head += localHeight*sizeof(Scalar);
            }
        }
        break;
    }
    case SHARED_QUASI2D:
    {
        typedef SharedQuasi2dHMatrix<Scalar,Conjugated> SharedQuasi2d;

        shell.data.SH = new SharedQuasi2d;
        SharedQuasi2d& SH = *shell.data.SH;

        std::size_t packedSize = SH.Unpack( head );
        head += packedSize;
        break;
    }
    case SHARED_LOW_RANK:
    {
        shell.data.SF = new SharedLowRankMatrix<Scalar,Conjugated>;
        SharedLowRankMatrix<Scalar,Conjugated>& SF = *shell.data.SF;

        SF.height = m;
        SF.width = n;

        SF.rank          = *((int*)head);  head += sizeof(int);
        SF.ownSourceSide = *((bool*)head); head += sizeof(bool);
        SF.partner       = *((int*)head);  head += sizeof(int);

        SF.D.SetType( GENERAL );
        if( SF.ownSourceSide )
        {
            SF.D.Resize( n, SF.rank );
            for( int j=0; j<SF.rank; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, n*sizeof(Scalar) ); 
                head += n*sizeof(Scalar);
            }
        }
        else
        {
            SF.D.Resize( m, SF.rank );
            for( int j=0; j<SF.rank; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        break;
    }
    case SHARED_DENSE:
    {
        shell.data.SD = new SharedDenseMatrix<Scalar>;
        SharedDenseMatrix<Scalar>& SD = *shell.data.SD;

        SD.height = m;
        SD.width = n;

        SD.ownSourceSide = *((bool*)head); head += sizeof(bool);
        SD.partner       = *((int*)head);  head += sizeof(int);
        
        if( SD.ownSourceSide )
        {
            const MatrixType type = *((MatrixType*)head); 
            head += sizeof(MatrixType);

            SD.D.SetType( type );
            if( type == GENERAL )
            {
                for( int j=0; j<n; ++j )
                {
                    std::memcpy( SD.D.Buffer(0,j), head, m*sizeof(Scalar) );
                    head += m*sizeof(Scalar);
                }
            }
            else
            {
                for( int j=0; j<n; ++j )
                {
                    std::memcpy( SD.D.Buffer(j,j), head, (m-j)*sizeof(Scalar) );
                    head += (m-j)*sizeof(Scalar);
                }
            }
        }
        break;
    }
    case QUASI2D:
    {
        shell.data.H = new Quasi2dHMatrix<Scalar,Conjugated>;
        Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;

        const std::size_t packedSize = H.Unpack( head );
        head += packedSize;
        break;
    }
    case DENSE:
    {
        shell.data.D = new DenseMatrix<Scalar>;
        DenseMatrix<Scalar>& D = *shell.data.D;

        const MatrixType type = *((MatrixType*)head); 
        head += sizeof(MatrixType);
        D.SetType( type );
        if( type == GENERAL )
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        else
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( D.Buffer(j,j), head, (m-j)*sizeof(Scalar) );
                head += (m-j)*sizeof(Scalar);
            }
        }
        break;
    }
    case EMPTY:
        break;
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
