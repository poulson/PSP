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
  const Quasi2dHMatrix<Scalar,Conjugated>& H, const Subcomms& subcomms )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::PackedSizes");
#endif
    MPI_Comm comm = subcomms.Subcomm(0);
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
  const Quasi2dHMatrix<Scalar,Conjugated>& H, const Subcomms& subcomms )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Pack");
#endif
    MPI_Comm comm = subcomms.Subcomm(0);
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
    if( sourceRankOffset != targetRankOffset )
    {
        for( int i=0; i<teamSize; ++i )
            packedSizes[targetRankOffset+i] += headerSize;
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
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store the entire H-matrix on one process
                packedSizes[sourceRank] += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix
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
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store the entire H-matrix on one process
                packedSizes[sourceRank] += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix
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
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store a serial low-rank matrix
                packedSizes[sourceRank] += sizeof(int) + (m+n)*r*sizeof(Scalar);
            }
            else
            {
                // Store a shared low-rank matrix                

                // The source and target processes store the matrix rank, 
                // whether they are on the source side, their partner's (MPI) 
                // rank, and their factor's entries.
                packedSizes[sourceRank] += 
                    2*sizeof(int) + sizeof(bool) + n*r*sizeof(Scalar);
                packedSizes[targetRank] += 
                    2*sizeof(int) + sizeof(bool) + m*r*sizeof(Scalar);
            }
        }
        else
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store a distributed low-rank matrix
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    const int localHeight = localHeights[sourceRank];
                    const int localWidth = localWidths[sourceRank];
                    packedSizes[sourceRank] += 
                        sizeof(int) + (localHeight+localWidth)*r*sizeof(Scalar);
                }
            }
            else
            {
                // Store a distributed shared low-rank matrix

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
        }
        break;
    }
    case Quasi2d::DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const MatrixType type = D.Type();

        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store the type and entries
                packedSizes[sourceRank] += sizeof(MatrixType);
                if( type == GENERAL )
                    packedSizes[sourceRank] += m*n*sizeof(Scalar);
                else
                    packedSizes[sourceRank] += ((m*m+m)/2)*sizeof(Scalar);
            }
            else
            {
                // Both teams store whether or not they are on the source side
                // and who their partner is
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
    if( targetRankOffset != sourceRankOffset )
    {
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
    }

    const int rank = mpi::CommRank( MPI_COMM_WORLD );
    if( rank == 0 )
        std::cout << "souce/target offsets: " << sourceRankOffset 
                  << ", " << targetRankOffset << std::endl;

    const typename Quasi2d::Shell& shell = H._shell;
    const int m = H._height;
    const int n = H._width;
    switch( shell.type )
    {
    case Quasi2d::NODE:
    {
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store the entire H-matrix on one process
                if( rank == 0 )
                {
                    std::cout << "Packing QUASI2D for " << sourceRank 
                              << std::endl;
                }
                byte** h = headPointers[sourceRank];
                *((ShellType*)*h) = QUASI2D; *h += sizeof(ShellType);
                H.Pack( *h ); *h += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix 
                if( rank == 0 )
                {
                    std::cout << "Packing SHARED_QUASI2D for " 
                              << sourceRank << " and " << targetRank 
                              << std::endl;
                }
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
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            if( rank == 0 )
            {
                std::cout << "Packing NODE for " 
                          << sourceRankOffset << "-" 
                          << sourceRankOffset+teamSize-1 << " and "
                          << targetRankOffset << "-"
                          << targetRankOffset+teamSize-1 << std::endl;
            }
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                *((ShellType*)*hSource) = NODE; *hSource += sizeof(ShellType);
            }
            if( sourceRankOffset != targetRankOffset )
            {
                for( int i=0; i<teamSize; ++i )
                {
                    byte** hTarget = headPointers[targetRankOffset+i];
                    *((ShellType*)*hTarget) = NODE; 
                    *hTarget += sizeof(ShellType);
                }
            }
            const int newTeamSize = teamSize/2;
            // Top-left block
            for( int t=0; t<2; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    if( rank == 0 )
                    {
                        std::cout << "(t,s)=(" << t << "," << s << ")" 
                                  << std::endl;
                    }
                    PackRecursion
                    ( headPointers, localHeights, localWidths,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
            // Top-right block
            for( int t=0; t<2; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    if( rank == 0 )
                    {
                        std::cout << "(t,s)=(" << t << "," << s << ")" 
                                  << std::endl;
                    }
                    PackRecursion
                    ( headPointers, localHeights, localWidths,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
            // Bottom-left block
            for( int t=2; t<4; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    if( rank == 0 )
                    {
                        std::cout << "(t,s)=(" << t << "," << s << ")" 
                                  << std::endl;
                    }
                    PackRecursion
                    ( headPointers, localHeights, localWidths,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
            // Bottom-right block
            for( int t=2; t<4; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    if( rank == 0 )
                    {
                        std::cout << "(t,s)=(" << t << "," << s << ")" 
                                  << std::endl;
                    }
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
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store the entire H-matrix on one process
                byte** h = headPointers[sourceRank];
                *((ShellType*)*h) = QUASI2D; *h += sizeof(ShellType);
                H.Pack( *h ); *h += H.PackedSize();
            }
            else
            {
                // Store a shared H-matrix 
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
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store a serial low-rank representation
                if( rank == 0 )
                {
                    std::cout << "Packing LOW_RANK for " << sourceRank 
                              << std::endl;
                }
                byte** h = headPointers[sourceRank];
                *((ShellType*)*h) = LOW_RANK; *h += sizeof(ShellType);

                // Store the rank and matrix entries
                *((int*)*h) = r; *h += sizeof(int);
                for( int j=0; j<r; ++j )
                {
                    std::memcpy( *h, U.LockedBuffer(0,j), m*sizeof(Scalar) );
                    *h += m*sizeof(Scalar);
                }
                for( int j=0; j<r; ++j )
                {
                    std::memcpy( *h, V.LockedBuffer(0,j), n*sizeof(Scalar) );
                    *h += n*sizeof(Scalar);
                }
            }
            else
            {
                // Store a shared low-rank representation
                if( rank == 0 )
                {
                    std::cout << "Packing SHARED_LOW_RANK for " 
                              << sourceRank << " and " << targetRank 
                              << std::endl;
                }

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
                    std::memcpy
                    ( *hSource, V.LockedBuffer(0,j), n*sizeof(Scalar) );
                    *hSource += n*sizeof(Scalar);
                }

                // Store the target data
                *((int*)*hTarget) = r;          *hTarget += sizeof(int);
                *((bool*)*hTarget) = false;     *hTarget += sizeof(bool);
                *((int*)*hTarget) = sourceRank; *hTarget += sizeof(int);
                for( int j=0; j<r; ++j )
                {
                    std::memcpy
                    ( *hTarget, U.LockedBuffer(0,j), m*sizeof(Scalar) );
                    *hTarget += m*sizeof(Scalar);
                }
            }
        }
        else
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Store a distributed low-rank representation
                if( rank == 0 )
                {
                    std::cout << "Packing DIST_LOW_RANK for "
                              << sourceRankOffset << "-"
                              << sourceRankOffset+teamSize-1 << std::endl;
                }
                int rowOffset = 0;
                int colOffset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    // Store the header information
                    const int sourceRank = sourceRankOffset + i;
                    byte** h = headPointers[sourceRank];
                    *((ShellType*)*h) = DIST_LOW_RANK;
                    *h += sizeof(ShellType);
                    *((int*)*h) = r;

                    // Store our local U and V
                    const int localHeight = localHeights[sourceRank];
                    const int localWidth = localWidths[sourceRank];
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( h, U.LockedBuffer(rowOffset,j), 
                          localHeight*sizeof(Scalar) );
                        *h += localHeight*sizeof(Scalar);
                    }
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( h, V.LockedBuffer(colOffset,j),
                          localWidth*sizeof(Scalar) );
                        *h += localWidth*sizeof(Scalar);
                    }
                    rowOffset += localHeight;
                    colOffset += localWidth;
                }
            }
            else
            {
                // Store a distributed shared low-rank representation
                if( rank == 0 )
                {
                    std::cout << "Packing DIST_SHARED_LOW_RANK for " 
                              << sourceRankOffset << "-" 
                              << sourceRankOffset+teamSize-1 << " and "
                              << targetRankOffset << "-"
                              << targetRankOffset+teamSize-1 << std::endl;
                }
                for( int i=0; i<teamSize; ++i )
                {
                    byte** hSource = headPointers[sourceRankOffset+i];
                    byte** hTarget = headPointers[targetRankOffset+i];
                    *((ShellType*)*hSource) = DIST_SHARED_LOW_RANK;
                    *((ShellType*)*hTarget) = DIST_SHARED_LOW_RANK;
                    *hSource += sizeof(ShellType);
                    *hTarget += sizeof(ShellType);
                }

                // Store the source data
                int colOffset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    byte** hSource = headPointers[sourceRank];
                    const int localWidth = localWidths[sourceRank];

                    *((int*)*hSource) = r;
                    *hSource += sizeof(int);
                    *((bool*)*hSource) = true; 
                    *hSource += sizeof(bool);
                    *((int*)*hSource) = targetRankOffset; 
                    *hSource += sizeof(int);

                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( *hSource, V.LockedBuffer(colOffset,j), 
                          localWidth*sizeof(Scalar) );
                        *hSource += localWidth*sizeof(Scalar);
                    }
                    colOffset += localWidth;
                }

                // Store the target data
                int rowOffset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int targetRank = targetRankOffset + i;
                    byte** hTarget = headPointers[targetRank];
                    const int localHeight = localHeights[targetRank];

                    *((int*)*hTarget) = r;
                    *hTarget += sizeof(int);
                    *((bool*)*hTarget) = false;
                    *hTarget += sizeof(bool);
                    *((int*)*hTarget) = sourceRankOffset; 
                    *hTarget += sizeof(int);

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
        }
        break;
    }
    case Quasi2d::DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const MatrixType type = D.Type();
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                if( rank == 0 )
                {
                    std::cout << "Packing DENSE for " << sourceRank 
                              << std::endl;
                }
                // Store a serial dense matrix
                byte** h = headPointers[sourceRank];
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
                if( rank == 0 )
                {
                    std::cout << "Packing SHARED_DENSE for " << sourceRank 
                              << " and " << targetRank << std::endl;
                }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), _symmetric(false), 
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(0)
{ 
    _shell.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms, unsigned level, 
  bool inSourceTeam, bool inTargetTeam )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), _symmetric(false), 
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(level),
  _inSourceTeam(inSourceTeam), _inTargetTeam(inTargetTeam)
{ 
    _shell.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const byte* packedPiece, const Subcomms& subcomms )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::DistQuasi2dHMatrix");
#endif
    Unpack( packedPiece, subcomms );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::~DistQuasi2dHMatrix()
{ }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::LocalHeight() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::LocalHeight");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

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
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

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
( const byte* packedDistHMatrix, const Subcomms& subcomms )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Unpack");
#endif
    _subcomms = &subcomms;
    _level = 0;
    _inSourceTeam = true;
    _inTargetTeam = true;

    const byte* head = packedDistHMatrix;
    UnpackRecursion( head, *this );
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
( const byte*& head, DistQuasi2dHMatrix<Scalar,Conjugated>& H )
{
    MPI_Comm comm = H._subcomms->Subcomm( 0 );
    MPI_Comm team = H._subcomms->Subcomm( H._level );
    const bool inSourceTeam = H._inSourceTeam;
    const bool inTargetTeam = H._inTargetTeam;
    if( !inSourceTeam && !inTargetTeam )
    {
        H._shell.type = EMPTY;
        return;
    }

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
    case NODE:                 delete shell.data.node; break;
    case NODE_SYMMETRIC:       delete shell.data.nodeSymmetric; break;
    case DIST_SHARED_LOW_RANK: delete shell.data.DSF; break;
    case DIST_LOW_RANK:        delete shell.data.DF; break;
    case SHARED_QUASI2D:       delete shell.data.SH; break;
    case SHARED_LOW_RANK:      delete shell.data.SF; break;
    case SHARED_DENSE:         delete shell.data.SD; break;
    case QUASI2D:              delete shell.data.H; break;
    case LOW_RANK:             delete shell.data.F; break;
    case DENSE:                delete shell.data.D; break;
    case EMPTY: break;
    }

    // Read in the information for the new shell
    shell.type = *((ShellType*)head); head += sizeof(ShellType);
    const int m = H._height;
    const int n = H._width;
    const int rank = mpi::CommRank( comm );
    switch( shell.type )
    {
    case NODE:
    { 
        shell.data.node = 
            new Node
            ( H._xSizeSource, H._xSizeTarget,
              H._ySizeSource, H._ySizeTarget, H._zSize );
        Node& node = *shell.data.node;

        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        const bool inUpperHalfOfTeam = ( teamRank >= teamSize/2 );
        // Take care of the top-left block
        for( int t=0; t<2; ++t )
        {
            for( int s=0; s<2; ++s )
            {
                bool inSplitSourceTeam = ( !inUpperHalfOfTeam && inSourceTeam );
                bool inSplitTargetTeam = ( !inUpperHalfOfTeam && inTargetTeam );
                node.children[s+4*t] = 
                    new DistQuasi2dHMatrix<Scalar,Conjugated>
                    ( *H._subcomms, H._level+1, 
                      inSplitSourceTeam, inSplitTargetTeam );
                UnpackRecursion( head, node.Child(t,s) );
            }
        }
        // Take care of the top-right block
        for( int t=0; t<2; ++t )
        {
            for( int s=2; s<4; ++s )
            {
                bool inSplitSourceTeam = ( inUpperHalfOfTeam && inSourceTeam );
                bool inSplitTargetTeam = ( !inUpperHalfOfTeam && inTargetTeam );
                node.children[s+4*t] = 
                    new DistQuasi2dHMatrix<Scalar,Conjugated>
                    ( *H._subcomms, H._level+1,
                      inSplitSourceTeam, inSplitTargetTeam );
                UnpackRecursion( head, node.Child(t,s) );
            }
        }
        // Take care of the bottom-left block
        for( int t=2; t<4; ++t )
        {
            for( int s=0; s<2; ++s )
            {
                bool inSplitSourceTeam = ( !inUpperHalfOfTeam && inSourceTeam );
                bool inSplitTargetTeam = ( inUpperHalfOfTeam && inTargetTeam );
                node.children[s+4*t] =
                    new DistQuasi2dHMatrix<Scalar,Conjugated>
                    ( *H._subcomms, H._level+1,
                      inSplitSourceTeam, inSplitTargetTeam );
                UnpackRecursion( head, node.Child(t,s) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_SHARED_LOW_RANK:
    {
        if( rank == 0 )
            std::cout << "Unpacking DIST_SHARED_LOW_RANK" << std::endl;
        shell.data.DSF = new DistSharedLowRankMatrix<Scalar,Conjugated>;
        DistSharedLowRankMatrix<Scalar,Conjugated>& DSF = *shell.data.DSF;

        DSF.height = m;
        DSF.width = n;
        DSF.comm = comm;
        DSF.team = team;

        DSF.rank            = *((int*)head);  head += sizeof(int);
        DSF.inSourceTeam    = *((bool*)head); head += sizeof(bool);
        DSF.rootOfOtherTeam = *((int*)head);  head += sizeof(int);

        DSF.DLocal.SetType( GENERAL );
        if( DSF.inSourceTeam )
        {
            const int localWidth = this->LocalWidth();
            if( rank == 0 )
                std::cout << "localWidth=" << localWidth << std::endl;
            DSF.DLocal.Resize( localWidth, DSF.rank );
            for( int j=0; j<DSF.rank; ++j )
            {
                std::memcpy
                ( DSF.DLocal.Buffer(0,j), head, localWidth*sizeof(Scalar) );
                head += localWidth*sizeof(Scalar);
            }
        }
        else
        {
            const int localHeight = this->LocalHeight();
            if( rank == 0 )
                std::cout << "localHeight=" << localHeight << std::endl;
            DSF.DLocal.Resize( localHeight, DSF.rank );
            for( int j=0; j<DSF.rank; ++j )
            {
                std::memcpy
                ( DSF.DLocal.Buffer(0,j), head, localHeight*sizeof(Scalar) );
                head += localHeight*sizeof(Scalar);
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( rank == 0 )
            std::cout << "Unpacking DIST_LOW_RANK" << std::endl;
        shell.data.DF = new DistLowRankMatrix<Scalar,Conjugated>;
        DistLowRankMatrix<Scalar,Conjugated>& DF = *shell.data.DF;

        DF.height = m;
        DF.width = n;
        DF.comm = comm;
        DF.team = team;

        DF.rank = *((int*)head); 
        head += sizeof(int);

        const int localHeight = this->LocalHeight();
        if( rank == 0 )
            std::cout << "localHeight=" << localHeight << std::endl;
        DF.ULocal.SetType( GENERAL );
        DF.ULocal.Resize( localHeight, DF.rank );
        for( int j=0; j<DF.rank; ++j )
        {
            std::memcpy
            ( DF.ULocal.Buffer(0,j), head, localHeight*sizeof(Scalar) );
            head += localHeight*sizeof(Scalar);
        }

        const int localWidth = this->LocalWidth();
        if( rank == 0 )
            std::cout << "localWidth=" << localWidth << std::endl;
        DF.VLocal.SetType( GENERAL );
        DF.VLocal.Resize( localWidth, DF.rank );
        for( int j=0; j<DF.rank; ++j )
        {
            std::memcpy
            ( DF.VLocal.Buffer(0,j), head, localWidth*sizeof(Scalar) );
            head += localWidth*sizeof(Scalar);
        }
        break;
    }
    case SHARED_QUASI2D:
    {
        if( rank == 0 )
            std::cout << "Unpacking SHARED_QUASI2D" << std::endl;
        typedef SharedQuasi2dHMatrix<Scalar,Conjugated> SharedQuasi2d;

        shell.data.SH = new SharedQuasi2d;
        SharedQuasi2d& SH = *shell.data.SH;

        std::size_t packedSize = SH.Unpack( head );
        head += packedSize;
        break;
    }
    case SHARED_LOW_RANK:
    {
        if( rank == 0 )
            std::cout << "Unpacking SHARED_LOW_RANK" << std::endl;
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
        if( rank == 0 )
            std::cout << "Unpacking SHARED_DENSE" << std::endl;
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
            SD.D.Resize( m, n );
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
        if( rank == 0 )
            std::cout << "Unpacking QUASI2D" << std::endl;
        shell.data.H = new Quasi2dHMatrix<Scalar,Conjugated>;
        Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;

        const std::size_t packedSize = H.Unpack( head );
        head += packedSize;
        break;
    }
    case LOW_RANK:
    {
        if( rank == 0 )
            std::cout << "Unpacking LOW_RANK" << std::endl;
        shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        LowRankMatrix<Scalar,Conjugated>& F = *shell.data.F;

        // Read in the rank
        const int r = *((int*)head); head += sizeof(int);

        // Read in U
        F.U.SetType( GENERAL ); F.U.Resize( m, r );
        for( int j=0; j<r; ++j )
        {
            std::memcpy( F.U.Buffer(0,j), head, m*sizeof(Scalar) );
            head += m*sizeof(Scalar);
        }

        // Read in V
        F.V.SetType( GENERAL ); F.V.Resize( n, r );
        for( int j=0; j<r; ++j )
        {
            std::memcpy( F.V.Buffer(0,j), head, n*sizeof(Scalar) );
            head += n*sizeof(Scalar);
        }
        break;
    }
    case DENSE:
    {
        if( rank == 0 )
            std::cout << "Unpacking DENSE" << std::endl;
        shell.data.D = new DenseMatrix<Scalar>;
        DenseMatrix<Scalar>& D = *shell.data.D;

        const MatrixType type = *((MatrixType*)head); 
        head += sizeof(MatrixType);
        D.SetType( type );
        D.Resize( m, n );
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
#ifndef RELEASE
        throw std::logic_error("Should not need to unpack empty submatrix");
#endif
        break;
    default:
#ifndef RELEASE
        throw std::logic_error("Invalid enum value");
#endif
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
