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
    std::vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );

    // Recurse on this shell to compute the packed sizes
    PackedSizesRecursion( packedSizes, localSizes, 0, 0, p, H );
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
    std::vector<byte*> heads = packedPieces;
    std::vector<byte**> headPointers(p); 
    for( int i=0; i<p; ++i )
        headPointers[i] = &heads[i];
    
    std::vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );
    PackRecursion( headPointers, localSizes, 0, 0, p, H );

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
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeFirstLocalRow
( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::ComputeFirstLocalRow");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalRow = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalRow, p, rank, H._xSizeTarget, H._ySizeTarget, H._zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalRow;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeFirstLocalCol
( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::ComputeFirstLocalCol");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalCol = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalCol, p, rank, H._xSizeSource, H._ySizeSource, H._zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalCol;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalSizes
( std::vector<int>& localSizes, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE    
    PushCallStack("DistQuasi2dHMatrix::ComputeLocalSizes");
    const int p = localSizes.size();
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
    if( (H._xSizeSource != H._xSizeTarget) || 
        (H._ySizeSource != H._ySizeTarget) )
        throw std::logic_error("Routine meant for square nodes");
#endif
    ComputeLocalSizesRecursion
    ( &localSizes[0], localSizes.size(), 
      H._xSizeSource, H._ySizeSource, H._zSize );
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
  const std::vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SplitQuasi2dHMatrix<Scalar,Conjugated> SplitQuasi2d;

    const std::size_t headerSize = 
        15*sizeof(int) + sizeof(bool) + sizeof(ShellType);
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
                // Store a split H-matrix
                std::pair<std::size_t,std::size_t> sizes = 
                    SplitQuasi2d::PackedSizes( H );
                packedSizes[sourceRank] += sizes.first;
                packedSizes[targetRank] += sizes.second;
            }
        }
        else if( teamSize == 2 )
        {
            // Give the upper-left 2x2 to the first halves of the teams 
            // and the lower-right 2x2 to the second halves.
            const int newTeamSize = teamSize/2;
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+newTeamSize*(s/2),
                      targetRankOffset+newTeamSize*(t/2), newTeamSize,
                      shell.data.node->Child(t,s) );
                }
            }
        }
        else // team Size >= 4
        {
            // Give each diagonal block of the 4x4 partition to a different
            // quarter of the teams
            const int newTeamSize = teamSize/4;
            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+newTeamSize*s,
                      targetRankOffset+newTeamSize*t, newTeamSize,
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
                // Store a split H-matrix
                std::pair<std::size_t,std::size_t> sizes = 
                    SplitQuasi2d::PackedSizes( H );
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
                // Store a split low-rank matrix                

                // The source and target processes store the matrix rank and
                // their factor's entries.
                packedSizes[sourceRank] += sizeof(int) + n*r*sizeof(Scalar);
                packedSizes[targetRank] += sizeof(int) + m*r*sizeof(Scalar);
            }
        }
        else
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Every process owns a piece of both U and V. Store those 
                // pieces along with the matrix rank.
                std::cerr << "WARNING: Unlikely admissible case." << std::endl;
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    const int localSize = localSizes[sourceRank];
                    packedSizes[sourceRank] += 
                        sizeof(int) + 2*localSize*r*sizeof(Scalar);
                }
            }
            else
            {
                // Each process either owns a piece of U or V. Store it along
                // with the matrix rank.

                // Write out the source information
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    packedSizes[sourceRank] += 
                        sizeof(int) + localSizes[sourceRank]*r*sizeof(Scalar);
                }

                // Write out the target information
                for( int i=0; i<teamSize; ++i )
                {
                    const int targetRank = targetRankOffset + i;
                    packedSizes[targetRank] += 
                        sizeof(int) + localSizes[targetRank]*r*sizeof(Scalar);
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
  const std::vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SplitQuasi2dHMatrix<Scalar,Conjugated> SplitQuasi2d;

    // Write the header information for every process in the source team
    for( int i=0; i<teamSize; ++i )
    {
        const int sourceRank = sourceRankOffset + i;
        byte** h = headPointers[sourceRank];

        Write( h, H._height );
        Write( h, H._width );
        Write( h, H._numLevels );
        Write( h, H._maxRank );
        Write( h, H._sourceOffset );
        Write( h, H._targetOffset );
        // Write( h, H._type );
        Write( h, H._stronglyAdmissible );
        Write( h, H._xSizeSource );
        Write( h, H._xSizeTarget );
        Write( h, H._ySizeSource );
        Write( h, H._ySizeTarget );
        Write( h, H._zSize );
        Write( h, H._xSource );
        Write( h, H._xTarget );
        Write( h, H._ySource );
        Write( h, H._yTarget );
    }

    // Write the header information for every process in the target team
    // (shamelessly copy and pasted from above...)
    if( targetRankOffset != sourceRankOffset )
    {
        for( int i=0; i<teamSize; ++i )
        {
            const int targetRank = targetRankOffset + i;
            byte** h = headPointers[targetRank];

            Write( h, H._height );
            Write( h, H._width );
            Write( h, H._numLevels );
            Write( h, H._maxRank );
            Write( h, H._sourceOffset );
            Write( h, H._targetOffset );
            // Write( h, h._type );
            Write( h, H._stronglyAdmissible );
            Write( h, H._xSizeSource );
            Write( h, H._xSizeTarget );
            Write( h, H._ySizeSource );
            Write( h, H._ySizeTarget );
            Write( h, H._zSize );
            Write( h, H._xSource );
            Write( h, H._xTarget );
            Write( h, H._ySource );
            Write( h, H._yTarget );
        }
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
                byte** h = headPointers[sourceRank];
                Write( h, QUASI2D );
                H.Pack( *h ); 
                *h += H.PackedSize();
            }
            else
            {
                // Store a split H-matrix 
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                Write( hSource, SPLIT_QUASI2D );
                Write( hTarget, SPLIT_QUASI2D );

                std::pair<std::size_t,std::size_t> sizes =
                    SplitQuasi2d::Pack
                    ( *hSource, *hTarget, sourceRank, targetRank, H );

                *hSource += sizes.first;
                *hTarget += sizes.second;
            }
        }
        else if( teamSize == 2 )
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            const typename Quasi2d::Node& node = *shell.data.node;
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                Write( hSource, NODE );
            }
            if( sourceRankOffset != targetRankOffset )
            {
                for( int i=0; i<teamSize; ++i )
                {
                    byte** hTarget = headPointers[targetRankOffset+i];
                    Write( hTarget, NODE );
                }
            }
            const int newTeamSize = teamSize/2;
            // Top-left block
            for( int t=0; t<2; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset, newTeamSize,
                      node.Child(t,s) );
                }
            }
            // Top-right block
            for( int t=0; t<2; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, targetRankOffset, 
                      newTeamSize, node.Child(t,s) );
                }
            }
            // Bottom-left block
            for( int t=2; t<4; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset+newTeamSize, 
                      newTeamSize, node.Child(t,s) );
                }
            }
            // Bottom-right block
            for( int t=2; t<4; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, 
                      targetRankOffset+newTeamSize,
                      newTeamSize, node.Child(t,s) );
                }
            }
        }
        else
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            const typename Quasi2d::Node& node = *shell.data.node;
            for( int i=0; i<teamSize; ++i )
            {
                byte** hSource = headPointers[sourceRankOffset+i];
                Write( hSource, NODE );
            }
            if( sourceRankOffset != targetRankOffset )
            {
                for( int i=0; i<teamSize; ++i )
                {
                    byte** hTarget = headPointers[targetRankOffset+i];
                    Write( hTarget, NODE );
                }
            }
            const int newTeamSize = teamSize/4;
            // Top-left block
            for( int t=0; t<2; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize, 
                      newTeamSize, node.Child(t,s) );
                }
            }
            // Top-right block
            for( int t=0; t<2; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
                }
            }
            // Bottom-left block
            for( int t=2; t<4; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize, 
                      newTeamSize, node.Child(t,s) );
                }
            }
            // Bottom-right block
            for( int t=2; t<4; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
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
                Write( h, QUASI2D );
                H.Pack( *h ); 
                *h += H.PackedSize();
            }
            else
            {
                // Store a split H-matrix 
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                Write( hSource, SPLIT_QUASI2D );
                Write( hTarget, SPLIT_QUASI2D );

                std::pair<std::size_t,std::size_t> sizes =
                    SplitQuasi2d::Pack
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
                byte** h = headPointers[sourceRank];
                Write( h, LOW_RANK );

                // Store the rank and matrix entries
                Write( h, r );
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
                // Store a split low-rank representation
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                Write( hSource, SPLIT_LOW_RANK );
                Write( hTarget, SPLIT_LOW_RANK );

                // Store the rank and entries of V on the source side
                Write( hSource, r );
                for( int j=0; j<r; ++j )
                {
                    std::memcpy
                    ( *hSource, V.LockedBuffer(0,j), n*sizeof(Scalar) );
                    *hSource += n*sizeof(Scalar);
                }

                // Store the rank and entries of U on the target side
                Write( hTarget, r );
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
                // NOTE: This should only happen when there is a weird
                //       admissibility condition that allows diagonal blocks
                //       to be low-rank.
                std::cerr << "WARNING: Unlikely admissible case." << std::endl;

                // Store a distributed low-rank representation
                int rowOffset = 0;
                int colOffset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    // Store the header information
                    const int rank = sourceRankOffset + i;
                    byte** h = headPointers[rank];
                    Write( h, DIST_LOW_RANK );
                    Write( h, r );

                    // Store our local U and V
                    const int localSize = localSizes[rank];
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( h, U.LockedBuffer(rowOffset,j), 
                          localSize*sizeof(Scalar) );
                        *h += localSize*sizeof(Scalar);
                    }
                    rowOffset += localSize;
                    for( int j=0; j<r; ++j )
                    {
                        std::memcpy
                        ( h, V.LockedBuffer(colOffset,j),
                          localSize*sizeof(Scalar) );
                        *h += localSize*sizeof(Scalar);
                    }
                    colOffset += localSize;
                }
            }
            else
            {
                // Store a distributed split low-rank representation

                // Store the source data
                int colOffset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    byte** hSource = headPointers[sourceRank];

                    Write( hSource, DIST_LOW_RANK );
                    Write( hSource, r );

                    const int localWidth = localSizes[sourceRank];
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
                    
                    Write( hTarget, DIST_LOW_RANK );
                    Write( hTarget, r );

                    const int localHeight = localSizes[targetRank];
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
                // Store a serial dense matrix
                byte** h = headPointers[sourceRank];
                Write( h, DENSE );
                Write( h, type );
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
                // Store a split dense matrix
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];

                // Store the source data
                Write( hSource, SPLIT_DENSE );
                Write( hSource, type );
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

                // There is no target data to store
                Write( hTarget, SPLIT_DENSE );
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
    if( p >= 4 )
    {
        const int subteam = rank/(p/4);
        const int subteamRank = rank-subteam*(p/4);
        const bool onRight = (subteam & 1);
        const bool onTop = (subteam/2);

        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        ComputeLocalDimensionRecursion
        ( localDim, p/4, subteamRank, 
          (onRight ? xRightSize : xLeftSize),
          (onTop ? yTopSize : yBottomSize), zSize );
    }
    else if( p == 2 )
    {
        const int subteam = rank/(p/2);
        const int subteamRank = rank-subteam*(p/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        ComputeLocalDimensionRecursion
        ( localDim, p/2, subteamRank,
          xSize, 
          (subteam ? yTopSize : yBottomSize), zSize );
    }
    else // p == 1
    {
        localDim = xSize*ySize*zSize;
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeFirstLocalIndexRecursion
( int& firstLocalIndex, int p, int rank, int xSize, int ySize, int zSize )
{
    if( p >= 4 )
    {
        const int subteam = rank/(p/4);
        const int subteamRank = rank-subteam*(p/4);
        const bool onRight = (subteam & 1);
        const bool onTop = (subteam/2);

        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        // Add on this level of offsets
        if( onRight && onTop )
            firstLocalIndex += xSize*yBottomSize + xLeftSize*yTopSize;
        else if( onTop )
            firstLocalIndex += xSize*yBottomSize;
        else if( onRight )
            firstLocalIndex += xLeftSize*yBottomSize;

        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, p/4, subteamRank,
          (onRight ? xRightSize : xLeftSize),
          (onTop ? yTopSize : yBottomSize), zSize );
    }
    else if( p == 2 )
    {
        const int subteam = rank/(p/2);
        const int subteamRank = rank-subteam*(p/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        // Add on this level of offsets
        if( subteam )
            firstLocalIndex += xSize*yBottomSize;

        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, p/2, subteamRank,
          xSize, 
          (subteam ? yTopSize : yBottomSize), zSize );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::ComputeLocalSizesRecursion
( int* localSizes, int teamSize, int xSize, int ySize, int zSize ) 
{
    if( teamSize >=4 )
    {
        const int newTeamSize = teamSize/4;
        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        // Bottom-left piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( localSizes, newTeamSize, 
          xLeftSize, yBottomSize, zSize );
        // Bottom-right piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[newTeamSize], newTeamSize, 
          xRightSize, yBottomSize, zSize );
        // Top-left piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[2*newTeamSize], newTeamSize,
          xLeftSize, yTopSize, zSize );
        // Top-right piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[3*newTeamSize], newTeamSize,
          xRightSize, yTopSize, zSize );
    }
    else if( teamSize == 2 )
    {
        // Bottom piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( localSizes, 1, xSize, ySize/2, zSize );
        // Top piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[1], 1, xSize, ySize-ySize/2, zSize );
    }
    else
    {
        localSizes[0] = xSize*ySize*zSize;
    }
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), _rootOfOtherTeam(0),
  _localSourceOffset(0), _localTargetOffset(0)
{ 
    _shell.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms, unsigned level, 
  bool inSourceTeam, bool inTargetTeam, 
  int localSourceOffset, int localTargetOffset )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(level),
  _inSourceTeam(inSourceTeam), _inTargetTeam(inTargetTeam),
  _localSourceOffset(localSourceOffset), _localTargetOffset(localTargetOffset)
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
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FirstLocalRow() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FirstLocalRow");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

    int firstLocalRow = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalRow, teamSize, teamRank, _xSizeTarget, _ySizeTarget, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalRow;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FirstLocalCol() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FirstLocalCol");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

    int firstLocalCol = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalCol, teamSize, teamRank, _xSizeSource, _ySizeSource, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalCol;
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
    _rootOfOtherTeam = 0;
    _localSourceOffset = 0;
    _localTargetOffset = 0;

    const byte* head = packedDistHMatrix;
    UnpackRecursion( head, *this, 0, 0 );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedDistHMatrix);
}

// HERE: Rethink whether or not to create a DistVec class. The current sketch
//       simply uses serial vectors in order to simplify the interface with 
//       PETSc.
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVector");
#endif
    // y := beta y
    hmatrix_tools::Scale( beta, yLocal );

    // Perform the source team local computations
    MapVectorPrecompute( alpha, xLocal, yLocal );

    // Sum within source teams
    MapVectorSourceTeamSummations();
    //MapVectorNaiveSourceTeamSummations();

    // Pass data from source to target teams
    //MapVectorPassData();
    MapVectorNaivePassData();

    // Locally broadcast data from roots
    MapVectorTargetTeamBroadcasts();
    //MapVectorNaiveTargetTeamBroadcasts();

    // Add the submatrices' contributions onto yLocal
    MapVectorPostcompute( yLocal );
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
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPrecompute( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inSourceTeam )
        {
            // Form z := alpha VLocal^[T/H] xLocal
            DF.z.Resize( DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemv
            ( option, DF.VLocal.Height(), DF.rank, 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, DF.z.Buffer(),            1 );
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        if( _inSourceTeam )
        {
            Vector<Scalar> xLocalPiece;
            xLocalPiece.LockedView( xLocal, _localSourceOffset, SH._width );
            SH.MapVectorPrecompute( alpha, xLocalPiece );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;

        if( _inSourceTeam )
        {
            Vector<Scalar> xLocalPiece;
            xLocalPiece.LockedView( xLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeVector
                ( alpha, SF.D, xLocalPiece, SF.z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeVector
                ( alpha, SF.D, xLocalPiece, SF.z );
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( _inSourceTeam )
        {
            Vector<Scalar> xLocalPiece;
            xLocalPiece.LockedView( xLocal, _localSourceOffset, this->_width );
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocalPiece, SD.z );
        }
        break;
    }
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        Vector<Scalar> xLocalPiece, yLocalPiece;
        xLocalPiece.LockedView( xLocal, _localSourceOffset, H.Width() );
        yLocalPiece.View( yLocal, _localTargetOffset, H.Height() );
        H.MapVector( alpha, xLocalPiece, (Scalar)1, yLocalPiece );
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
        Vector<Scalar> xLocalPiece, yLocalPiece;
        xLocalPiece.LockedView( xLocal, _localSourceOffset, F.Width() );
        yLocalPiece.View( yLocal, _localTargetOffset, F.Height() );
        hmatrix_tools::MatrixVector
        ( alpha, F, xLocalPiece, (Scalar)1, yLocalPiece );
        break;
    }
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        Vector<Scalar> xLocalPiece, yLocalPiece;
        xLocalPiece.LockedView( xLocal, _localSourceOffset, D.Width() );
        yLocalPiece.View( yLocal, _localTargetOffset, D.Height() );
        hmatrix_tools::MatrixVector
        ( alpha, D, xLocalPiece, (Scalar)1, yLocalPiece );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSourceTeamSummations() 
const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSourceTeamSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MapVectorSourceTeamSummationsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapVectorSourceTeamSummationsPack( buffer, offsets );

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
    MapVectorSourceTeamSummationsUnpack( buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSourceTeamSummationsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSourceTeamSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSourceTeamSummationsCount( sizes );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inSourceTeam )
            sizes[_level-1] += DF.rank;
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSourceTeamSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSourceTeamSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                node.Child(t,s).MapVectorSourceTeamSummationsPack
                ( buffer, offsets );
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
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inSourceTeam )
        {
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.z.LockedBuffer(), 
              DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
        }
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSourceTeamSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSourceTeamSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                node.Child(t,s).MapVectorSourceTeamSummationsUnpack
                ( buffer, offsets );
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
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inSourceTeam )
        {
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( DF.z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank;
            }
        }
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaiveSourceTeamSummations()
const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaiveSourceTeamSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaiveSourceTeamSummations();
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inSourceTeam )
        {
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, DF.z.Buffer(), 
                  DF.rank, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( DF.z.LockedBuffer(), 0, DF.rank, 0, MPI_SUM, team );
            }
        }
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPassData() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPassData");
#endif
    // TODO: Implement AllToAll redistribution
    throw std::logic_error("Non-naive MapVectorPassData not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaivePassData()
const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).MapVectorNaivePassData();
                node.Child(0,1).MapVectorNaivePassData();
                node.Child(1,0).MapVectorNaivePassData();
                node.Child(1,1).MapVectorNaivePassData();
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).MapVectorNaivePassData();
                node.Child(2,3).MapVectorNaivePassData();
                node.Child(3,2).MapVectorNaivePassData();
                node.Child(3,3).MapVectorNaivePassData();
            }
            // Top-right quadrant
            node.Child(0,2).MapVectorNaivePassData();
            node.Child(0,3).MapVectorNaivePassData();
            node.Child(1,2).MapVectorNaivePassData();
            node.Child(1,3).MapVectorNaivePassData();

            // Bottom-left quadrant
            node.Child(2,0).MapVectorNaivePassData();
            node.Child(2,1).MapVectorNaivePassData();
            node.Child(3,0).MapVectorNaivePassData();
            node.Child(3,1).MapVectorNaivePassData();
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapVectorNaivePassData();
                // Interact with subteam 1
                node.Child(0,1).MapVectorNaivePassData();
                node.Child(1,0).MapVectorNaivePassData();
                // Interact with subteam 2
                node.Child(0,2).MapVectorNaivePassData();
                node.Child(2,0).MapVectorNaivePassData();
                // Interact with subteam 3
                node.Child(0,3).MapVectorNaivePassData();
                node.Child(3,0).MapVectorNaivePassData();
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapVectorNaivePassData();
                // Interact with subteam 0
                node.Child(0,1).MapVectorNaivePassData();
                node.Child(1,0).MapVectorNaivePassData();
                // Interact with subteam 3
                node.Child(1,3).MapVectorNaivePassData(); 
                node.Child(3,1).MapVectorNaivePassData();
                // Interact with subteam 2
                node.Child(1,2).MapVectorNaivePassData();
                node.Child(2,1).MapVectorNaivePassData();
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapVectorNaivePassData();
                // Interact with subteam 3
                node.Child(2,3).MapVectorNaivePassData();
                node.Child(3,2).MapVectorNaivePassData();
                // Interact with subteam 0
                node.Child(0,2).MapVectorNaivePassData();
                node.Child(2,0).MapVectorNaivePassData();
                // Interact with subteam 1
                node.Child(1,2).MapVectorNaivePassData();
                node.Child(2,1).MapVectorNaivePassData();
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapVectorNaivePassData();
                // Interact with subteam 2
                node.Child(2,3).MapVectorNaivePassData();
                node.Child(3,2).MapVectorNaivePassData();
                // Interact with subteam 1
                node.Child(1,3).MapVectorNaivePassData();
                node.Child(3,1).MapVectorNaivePassData();
                // Interact with subteam 0
                node.Child(0,3).MapVectorNaivePassData();
                node.Child(3,0).MapVectorNaivePassData();
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
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
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
                ( DF.z.LockedBuffer(), DF.rank, _rootOfOtherTeam, 0, comm );
            }
            else
            {
                DF.z.Resize( DF.rank );
                mpi::Recv
                ( DF.z.Buffer(), DF.rank, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        SH.MapVectorNaivePassData();
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( SF.z.LockedBuffer(), SF.rank, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, _rootOfOtherTeam, 0, comm );
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
            ( SD.z.LockedBuffer(), this->_height, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SD.z.Resize( this->_height );
            mpi::Recv
            ( SD.z.Buffer(), this->_height, _rootOfOtherTeam, 0, comm );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorTargetTeamBroadcasts() 
const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorTargetTeamBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MapVectorTargetTeamBroadcastsCount( sizes );

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
    MapVectorTargetTeamBroadcastsPack( buffer, offsets );

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
    MapVectorTargetTeamBroadcastsUnpack( buffer, offsets );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorTargetTeamBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorTargetTeamBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorTargetTeamBroadcastsCount( sizes );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inTargetTeam )
            sizes[_level-1] += DF.rank;
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorTargetTeamBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorTargetTeamBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                node.Child(t,s).MapVectorTargetTeamBroadcastsPack
                ( buffer, offsets );
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
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inTargetTeam )
        {
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], DF.z.LockedBuffer(), 
                  DF.rank*sizeof(Scalar) );
                offsets[_level-1] += DF.rank;
            }
        }
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorTargetTeamBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorTargetTeamBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                node.Child(t,s).MapVectorTargetTeamBroadcastsUnpack
                ( buffer, offsets );
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
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inTargetTeam )
        {
            DF.z.Resize( DF.rank );
            std::memcpy
            ( DF.z.Buffer(), &buffer[offsets[_level-1]], 
              DF.rank*sizeof(Scalar) );
            offsets[_level-1] += DF.rank;
        }
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaiveTargetTeamBroadcasts()
const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaiveTargetTeamBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaiveTargetTeamBroadcasts();
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inTargetTeam )
        {
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.z.Resize( DF.rank );
            mpi::Broadcast( DF.z.Buffer(), DF.rank, 0, team );
        }
        break;
    }
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPostcompute
( Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPostcompute( yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRankMatrix& DF = *shell.data.DF;
        if( _inTargetTeam )
        {
            // yLocal += ULocal z
            blas::Gemv
            ( 'N', DF.VLocal.Height(), DF.rank,
              (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                         DF.z.LockedBuffer(),      1,
              (Scalar)1, yLocal.Buffer(),          1 );
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        if( _inTargetTeam )
        {
            Vector<Scalar> yLocalPiece;
            yLocalPiece.View( yLocal, _localTargetOffset, this->_height );
            SH.MapVectorPostcompute( yLocalPiece );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        if( _inTargetTeam )
        {
            Vector<Scalar> yLocalPiece;
            yLocalPiece.View( yLocal, _localTargetOffset, SF.D.Height() );
            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocalPiece );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( _inTargetTeam )
        {
            const int localHeight = this->_height;
            const Scalar* zBuffer = SD.z.LockedBuffer();
            Scalar* yLocalBuffer = yLocal.Buffer(_localTargetOffset);
            for( int i=0; i<localHeight; ++i )
                yLocalBuffer[i] += zBuffer[i];
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

// NOTE: Due to alowing for arbitrary power of 2 numbers of processes rather 
//       than just powers of 4, the last branch might split a 4x4 partition
//       into 2x2 quadrants and result in submatrices only acting-on/updating
//       portions of the local vectors. The parameters 'localSourceOffset'
//       and 'localTargetOffset' were introduced to keep track of these offsets.
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::UnpackRecursion
( const byte*& head, DistQuasi2dHMatrix<Scalar,Conjugated>& H,
  int sourceRankOffset, int targetRankOffset )
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
    H._rootOfOtherTeam = ( inSourceTeam ? targetRankOffset : sourceRankOffset );

    // Read in the header information
    H._height             = Read<int>( head );
    H._width              = Read<int>( head );
    H._numLevels          = Read<int>( head );
    H._maxRank            = Read<int>( head );
    H._sourceOffset       = Read<int>( head );
    H._targetOffset       = Read<int>( head );
    //H._type             = Read<MatrixType>( head );
    H._stronglyAdmissible = Read<bool>( head );
    H._xSizeSource        = Read<int>( head );
    H._xSizeTarget        = Read<int>( head );
    H._ySizeSource        = Read<int>( head );
    H._ySizeTarget        = Read<int>( head );
    H._zSize              = Read<int>( head );
    H._xSource            = Read<int>( head );
    H._xTarget            = Read<int>( head );
    H._ySource            = Read<int>( head );
    H._yTarget            = Read<int>( head );

    // Delete the old shell
    Shell& shell = H._shell;
    switch( shell.type ) 
    {
    case NODE:           delete shell.data.node; break;
    case NODE_SYMMETRIC: delete shell.data.nodeSymmetric; break;
    case DIST_LOW_RANK:  delete shell.data.DF; break;
    case SPLIT_QUASI2D:  delete shell.data.SH; break;
    case SPLIT_LOW_RANK: delete shell.data.SF; break;
    case SPLIT_DENSE:    delete shell.data.SD; break;
    case QUASI2D:        delete shell.data.H; break;
    case LOW_RANK:       delete shell.data.F; break;
    case DENSE:          delete shell.data.D; break;
    case EMPTY: break;
    }

    // Read in the information for the new shell
    shell.type = Read<ShellType>( head );
    const int m = H._height;
    const int n = H._width;
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
        if( teamSize >= 4 )
        {
            const int subteam = teamRank/(teamSize/4);
            // Top-left block
            for( int t=0; t<2; ++t )
            {
                const int targetRoot = targetRankOffset + t*(teamSize/4);
                for( int s=0; s<2; ++s )
                {
                    const int sourceRoot = sourceRankOffset + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inSourceTeam && (s==subteam),
                          inTargetTeam && (t==subteam) );
                    UnpackRecursion
                    ( head, node.Child(t,s), sourceRoot, targetRoot );
                }
            }
            // Top-right block
            for( int t=0; t<2; ++t )
            {
                const int targetRoot = targetRankOffset + t*(teamSize/4);
                for( int s=2; s<4; ++s )
                {
                    const int sourceRoot = sourceRankOffset + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inSourceTeam && (s==subteam),
                          inTargetTeam && (t==subteam) );
                    UnpackRecursion
                    ( head, node.Child(t,s), sourceRoot, targetRoot );
                }
            }
            // Bottom-left block
            for( int t=2; t<4; ++t )
            {
                const int targetRoot = targetRankOffset + t*(teamSize/4);
                for( int s=0; s<2; ++s )
                {
                    const int sourceRoot = sourceRankOffset + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inSourceTeam && (s==subteam),
                          inTargetTeam && (t==subteam) );
                    UnpackRecursion
                    ( head, node.Child(t,s), sourceRoot, targetRoot );
                }
            }
            // Bottom-right block
            for( int t=2; t<4; ++t )
            {
                const int targetRoot = targetRankOffset + t*(teamSize/4);
                for( int s=2; s<4; ++s )
                {
                    const int sourceRoot = sourceRankOffset + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inSourceTeam && (s==subteam),
                          inTargetTeam && (t==subteam) );
                    UnpackRecursion
                    ( head, node.Child(t,s), sourceRoot, targetRoot );
                }
            }
        }
        else  // teamSize == 2
        {
#ifndef RELEASE
            if( teamSize != 2 )
                throw std::logic_error("Team size was not 2 as expected");
#endif
            const bool inUpperTeam = ( teamRank >= teamSize/2 );
            const bool inLeftSourceTeam = ( !inUpperTeam && inSourceTeam );
            const bool inRightSourceTeam = ( inUpperTeam && inSourceTeam );
            const bool inTopTargetTeam = ( !inUpperTeam && inTargetTeam );
            const bool inBottomTargetTeam = ( inUpperTeam && inTargetTeam );

            // Top-left block
            for( int t=0; t<2; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inLeftSourceTeam, inTopTargetTeam,
                          node.sourceSizes[0]*s, node.targetSizes[0]*t );
                    UnpackRecursion
                    ( head, node.Child(t,s), 
                      sourceRankOffset, targetRankOffset );
                }
            }
            // Top-right block
            for( int t=0; t<2; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inRightSourceTeam, inTopTargetTeam,
                          node.sourceSizes[2]*(s-2), node.targetSizes[0]*t );
                    UnpackRecursion
                    ( head, node.Child(t,s),
                      sourceRankOffset+1, targetRankOffset );
                }
            }
            // Bottom-left block
            for( int t=2; t<4; ++t )
            {
                for( int s=0; s<2; ++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inLeftSourceTeam, inBottomTargetTeam,
                          node.sourceSizes[0]*s, node.targetSizes[2]*(t-2) );
                    UnpackRecursion
                    ( head, node.Child(t,s),
                      sourceRankOffset, targetRankOffset+1 );
                }
            }
            // Bottom-right block
            for( int t=2; t<4; ++t )
            {
                for( int s=2; s<4; ++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMatrix<Scalar,Conjugated>
                        ( *H._subcomms, H._level+1,
                          inRightSourceTeam, inBottomTargetTeam,
                          node.sourceSizes[2]*(s-2), 
                          node.targetSizes[2]*(t-2) );
                    UnpackRecursion
                    ( head, node.Child(t,s),
                      sourceRankOffset+1, targetRankOffset+1 );
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
        shell.data.DF = new DistLowRankMatrix;
        DistLowRankMatrix& DF = *shell.data.DF;

        DF.rank = Read<int>( head );
        if( inSourceTeam )
        {
            // Read in U
            const int localWidth = this->LocalWidth();
            DF.ULocal.SetType( GENERAL );
            DF.ULocal.Resize( localWidth, DF.rank );
            for( int j=0; j<DF.rank; ++j )
            {
                std::memcpy
                ( DF.ULocal.Buffer(0,j), head, localWidth*sizeof(Scalar) );
                head += localWidth*sizeof(Scalar);
            }
        }
        if( inTargetTeam )
        {
            // Read in V
            const int localHeight = this->LocalHeight();
            DF.VLocal.SetType( GENERAL );
            DF.VLocal.Resize( localHeight, DF.rank );
            for( int j=0; j<DF.rank; ++j )
            {
                std::memcpy
                ( DF.VLocal.Buffer(0,j), head, localHeight*sizeof(Scalar) );
                head += localHeight*sizeof(Scalar);
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        typedef SplitQuasi2dHMatrix<Scalar,Conjugated> SplitQuasi2d;

        shell.data.SH = new SplitQuasi2d( comm );
        SplitQuasi2d& SH = *shell.data.SH;

        std::size_t packedSize = SH.Unpack( head, comm );
        head += packedSize;
        break;
    }
    case SPLIT_LOW_RANK:
    {
        shell.data.SF = new SplitLowRankMatrix;
        SplitLowRankMatrix& SF = *shell.data.SF;

        SF.rank = Read<int>( head );

        SF.D.SetType( GENERAL );
        if( inSourceTeam )
        {
            // Read in V
            SF.D.Resize( n, SF.rank );
            for( int j=0; j<SF.rank; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, n*sizeof(Scalar) ); 
                head += n*sizeof(Scalar);
            }
        }
        else
        {
            // Read in U
            SF.D.Resize( m, SF.rank );
            for( int j=0; j<SF.rank; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        shell.data.SD = new SplitDenseMatrix;
        SplitDenseMatrix& SD = *shell.data.SD;

        if( inSourceTeam )
        {
            const MatrixType type = Read<MatrixType>( head );

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
        shell.data.H = new Quasi2dHMatrix<Scalar,Conjugated>;
        Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;

        const std::size_t packedSize = H.Unpack( head );
        head += packedSize;
        break;
    }
    case LOW_RANK:
    {
        shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        LowRankMatrix<Scalar,Conjugated>& F = *shell.data.F;

        // Read in the rank
        const int r = Read<int>( head );

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
        shell.data.D = new DenseMatrix<Scalar>;
        DenseMatrix<Scalar>& D = *shell.data.D;

        const MatrixType type = Read<MatrixType>( head );

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
