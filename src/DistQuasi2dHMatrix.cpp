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
( std::vector<byte*>& packedSubs, 
  const Quasi2dHMatrix<Scalar,Conjugated>& H, const Subcomms& subcomms )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Pack");
#endif
    MPI_Comm comm = subcomms.Subcomm(0);
    const int p = mpi::CommSize( comm );
    std::vector<byte*> heads = packedSubs;
    std::vector<byte**> headPointers(p); 
    for( int i=0; i<p; ++i )
        headPointers[i] = &heads[i];
    
    std::vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );
    PackRecursion( headPointers, localSizes, 0, 0, p, H );

    std::size_t totalSize = 0;
    for( int i=0; i<p; ++i )
        totalSize += (*headPointers[i]-packedSubs[i]);
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
                      shell.data.N->Child(t,s) );
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
                      shell.data.N->Child(t,s) );
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
            const typename Quasi2d::Node& node = *shell.data.N;
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
            const typename Quasi2d::Node& node = *shell.data.N;
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
( const byte* packedSub, const Subcomms& subcomms )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::DistQuasi2dHMatrix");
#endif
    Unpack( packedSub, subcomms );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrix");
#endif
    YLocal.Resize( LocalHeight(), XLocal.Width() );
    MapMatrix( alpha, XLocal, (Scalar)0, YLocal );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrix");
#endif
    YLocal.Resize( LocalWidth(), XLocal.Width() );
    TransposeMapMatrix( alpha, XLocal, (Scalar)0, YLocal );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrix");
#endif
    YLocal.Resize( LocalWidth(), XLocal.Width() );
    HermitianTransposeMapMatrix( alpha, XLocal, (Scalar)0, YLocal );
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
    // y := beta y
    hmatrix_tools::Scale( beta, yLocal );

    MapVectorPrecompute( alpha, xLocal, yLocal );

    MapVectorSummations( alpha, xLocal, yLocal );
    //MapVectorNaiveSummations( alpha, xLocal, yLocal );

    //MapVectorPassData( alpha, xLocal, yLocal );
    MapVectorNaivePassData( alpha, xLocal, yLocal );

    MapVectorBroadcasts( alpha, xLocal, yLocal );
    //MapVectorNaiveBroadcasts( alpha, xLocal, yLocal );

    MapVectorPostcompute( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
  Scalar beta,        DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVector");
#endif
    // Y := beta Y 
    hmatrix_tools::Scale( beta, YLocal );

    MapMatrixPrecompute( alpha, XLocal, YLocal );

    MapMatrixSummations( alpha, XLocal, YLocal);
    //MapMatrixNaiveSummations( alpha, XLocal, YLocal );

    //MapMatrixPassData( alpha, XLocal, YLocal );
    MapMatrixNaivePassData( alpha, XLocal, YLocal );

    MapMatrixBroadcasts( alpha, XLocal, YLocal );
    //MapMatrixNaiveBroadcasts( alpha, XLocal, YLocal );

    MapMatrixPostcompute( alpha, XLocal, YLocal );
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
    // y := beta y
    hmatrix_tools::Scale( beta, yLocal );

    TransposeMapVectorPrecompute( alpha, xLocal, yLocal );

    TransposeMapVectorSummations( alpha, xLocal, yLocal );
    //TransposeMapVectorNaiveSummations( alpha, xLocal, yLocal );

    //TransposeMapVectorPassData( alpha, xLocal, yLocal );
    TransposeMapVectorNaivePassData( alpha, xLocal, yLocal );

    TransposeMapVectorBroadcasts( alpha, xLocal, yLocal );
    //TransposeMapVectorNaiveBroadcasts( alpha, xLocal, yLocal );

    TransposeMapVectorPostcompute( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
  Scalar beta,        DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrix");
#endif
    // Y := beta Y 
    hmatrix_tools::Scale( beta, YLocal );

    TransposeMapMatrixPrecompute( alpha, XLocal, YLocal );

    TransposeMapMatrixSummations( alpha, XLocal, YLocal );
    //TransposeMapMatrixNaiveSummations( alpha, XLocal, YLocal );

    //TransposeMapMatrixPassData( alpha, XLocal, YLocal );
    TransposeMapMatrixNaivePassData( alpha, XLocal, YLocal );

    TransposeMapMatrixBroadcasts( alpha, XLocal, YLocal );
    //TransposeMapMatrixNaiveBroadcasts( alpha, XLocal, YLocal );

    TransposeMapMatrixPostcompute( alpha, XLocal, YLocal );
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
    // y := beta y
    hmatrix_tools::Scale( beta, yLocal );

    HermitianTransposeMapVectorPrecompute( alpha, xLocal, yLocal );

    HermitianTransposeMapVectorSummations( alpha, xLocal, yLocal );
    //HermitianTransposeMapVectorNaiveSummations( alpha, xLocal, yLocal );

    //HermitianTransposeMapVectorPassData( alpha, xLocal, yLocal );
    HermitianTransposeMapVectorNaivePassData( alpha, xLocal, yLocal );

    HermitianTransposeMapVectorBroadcasts( alpha, xLocal, yLocal);
    //HermitianTransposeMapVectorNaiveBroadcasts( alpha, xLocal, yLocal );

    HermitianTransposeMapVectorPostcompute( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
  Scalar beta,        DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVector");
#endif
    // Y := beta Y 
    hmatrix_tools::Scale( beta, YLocal );

    HermitianTransposeMapMatrixPrecompute( alpha, XLocal, YLocal );

    HermitianTransposeMapMatrixSummations( alpha, XLocal, YLocal );
    //HermitianTransposeMapMatrixNaiveSummations( alpha, XLocal, YLocal );

    //HermitianTransposeMapMatrixPassData( alpha, XLocal, YLocal );
    HermitianTransposeMapMatrixNaivePassData( alpha, XLocal, YLocal );

    HermitianTransposeMapMatrixBroadcasts( alpha, XLocal, YLocal );
    //HermitianTransposeMapMatrixNaiveBroadcasts( alpha, XLocal, YLocal );

    HermitianTransposeMapMatrixPostcompute( alpha, XLocal, YLocal );
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
        const Node& node = *shell.data.N;
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
        if( _inSourceTeam )
        {
            // Form z := alpha VLocal^[T/H] xLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.z.Resize( DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemv
            ( option, DF.VLocal.Height(), DF.rank, 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, DF.z.Buffer(),            1 );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, SH._width );
            yLocalSub.View( yLocal, _localTargetOffset, SH._height );
            SH.MapVectorPrecompute( alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, SF.D.Height() );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeVector
                ( alpha, SF.D, xLocalSub, SF.z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeVector
                ( alpha, SF.D, xLocalSub, SF.z );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, this->_width );
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocalSub, SD.z );
        }
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, H.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, H.Height() );
        H.MapVector( alpha, xLocalSub, (Scalar)1, yLocalSub );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, F.Width() );
        yLocalSub.View( yLocal, _localTargetOffset, F.Height() );
        hmatrix_tools::MatrixVector
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case DENSE:
    {
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
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPrecompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixPrecompute( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // Form Z := alpha VLocal^[T/H] XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            const char option = ( Conjugated ? 'C' : 'T' );
            blas::Gemm
            ( option, 'N', DF.rank, width, DF.VLocal.Height(), 
              alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, DF.Z.Buffer(),            DF.Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SH._width, width );
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SH._height, width );
            SH.MapMatrixPrecompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, SF.D.Height(), width );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( alpha, SF.D, XLocalSub, SF.Z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, SF.D, XLocalSub, SF.Z );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, this->_width, width );
            hmatrix_tools::MatrixMatrix( alpha, SD.D, XLocalSub, SD.Z );
        }
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, H.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, H.Height(), width );
        H.MapMatrix( alpha, XLocalSub, (Scalar)1, YLocalSub );
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
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, F.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, F.Height(), width );
        hmatrix_tools::MatrixMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, D.Width(), width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, D.Height(), width );
        hmatrix_tools::MatrixMatrix
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPrecompute
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorPrecompute
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form z := alpha ULocal^T xLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.z.Resize( DF.rank );
            blas::Gemv
            ( 'T', DF.ULocal.Height(), DF.rank, 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, DF.z.Buffer(),            1 );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SH._height );
            yLocalSub.View( yLocal, _localSourceOffset, SH._width );
            SH.TransposeMapVectorPrecompute( alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SF.D.Width() );
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SF.D, xLocalSub, SF.z );
        }
        break;
    case SPLIT_DENSE:
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, H.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, H.Width() );
        H.TransposeMapVector( alpha, xLocalSub, (Scalar)1, yLocalSub );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, F.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, F.Width() );
        hmatrix_tools::MatrixTransposeVector
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case DENSE:
    {
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
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPrecompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixPrecompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^T XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            blas::Gemm
            ( 'T', 'N', DF.rank, width, DF.ULocal.Height(),
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, DF.Z.Buffer(),            DF.Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SH._height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SH._width, width );
            SH.TransposeMapMatrixPrecompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Width(), width );
            hmatrix_tools::MatrixTransposeMatrix
            ( alpha, SF.D, XLocalSub, SF.Z );
        }
        break;
    case SPLIT_DENSE:
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, H.Height(), width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, H.Width(), width );
        H.TransposeMapMatrix( alpha, XLocalSub, (Scalar)1, YLocalSub );
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
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorPrecompute
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapVectorPrecompute
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form z := alpha ULocal^H xLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.z.Resize( DF.rank );
            blas::Gemv
            ( 'C', DF.ULocal.Height(), DF.rank, 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         xLocal.LockedBuffer(),    1,
              (Scalar)0, DF.z.Buffer(),            1 );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SH._height );
            yLocalSub.View( yLocal, _localSourceOffset, SH._width );
            SH.HermitianTransposeMapVectorPrecompute
            ( alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, SF.D.Width() );
            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SF.D, xLocalSub, SF.z );
        }
        break;
    case SPLIT_DENSE:
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, H.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, H.Width() );
        H.HermitianTransposeMapVector
        ( alpha, xLocalSub, (Scalar)1, yLocalSub );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, F.Height() );
        yLocalSub.View( yLocal, _localSourceOffset, F.Width() );
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, F, xLocalSub, (Scalar)1, yLocalSub );
        break;
    }
    case DENSE:
    {
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
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixPrecompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPrecompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapMatrixPrecompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // Form Z := alpha ULocal^H XLocal
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            blas::Gemm
            ( 'C', 'N', DF.rank, width, DF.ULocal.Height(), 
              alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                         XLocal.LockedBuffer(),    XLocal.LDim(),
              (Scalar)0, DF.Z.Buffer(),            DF.Z.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SH._height, width );
            YLocalSub.View( YLocal, _localSourceOffset, 0, SH._width, width );
            SH.HermitianTransposeMapMatrixPrecompute
            ( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, SF.D.Width(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SF.D, XLocalSub, SF.Z );
        }
        break;
    case SPLIT_DENSE:
        break;
    case QUASI2D:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Quasi2dHMatrix<Scalar,Conjugated>& H = *shell.data.H;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, H.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, H.Width(), width );
        H.HermitianTransposeMapMatrix
        ( alpha, XLocalSub, (Scalar)1, YLocalSub );
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
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, F.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, F.Width(), width );
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, F, XLocalSub, (Scalar)1, YLocalSub );
        break;
    }
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const DenseMatrix<Scalar>& D = *shell.data.D;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, D.Height(), width );
        YLocalSub.View( YLocal, _localSourceOffset, 0, D.Width(), width );
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, D, XLocalSub, (Scalar)1, YLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSummations
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
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
    MapVectorSummationsCount( sizes, alpha, xLocal, yLocal );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapVectorSummationsPack( buffer, offsets, alpha, xLocal, yLocal );

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
    MapVectorSummationsUnpack( buffer, offsets, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MapMatrixSummationsCount( sizes, alpha, XLocal, YLocal );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    MapMatrixSummationsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    MapMatrixSummationsUnpack( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorSummations
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
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
    TransposeMapVectorSummationsCount( sizes, alpha, xLocal, yLocal );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    TransposeMapVectorSummationsPack( buffer, offsets, alpha, xLocal, yLocal );

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
    TransposeMapVectorSummationsUnpack
    ( buffer, offsets, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummations");
#endif
    // Compute the message sizes for each reduce 
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numReduces = std::max(0,numLevels-2);
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    TransposeMapMatrixSummationsCount( sizes, alpha, XLocal, YLocal );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=offsets[i],++i )
        offsets[i] = offset;
    std::memset( &offsets[0], 0, numReduces*sizeof(int) );
    TransposeMapMatrixSummationsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    TransposeMapMatrixSummationsUnpack
    ( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorSummations
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorSummations");
#endif
    // This unconjugated version is identical
    TransposeMapVectorSummations( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixSummations");
#endif
    // This unconjugated version is identical
    TransposeMapMatrixSummations( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorSummationsCount
( std::vector<int>& sizes, 
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSummationsCount
                ( sizes, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsCount
( std::vector<int>& sizes, 
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
( std::vector<int>& sizes,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorSummationsCount
                ( sizes, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummationsCount
( std::vector<int>& sizes,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSummationsPack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.z.LockedBuffer(), 
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorSummationsPack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.z.LockedBuffer(), 
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummationsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            std::memcpy
            ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorSummationsUnpack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixSummationsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorSummationsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorSummationsUnpack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixSummationsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixSummationsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixSummationsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaiveSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaiveSummations
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaiveSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaiveSummations");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixNaiveSummations
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, DF.Z.Buffer(), 
                  DF.rank*width, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( DF.Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
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
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorNaiveSummations");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorNaiveSummations
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaiveSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaiveSummations");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixNaiveSummations
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, DF.Z.Buffer(), 
                  DF.rank*width, 0, MPI_SUM, team );
            }
            else
            {
                mpi::Reduce
                ( DF.Z.LockedBuffer(), 0, DF.rank*width, 0, MPI_SUM, team );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorNaiveSummations
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapVectorNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorNaiveSummations( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaiveSummations
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaiveSummations");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaiveSummations( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPassData
( Scalar alpha, const Vector<Scalar>& xLocal, 
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPassData");
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
( Scalar alpha, const Vector<Scalar>& xLocal,
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPassData");
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorPassData");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorPassData( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrixPassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPassData");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixPassData( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaivePassData
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(0,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(1,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(1,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(2,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
            }
            // Top-right quadrant
            node.Child(0,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
            node.Child(0,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
            node.Child(1,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
            node.Child(1,3).MapVectorNaivePassData( alpha, xLocal, yLocal );

            // Bottom-left quadrant
            node.Child(2,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
            node.Child(2,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
            node.Child(3,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
            node.Child(3,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(0,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(1,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(0,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(2,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(0,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(1,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(1,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(1,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(2,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(2,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(2,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(2,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(2,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,2).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,1).MapVectorNaivePassData( alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,3).MapVectorNaivePassData( alpha, xLocal, yLocal );
                node.Child(3,0).MapVectorNaivePassData( alpha, xLocal, yLocal );
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localSourceOffset, SH._width );
        yLocalSub.View( yLocal, _localTargetOffset, SH._height );
        SH.MapVectorNaivePassData( alpha, xLocalSub, yLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaivePassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(0,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            node.Child(0,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(0,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(1,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(1,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );

            // Bottom-left quadrant
            node.Child(2,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(2,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(3,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
            node.Child(3,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(1,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(2,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,2).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,1).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).MapMatrixNaivePassData( alpha, XLocal, YLocal );
                node.Child(3,0).MapMatrixNaivePassData( alpha, XLocal, YLocal );
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
                ( DF.Z.LockedBuffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
            else
            {
                DF.Z.Resize( DF.rank, width, DF.rank );
                mpi::Recv
                ( DF.Z.Buffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView( XLocal, _localSourceOffset, 0, SH._width, width );
        YLocalSub.View( YLocal, _localTargetOffset, 0, SH._height, width );
        SH.MapMatrixNaivePassData( alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inSourceTeam )
        {
            mpi::Send
            ( SF.Z.LockedBuffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SF.Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv( SF.Z.Buffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
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
            ( SD.Z.LockedBuffer(), this->_height*width, 
              _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SD.Z.Resize( this->_height, width, this->_height );
            mpi::Recv
            ( SD.Z.Buffer(), this->_height*width, _rootOfOtherTeam, 0, comm );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(0,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(1,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(2,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
            }
            // Top-right quadrant
            node.Child(0,2).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
            node.Child(0,3).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
            node.Child(1,2).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
            node.Child(1,3).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );

            // Bottom-left quadrant
            node.Child(2,0).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
            node.Child(2,1).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
            node.Child(3,0).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
            node.Child(3,1).TransposeMapVectorNaivePassData
            ( alpha, xLocal, yLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(2,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(1,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal ); 
                node.Child(3,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(2,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(2,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(2,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,2).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,1).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                node.Child(3,0).TransposeMapVectorNaivePassData
                ( alpha, xLocal, yLocal );
                break;
            default:
                // This should be impossible
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
            if( _inTargetTeam )
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
        Vector<Scalar> xLocalSub, yLocalSub;
        xLocalSub.LockedView( xLocal, _localTargetOffset, SH._height );
        yLocalSub.View( yLocal, _localSourceOffset, SH._width );
        SH.TransposeMapVectorNaivePassData( alpha, xLocalSub, yLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaivePassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;

        MPI_Comm team = _subcomms->Subcomm( _level );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize == 2 )
        {
            if( teamRank == 0 )     
            {
                // Take care of the top-left quadrant within our subteams
                node.Child(0,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
            }
            else
            {
                // Take care of the bottom-right quadrant within our subteams
                node.Child(2,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
            }
            // Top-right quadrant
            node.Child(0,2).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(0,3).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(1,2).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(1,3).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );

            // Bottom-left quadrant
            node.Child(2,0).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(2,1).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(3,0).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
            node.Child(3,1).TransposeMapMatrixNaivePassData
            ( alpha, XLocal, YLocal );
        }
        else // teamSize >= 4
        {
            const int subteam = teamRank / (teamSize/4);
            switch( subteam )
            {
            case 0:
                // Take care of the work specific to our subteams
                node.Child(0,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(0,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(0,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            case 1:
                // Take care of the work specific to our subteams
                node.Child(1,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(1,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(1,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal ); 
                node.Child(3,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(1,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            case 2:
                // Take care of the work specific to our subteams
                node.Child(2,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 3
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(2,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            case 3:
                // Take care of the work specific to our subteams
                node.Child(3,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 2
                node.Child(2,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,2).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 1
                node.Child(1,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,1).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                // Interact with subteam 0
                node.Child(0,3).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                node.Child(3,0).TransposeMapMatrixNaivePassData
                ( alpha, XLocal, YLocal );
                break;
            default:
                // This should be impossible
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
            if( _inTargetTeam )
            {
                mpi::Send
                ( DF.Z.LockedBuffer(), DF.rank*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                DF.Z.Resize( DF.rank, width, DF.rank );
                mpi::Recv
                ( DF.Z.Buffer(), DF.rank*width, _rootOfOtherTeam, 0, comm );
            }
        }
        break;
    }
    case SPLIT_QUASI2D:
    {
        const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        XLocalSub.LockedView
        ( XLocal, _localTargetOffset, 0, SH._height, width );
        YLocalSub.View
        ( YLocal, _localSourceOffset, 0, SH._width, width );
        SH.TransposeMapMatrixNaivePassData( alpha, XLocalSub, YLocalSub );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            mpi::Send
            ( SF.Z.LockedBuffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        else
        {
            SF.Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv
            ( SF.Z.Buffer(), SF.rank*width, _rootOfOtherTeam, 0, comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        MPI_Comm comm = _subcomms->Subcomm( 0 );

        if( _inTargetTeam )
        {
            DenseMatrix<Scalar> XLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            if( XLocalSub.LDim() != XLocalSub.Height() )
            {
                // We must pack first
                SD.Z.Resize( this->_height, width, this->_height );
                for( int j=0; j<width; ++j )
                {
                    std::memcpy
                    ( SD.Z.Buffer(0,j), XLocalSub.LockedBuffer(0,j), 
                      this->_height*sizeof(Scalar) );
                }
                mpi::Send
                ( SD.Z.LockedBuffer(), this->_height*width, 
                  _rootOfOtherTeam, 0, comm );
            }
            else
            {
                mpi::Send
                ( XLocalSub.LockedBuffer(), this->_height*width, 
                  _rootOfOtherTeam, 0, comm );
            }
        }
        else
        {
            SD.Z.Resize( this->_height, width, this->_height );
            mpi::Recv
            ( SD.Z.Buffer(), this->_height*width, _rootOfOtherTeam, 0, comm );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapVectorNaivePassData");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorNaivePassData( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaivePassData
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaivePassData");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaivePassData( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorBroadcasts
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
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
    MapVectorBroadcastsCount( sizes, alpha, xLocal, yLocal );

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
    MapVectorBroadcastsPack( buffer, offsets, alpha, xLocal, yLocal );

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
    MapVectorBroadcastsUnpack( buffer, offsets, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MapMatrixBroadcastsCount( sizes, alpha, XLocal, YLocal );

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
    MapMatrixBroadcastsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    MapMatrixBroadcastsUnpack( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorBroadcasts
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
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
    TransposeMapVectorBroadcastsCount( sizes, alpha, xLocal, yLocal );

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
    TransposeMapVectorBroadcastsPack( buffer, offsets, alpha, xLocal, yLocal );

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
    TransposeMapVectorBroadcastsUnpack
    ( buffer, offsets, alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    // (the first and last comms are unneeded)
    const int numLevels = _subcomms->NumLevels();
    const int numBroadcasts = std::max(0,numLevels-2);
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    TransposeMapMatrixBroadcastsCount( sizes, alpha, XLocal, YLocal );

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
    TransposeMapMatrixBroadcastsPack( buffer, offsets, alpha, XLocal, YLocal );

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
    TransposeMapMatrixBroadcastsUnpack
    ( buffer, offsets, alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorBroadcasts
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorBroadcasts( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixBroadcasts( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorBroadcastsCount
( std::vector<int>& sizes,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorBroadcastsCount
                ( sizes, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsCount
( std::vector<int>& sizes,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
( std::vector<int>& sizes,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorBroadcastsCount
                ( sizes, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcastsCount
( std::vector<int>& sizes,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsCount");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsCount
                ( sizes, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
            sizes[_level-1] += shell.data.DF->rank*XLocal.Width();
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorBroadcastsPack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorBroadcastsPack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsPack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                std::memcpy
                ( &buffer[offsets[_level-1]], DF.Z.LockedBuffer(), 
                  DF.rank*width*sizeof(Scalar) );
                offsets[_level-1] += DF.rank*width;
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorBroadcastsUnpack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.z.Resize( DF.rank );
            std::memcpy
            ( DF.z.Buffer(), &buffer[offsets[_level-1]], 
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixBroadcastsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            std::memcpy
            ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
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
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorBroadcastsPack");
#endif
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorBroadcastsUnpack
                ( buffer, offsets, alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.z.Resize( DF.rank );
            std::memcpy
            ( DF.z.Buffer(), &buffer[offsets[_level-1]], 
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixBroadcastsUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixBroadcastsPack");
#endif
    const int width = XLocal.Width();
    const Shell& shell = _shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixBroadcastsUnpack
                ( buffer, offsets, alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            DF.Z.Resize( DF.rank, width, DF.rank );
            std::memcpy
            ( DF.Z.Buffer(), &buffer[offsets[_level-1]], 
              DF.rank*width*sizeof(Scalar) );
            offsets[_level-1] += DF.rank*width;
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
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorNaiveBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaiveBroadcasts
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.z.Resize( DF.rank );
            mpi::Broadcast( DF.z.Buffer(), DF.rank, 0, team );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaiveBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixNaiveBroadcasts");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixNaiveBroadcasts
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.Z.Resize( DF.rank, width, DF.rank );
            mpi::Broadcast( DF.Z.Buffer(), DF.rank*width, 0, team );
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
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorNaiveBroadcasts");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorNaiveBroadcasts
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.z.Resize( DF.rank );
            mpi::Broadcast( DF.z.Buffer(), DF.rank, 0, team );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaiveBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixNaiveBroadcasts");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixNaiveBroadcasts
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            const DistLowRankMatrix& DF = *shell.data.DF;
            MPI_Comm team = _subcomms->Subcomm( _level );
            DF.Z.Resize( DF.rank, width, DF.rank );
            mpi::Broadcast( DF.Z.Buffer(), DF.rank*width, 0, team );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapVectorNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapVectorNaiveBroadcasts( alpha, xLocal, yLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaiveBroadcasts
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::HermitianTransposeMapMatrixNaiveBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMapMatrixNaiveBroadcasts( alpha, XLocal, YLocal );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPostcompute
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPostcompute( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // yLocal += ULocal z
            const DistLowRankMatrix& DF = *shell.data.DF;
            blas::Gemv
            ( 'N', DF.ULocal.Height(), DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         DF.z.LockedBuffer(),      1,
              (Scalar)1, yLocal.Buffer(),          1 );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localSourceOffset, this->_width );
            yLocalSub.View( yLocal, _localTargetOffset, this->_height );
            SH.MapVectorPostcompute( alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localTargetOffset, SF.D.Height() );
            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const int localHeight = this->_height;
            const Scalar* zBuffer = SD.z.LockedBuffer();
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPostcompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapMatrixPostcompute( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
        {
            // YLocal += ULocal Z 
            const DistLowRankMatrix& DF = *shell.data.DF;
            blas::Gemm
            ( 'N', 'N', DF.ULocal.Height(), width, DF.rank,
              (Scalar)1, DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                         DF.Z.LockedBuffer(),      DF.Z.LDim(),
              (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
        }
        break;
    case SPLIT_QUASI2D:
        if( _inTargetTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localSourceOffset, 0, this->_width, width );
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, this->_height, width );
            SH.MapMatrixPostcompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localTargetOffset, 0, SF.D.Height(), width );
            hmatrix_tools::MatrixMatrix
            ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
        }
        break;
    case SPLIT_DENSE:
        if( _inTargetTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const int localHeight = this->_height;
            for( int j=0; j<width; ++j )
            {
                const Scalar* ZCol = SD.Z.LockedBuffer(0,j);
                Scalar* YCol = YLocal.Buffer(0,j);
                for( int i=0; i<localHeight; ++i )
                    YCol[i] += ZCol[i];
            }
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
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorPostcompute
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // yLocal += (VLocal^[T/H])^T z
            const DistLowRankMatrix& DF = *shell.data.DF;
            if( Conjugated )
            {
                // yLocal += conj(VLocal) z
                hmatrix_tools::Conjugate( DF.z );
                hmatrix_tools::Conjugate( yLocal );
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.z.LockedBuffer(),      1,
                  (Scalar)1, yLocal.Buffer(),          1 );
                hmatrix_tools::Conjugate( yLocal );
            }
            else
            {
                // yLocal += VLocal z
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.z.LockedBuffer(),      1,
                  (Scalar)1, yLocal.Buffer(),          1 );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, this->_height );
            yLocalSub.View( yLocal, _localSourceOffset, this->_width );
            SH.TransposeMapVectorPostcompute( alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SF.D.Width() );
            if( Conjugated )
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( SF.z );
                hmatrix_tools::Conjugate( yLocalSub );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocalSub );
                hmatrix_tools::Conjugate( yLocalSub );
            }
            else
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SD.D.Width() );
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SD.D, SD.z, (Scalar)1, yLocalSub );
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
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPostcompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::TransposeMapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapMatrixPostcompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^T Z 
            const DistLowRankMatrix& DF = *shell.data.DF;
            if( Conjugated )
            {
                // YLocal += conj(VLocal) Z
                hmatrix_tools::Conjugate( DF.Z );
                hmatrix_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmatrix_tools::Conjugate( YLocal );
            }
            else
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, this->_width, width );
            SH.TransposeMapMatrixPostcompute( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Width(), width );
            if( Conjugated )
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( SF.Z );
                hmatrix_tools::Conjugate( YLocalSub );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
                hmatrix_tools::Conjugate( YLocalSub );
            }
            else
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixTransposeMatrix
            ( alpha, SD.D, SD.Z, (Scalar)1, YLocalSub );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapVectorPostcompute
                ( alpha, xLocal, yLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // yLocal += (VLocal^[T/H])^H z
            const DistLowRankMatrix& DF = *shell.data.DF;
            if( Conjugated )
            {
                // yLocal += VLocal z
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.z.LockedBuffer(),      1,
                  (Scalar)1, yLocal.Buffer(),          1 );
            }
            else
            {
                // yLocal += conj(VLocal) z
                hmatrix_tools::Conjugate( DF.z );
                hmatrix_tools::Conjugate( yLocal );
                blas::Gemv
                ( 'N', DF.VLocal.Height(), DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.z.LockedBuffer(),      1,
                  (Scalar)1, yLocal.Buffer(),          1 );
                hmatrix_tools::Conjugate( yLocal );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            Vector<Scalar> xLocalSub, yLocalSub;
            xLocalSub.LockedView( xLocal, _localTargetOffset, this->_height );
            yLocalSub.View( yLocal, _localSourceOffset, this->_width );
            SH.HermitianTransposeMapVectorPostcompute
            ( alpha, xLocalSub, yLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SF.D.Width() );
            if( Conjugated )
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocalSub );
            }
            else
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( SF.z );
                hmatrix_tools::Conjugate( yLocalSub );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocalSub );
                hmatrix_tools::Conjugate( yLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, _localSourceOffset, SD.D.Width() );
            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SD.D, SD.z, (Scalar)1, yLocalSub );
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
HermitianTransposeMapMatrixPostcompute
( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::HermitianTransposeMapMatrixPostcompute");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).HermitianTransposeMapMatrixPostcompute
                ( alpha, XLocal, YLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet written");
#endif
        break;
    case DIST_LOW_RANK:
        if( _inSourceTeam )
        {
            // YLocal += (VLocal^[T/H])^H Z
            const DistLowRankMatrix& DF = *shell.data.DF;
            if( Conjugated )
            {
                // YLocal += VLocal Z
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
            }
            else
            {
                // YLocal += conj(VLocal) Z
                hmatrix_tools::Conjugate( DF.Z );
                hmatrix_tools::Conjugate( YLocal );
                blas::Gemm
                ( 'N', 'N', DF.VLocal.Height(), width, DF.rank,
                  (Scalar)1, DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                             DF.Z.LockedBuffer(),      DF.Z.LDim(),
                  (Scalar)1, YLocal.Buffer(),          YLocal.LDim() );
                hmatrix_tools::Conjugate( YLocal );
            }
        }
        break;
    case SPLIT_QUASI2D:
        if( _inSourceTeam )
        {
            const SplitQuasi2dHMatrix<Scalar,Conjugated>& SH = *shell.data.SH;
            DenseMatrix<Scalar> XLocalSub, YLocalSub;
            XLocalSub.LockedView
            ( XLocal, _localTargetOffset, 0, this->_height, width );
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, this->_width, width );
            SH.HermitianTransposeMapMatrixPostcompute
            ( alpha, XLocalSub, YLocalSub );
        }
        break;
    case SPLIT_LOW_RANK:
        if( _inSourceTeam )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SF.D.Width(), width );
            if( Conjugated )
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
            }
            else
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( SF.Z );
                hmatrix_tools::Conjugate( YLocalSub );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, SF.Z, (Scalar)1, YLocalSub );
                hmatrix_tools::Conjugate( YLocalSub );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar> YLocalSub;
            YLocalSub.View
            ( YLocal, _localSourceOffset, 0, SD.D.Width(), width );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SD.D, SD.Z, (Scalar)1, YLocalSub );
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
    case NODE:           delete shell.data.N; break;
    case NODE_SYMMETRIC: delete shell.data.NS; break;
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
        shell.data.N = 
            new Node
            ( H._xSizeSource, H._xSizeTarget,
              H._ySizeSource, H._ySizeTarget, H._zSize );
        Node& node = *shell.data.N;

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
