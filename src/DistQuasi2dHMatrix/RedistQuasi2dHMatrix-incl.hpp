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
                Write( hSource, DIST_NODE );
            }
            if( sourceRankOffset != targetRankOffset )
            {
                for( int i=0; i<teamSize; ++i )
                {
                    byte** hTarget = headPointers[targetRankOffset+i];
                    Write( hTarget, DIST_NODE );
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
                Write( hSource, DIST_NODE );
            }
            if( sourceRankOffset != targetRankOffset )
            {
                for( int i=0; i<teamSize; ++i )
                {
                    byte** hTarget = headPointers[targetRankOffset+i];
                    Write( hTarget, DIST_NODE );
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
            firstLocalIndex += (xSize*yBottomSize + xLeftSize*yTopSize)*zSize;
        else if( onTop )
            firstLocalIndex += xSize*yBottomSize*zSize;
        else if( onRight )
            firstLocalIndex += xLeftSize*yBottomSize*zSize;

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
            firstLocalIndex += xSize*yBottomSize*zSize;

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
    case DIST_NODE:      delete shell.data.DN; break;
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
    case DIST_NODE:
    { 
        shell.data.DN = 
            new DistNode
            ( H._xSizeSource, H._xSizeTarget,
              H._ySizeSource, H._ySizeTarget, H._zSize );
        DistNode& node = *shell.data.DN;

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

