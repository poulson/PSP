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
psp::DistQuasi2dHMat<Scalar,Conjugated>::PackedSizes
( std::vector<std::size_t>& packedSizes, 
  const Quasi2dHMat<Scalar,Conjugated>& H, const Teams& teams )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::PackedSizes");
#endif
    MPI_Comm comm = teams.Team(0);
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

    // Count the top-level header data
    const std::size_t headerSize = 13*sizeof(int) + sizeof(bool);
    for( unsigned i=0; i<p; ++i )
        packedSizes[i] += headerSize;

    // Recurse on this block to compute the packed sizes
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::Pack
( std::vector<byte*>& packedSubs, 
  const Quasi2dHMat<Scalar,Conjugated>& H, const Teams& teams )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Pack");
#endif
    MPI_Comm comm = teams.Team(0);
    const int p = mpi::CommSize( comm );
    std::vector<byte*> heads = packedSubs;
    std::vector<byte**> headPointers(p); 
    for( int i=0; i<p; ++i )
        headPointers[i] = &heads[i];

    // Write the top-level header data
    for( int i=0; i<p; ++i )
    {
        byte** h = headPointers[i];

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
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeLocalHeight
( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::ComputeLocalHeight");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localHeight;
    int xSize = H._xSizeTarget;
    int ySize = H._ySizeTarget;
    int zSize = H._zSize;
    ComputeLocalDimensionRecursion( localHeight, xSize, ySize, zSize, p, rank );
#ifndef RELEASE
    PopCallStack();
#endif
    return localHeight;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeLocalWidth
( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::ComputeLocalWidth");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localWidth;
    int xSize = H._xSizeSource;
    int ySize = H._ySizeSource;
    int zSize = H._zSize;
    ComputeLocalDimensionRecursion( localWidth, xSize, ySize, zSize, p, rank );
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidth;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeFirstLocalRow
( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::ComputeFirstLocalRow");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalRow = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalRow, H._xSizeTarget, H._ySizeTarget, H._zSize, p, rank );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalRow;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeFirstLocalCol
( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::ComputeFirstLocalCol");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalCol = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalCol, H._xSizeSource, H._ySizeSource, H._zSize, p, rank );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalCol;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeLocalSizes
( std::vector<int>& localSizes, const Quasi2dHMat<Scalar,Conjugated>& H )
{
#ifndef RELEASE    
    PushCallStack("DistQuasi2dHMat::ComputeLocalSizes");
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::PackedSizesRecursion
( std::vector<std::size_t>& packedSizes, 
  const std::vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMat<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMat<Scalar,Conjugated> Quasi2d;

    for( int i=0; i<teamSize; ++i )
        packedSizes[sourceRankOffset+i] += sizeof(BlockType);
    if( sourceRankOffset != targetRankOffset )
        for( int i=0; i<teamSize; ++i )
            packedSizes[targetRankOffset+i] += sizeof(BlockType);

    const typename Quasi2d::Block& block = H._block;
    const int m = H.Height();
    const int n = H.Width();
    switch( block.type )
    {
    case Quasi2d::NODE:
    {
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes, sourceRank, targetRank, 1,
                      block.data.N->Child(t,s) );
        }
        else if( teamSize == 2 )
        {
            // Give the upper-left 2x2 to the first halves of the teams 
            // and the lower-right 2x2 to the second halves.
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+s/2, targetRankOffset+t/2, 1,
                      block.data.N->Child(t,s) );
        }
        else // team Size >= 4
        {
            // Give each diagonal block of the 4x4 partition to a different
            // quarter of the teams
            const int newTeamSize = teamSize/4;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+newTeamSize*s,
                      targetRankOffset+newTeamSize*t, newTeamSize,
                      block.data.N->Child(t,s) );
        }
        break;
    }
    case Quasi2d::NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    }
    case Quasi2d::LOW_RANK:
    {
        const int r = block.data.F->Rank();
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
        const Dense<Scalar>& D = *block.data.D;
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::PackRecursion
( std::vector<byte**>& headPointers,
  const std::vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const Quasi2dHMat<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMat<Scalar,Conjugated> Quasi2d;

    const typename Quasi2d::Block& block = H._block;
    const int m = H.Height();
    const int n = H.Width();
    switch( block.type )
    {
    case Quasi2d::NODE:
    {
        const typename Quasi2d::Node& node = *block.data.N;
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                Write( headPointers[sourceRank], NODE );
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        PackRecursion
                        ( headPointers, localSizes, sourceRank, targetRank, 1,
                          node.Child(t,s) );
            }
            else
            {
                Write( headPointers[sourceRank], SPLIT_NODE );
                Write( headPointers[targetRank], SPLIT_NODE );
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        PackRecursion
                        ( headPointers, localSizes, sourceRank, targetRank, 1,
                          node.Child(t,s) );
            }
        }
        else if( teamSize == 2 )
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            for( int i=0; i<teamSize; ++i )
                Write( headPointers[sourceRankOffset+i], DIST_NODE );
            if( sourceRankOffset != targetRankOffset )
                for( int i=0; i<teamSize; ++i )
                    Write( headPointers[targetRankOffset+i], DIST_NODE );
            const int newTeamSize = teamSize/2;
            // Top-left block
            for( int t=0; t<2; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset, newTeamSize,
                      node.Child(t,s) );
            // Top-right block
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, targetRankOffset, 
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset+newTeamSize, 
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=2; t<4; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, 
                      targetRankOffset+newTeamSize,
                      newTeamSize, node.Child(t,s) );
        }
        else
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            for( int i=0; i<teamSize; ++i )
                Write( headPointers[sourceRankOffset+i], DIST_NODE );
            if( sourceRankOffset != targetRankOffset )
                for( int i=0; i<teamSize; ++i )
                    Write( headPointers[targetRankOffset+i], DIST_NODE );
            const int newTeamSize = teamSize/4;
            // Top-left block
            for( int t=0; t<2; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize, 
                      newTeamSize, node.Child(t,s) );
            // Top-right block
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize, 
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=2; t<4; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
        }
        break;
    }
    case Quasi2d::NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    }
    case Quasi2d::LOW_RANK:
    {
        const Dense<Scalar>& U = block.data.F->U;
        const Dense<Scalar>& V = block.data.F->V;
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
                    Write( h, U.LockedBuffer(0,j), m );
                for( int j=0; j<r; ++j )
                    Write( h, V.LockedBuffer(0,j), n );
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
                    Write( hSource, V.LockedBuffer(0,j), n );

                // Store the rank and entries of U on the target side
                Write( hTarget, r );
                for( int j=0; j<r; ++j )
                    Write( hTarget, U.LockedBuffer(0,j), m );
            }
        }
        else
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // NOTE: This should only happen when there is a weird
                //       admissibility condition that allows diagonal blocks
                //       to be low-rank.
#ifndef RELEASE
                std::cerr << "WARNING: Unlikely admissible case." << std::endl;
#endif

                // Store a distributed low-rank representation
                int offset = 0;
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
                        Write( h, U.LockedBuffer(offset,j), localSize );
                    for( int j=0; j<r; ++j )
                        Write( h, V.LockedBuffer(offset,j), localSize );
                    offset += localSize;
                }
            }
            else
            {
                // Store a distributed split low-rank representation

                // Store the source data
                int offset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    byte** hSource = headPointers[sourceRank];

                    Write( hSource, DIST_LOW_RANK );
                    Write( hSource, r );

                    const int localWidth = localSizes[sourceRank];
                    for( int j=0; j<r; ++j )
                        Write( hSource, V.LockedBuffer(offset,j), localWidth );
                    offset += localWidth;
                }

                // Store the target data
                offset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int targetRank = targetRankOffset + i;
                    byte** hTarget = headPointers[targetRank];
                    
                    Write( hTarget, DIST_LOW_RANK );
                    Write( hTarget, r );

                    const int localHeight = localSizes[targetRank];
                    for( int j=0; j<r; ++j )
                        Write( hTarget, U.LockedBuffer(offset,j), localHeight );
                    offset += localHeight;
                }
            }
        }
        break;
    }
    case Quasi2d::DENSE:
    {
        const Dense<Scalar>& D = *block.data.D;
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
                    for( int j=0; j<n; ++j )
                        Write( h, D.LockedBuffer(0,j), m );
                else
                    for( int j=0; j<n; ++j )
                        Write( h, D.LockedBuffer(j,j), m-j );
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
                    for( int j=0; j<n; ++j )
                        Write( hSource, D.LockedBuffer(0,j), m );
                else
                    for( int j=0; j<n; ++j )
                        Write( hSource, D.LockedBuffer(j,j), m-j );

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
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeLocalDimensionRecursion
( int& localDim, int& xSize, int& ySize, int& zSize, int p, int rank )
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

        xSize = ( onRight ? xRightSize : xLeftSize   );
        ySize = ( onTop   ? yTopSize   : yBottomSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, zSize, p/4, subteamRank );
    }
    else if( p == 2 )
    {
        const int subteam = rank/(p/2);
        const int subteamRank = rank-subteam*(p/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        xSize = xSize;
        ySize = ( subteam ? yTopSize : yBottomSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, zSize, p/2, subteamRank );
    }
    else // p == 1
        localDim = xSize*ySize*zSize;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeFirstLocalIndexRecursion
( int& firstLocalIndex, int xSize, int ySize, int zSize, int p, int rank )
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

        const int xSizeNew = ( onRight ? xRightSize : xLeftSize   );
        const int ySizeNew = ( onTop   ? yTopSize   : yBottomSize );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSizeNew, ySizeNew, zSize, p/4, subteamRank );
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

        const int ySizeNew = ( subteam ? yTopSize : yBottomSize );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSize, ySizeNew, zSize, p/2, subteamRank );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::ComputeLocalSizesRecursion
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
        localSizes[0] = xSize*ySize*zSize;
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( const byte* packedSub, const Teams& teams )
: _haveDenseUpdate(false), _storedDenseUpdate(false),
  _beganRowSpaceComp(false), _finishedRowSpaceComp(false),
  _beganColSpaceComp(false), _finishedColSpaceComp(false)
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::DistQuasi2dHMat");
#endif
    Unpack( packedSub, teams );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::DistQuasi2dHMat<Scalar,Conjugated>::Unpack
( const byte* packedDistHMat, const Teams& teams )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Unpack");
#endif
    _teams = &teams;
    _level = 0;
    _inSourceTeam = true;
    _inTargetTeam = true;
    _sourceRoot = 0;
    _targetRoot = 0;

    const byte* head = packedDistHMat;
    
    // Read in the header information
    _numLevels          = Read<int>( head );
    _maxRank            = Read<int>( head );
    _sourceOffset       = Read<int>( head );
    _targetOffset       = Read<int>( head );
    //_type             = Read<MatrixType>( head );
    _stronglyAdmissible = Read<bool>( head );
    _xSizeSource        = Read<int>( head );
    _xSizeTarget        = Read<int>( head );
    _ySizeSource        = Read<int>( head );
    _ySizeTarget        = Read<int>( head );
    _zSize              = Read<int>( head );
    _xSource            = Read<int>( head );
    _xTarget            = Read<int>( head );
    _ySource            = Read<int>( head );
    _yTarget            = Read<int>( head );

    UnpackRecursion( head );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedDistHMat);
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::UnpackRecursion
( const byte*& head )
{
    MPI_Comm team = _teams->Team( _level );
    if( !_inSourceTeam && !_inTargetTeam )
    {
        _block.type = EMPTY;
        return;
    }

    // Read in the information for the new block
    _block.Clear();
    _block.type = Read<BlockType>( head );
    const int m = Height();
    const int n = Width();
    switch( _block.type )
    {
    case DIST_NODE:
    { 
        _block.data.N = NewNode();
        Node& node = *_block.data.N;

        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize >= 4 )
        {
            const int subteam = teamRank/(teamSize/4);
            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget+1,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget+1,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
        }
        else  // teamSize == 2
        {
#ifndef RELEASE
            if( teamSize != 2 )
                throw std::logic_error("Team size was not 2 as expected");
#endif
            const bool inUpperTeam = ( teamRank == 1 );
            const bool inLeftSourceTeam = ( !inUpperTeam && _inSourceTeam );
            const bool inRightSourceTeam = ( inUpperTeam && _inSourceTeam );
            const bool inTopTargetTeam = ( !inUpperTeam && _inTargetTeam );
            const bool inBottomTargetTeam = ( inUpperTeam && _inTargetTeam );

            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget,
                          *_teams, _level+1,
                          inLeftSourceTeam, inTopTargetTeam,
                          _sourceRoot, _targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget,
                          *_teams, _level+1,
                          inRightSourceTeam, inTopTargetTeam,
                          _sourceRoot+1, _targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget+1,
                          *_teams, _level+1,
                          inLeftSourceTeam, inBottomTargetTeam,
                          _sourceRoot, _targetRoot+1 );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget+1,
                          *_teams, _level+1,
                          inRightSourceTeam, inBottomTargetTeam,
                          _sourceRoot+1, _targetRoot+1 );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        _block.data.N = NewNode();
        Node& node = *_block.data.N;

        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                node.children[s+4*t] = 
                    new DistQuasi2dHMat<Scalar,Conjugated>
                    ( _numLevels-1, _maxRank, _stronglyAdmissible,
                      _sourceOffset+sOffset, _targetOffset+tOffset,
                      node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                      node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                      _zSize,
                      2*_xSource+(s&1), 2*_xTarget+(t&1),
                      2*_ySource+(s/2), 2*_yTarget+(t/2),
                      *_teams, _level+1, 
                      _inSourceTeam, _inTargetTeam,
                      _sourceRoot, _targetRoot );
                node.Child(t,s).UnpackRecursion( head );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        _block.data.DF = new DistLowRank;
        DistLowRank& DF = *_block.data.DF;

        DF.rank = Read<int>( head );
        if( _inSourceTeam )
        {
            // Read in V
            const int localWidth = LocalWidth();
            DF.VLocal.SetType( GENERAL );
            DF.VLocal.Resize( localWidth, DF.rank );
            for( int j=0; j<DF.rank; ++j )
                Read( DF.VLocal.Buffer(0,j), head, localWidth );
        }
        if( _inTargetTeam )
        {
            // Read in U 
            const int localHeight = LocalHeight();
            DF.ULocal.SetType( GENERAL );
            DF.ULocal.Resize( localHeight, DF.rank );
            for( int j=0; j<DF.rank; ++j )
                Read( DF.ULocal.Buffer(0,j), head, localHeight );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        _block.data.SF = new SplitLowRank;
        SplitLowRank& SF = *_block.data.SF;

        SF.rank = Read<int>( head );

        SF.D.SetType( GENERAL );
        if( _inSourceTeam )
        {
            // Read in V
            SF.D.Resize( n, SF.rank );
            for( int j=0; j<SF.rank; ++j )
                Read( SF.D.Buffer(0,j), head, n );
        }
        else
        {
            // Read in U
            SF.D.Resize( m, SF.rank );
            for( int j=0; j<SF.rank; ++j )
                Read( SF.D.Buffer(0,j), head, m );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        _block.data.SD = new SplitDense;
        SplitDense& SD = *_block.data.SD;

        if( _inSourceTeam )
        {
            const MatrixType type = Read<MatrixType>( head );
            SD.D.SetType( type );
            SD.D.Resize( m, n );
            if( type == GENERAL )
                for( int j=0; j<n; ++j )
                    Read( SD.D.Buffer(0,j), head, m );
            else
                for( int j=0; j<n; ++j )
                    Read( SD.D.Buffer(j,j), head, m-j );
        }
        break;
    }
    case LOW_RANK:
    {
        _block.data.F = new LowRank<Scalar,Conjugated>;
        LowRank<Scalar,Conjugated>& F = *_block.data.F;

        // Read in the rank
        const int r = Read<int>( head );

        // Read in U
        F.U.SetType( GENERAL ); F.U.Resize( m, r );
        for( int j=0; j<r; ++j )
            Read( F.U.Buffer(0,j), head, m );

        // Read in V
        F.V.SetType( GENERAL ); F.V.Resize( n, r );
        for( int j=0; j<r; ++j )
            Read( F.V.Buffer(0,j), head, n );
        break;
    }
    case DENSE:
    {
        _block.data.D = new Dense<Scalar>;
        Dense<Scalar>& D = *_block.data.D;

        const MatrixType type = Read<MatrixType>( head );

        D.SetType( type );
        D.Resize( m, n );
        if( type == GENERAL )
            for( int j=0; j<n; ++j )
                Read( D.Buffer(0,j), head, m );
        else
            for( int j=0; j<n; ++j )
                Read( D.Buffer(j,j), head, m-j );
        break;
    }
    default:
#ifndef RELEASE
        throw std::logic_error("Should not need to unpack empty submatrix");
#endif
        break;
    }
}

