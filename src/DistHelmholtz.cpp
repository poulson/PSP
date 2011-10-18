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

template<typename F>
psp::DistHelmholtz<F>::DistHelmholtz
( const FiniteDiffControl<F>& control, elemental::mpi::Comm comm )
: comm_(comm), control_(control),
  hx_(control.wx/(control.nx+1)),
  hy_(control.wy/(control.ny+1)),
  hz_(control.wz/(control.nz+1)),
  bx_(control.etax/hx_),
  by_(control.etay/hy_),
  bz_(control.etaz/hz_),
  bzCeil_(std::ceil(control.etaz/hz_)),
  initialized_(false)
{
    // Provide some notational shortcuts
    const int nx = control.nx;
    const int ny = control.ny;
    const int nz = control.nz;
    const int planesPerPanel = control.planesPerPanel;
    const int bzCeil = bzCeil_;
    const int cutoff = 10; // note that this is for a z-depth of 1!!!

    // Pull out some information about our communicator
    const int commRank = elemental::mpi::CommRank( comm );
    const int commSize = elemental::mpi::CommSize( comm );
    unsigned temp = commSize;
    unsigned log2CommSize = 0;
    while( temp >>= 1 )
        ++log2CommSize;

    // Decide if the domain is sufficiently deep to warrant sweeping
    const bool bottomHasPML = (control.bottomBC == PML);
    const int topDepth = bzCeil+planesPerPanel;
    const int bottomOrigDepth = 
        (bottomHasPML ? bzCeil+planesPerPanel : planesPerPanel );
    if( nz <= topDepth+bottomOrigDepth )
        throw std::logic_error
        ("The domain is very shallow. Please run a sparse-direct factorization "
         "instead.");

    // Compute the depths of each interior panel class and the number of 
    // full inner panels.
    //
    //    -----------
    //   | Top       |
    //   | Inner     |
    //       ...     
    //   | Leftover? |
    //   | Bottom    |
    //    -----------
    const int innerDepth = nz-(topDepth+bottomOrigDepth);
    const int leftoverInnerDepth = innerDepth % planesPerPanel;
    const bool haveLeftover = ( leftoverInnerDepth != 0 );
    const int numFullInnerPanels = innerDepth / planesPerPanel;

    // Compute the number of rows we own of the sparse distributed matrix
    const int localTopHeight = 
        LocalPanelHeight( nx, ny, topDepth, cutoff, commRank, log2CommSize );
    const int localFullInnerHeight = 
        LocalPanelHeight
        ( nx, ny, planesPerPanel, cutoff, commRank, log2CommSize );
    const int localLeftoverInnerHeight = 
        LocalPanelHeight
        ( nx, ny, leftoverInnerDepth, cutoff, commRank, log2CommSize );
    const int localBottomHeight = 
        LocalPanelHeight
        ( nx, ny, bottomOrigDepth, cutoff, commRank, log2CommSize );
    localHeight_ = localTopHeight + numFullInnerPanels*localFullInnerHeight + 
                   localLeftoverInnerHeight + localBottomHeight;

    // Compute the natural indices of our local indices and fill in the offsets
    // for the store of each local rows indices.
    localToNaturalMap_.resize( localHeight_ );
    localRowOffsets_.resize( localHeight_+1 );
    localRowOffsets_[0] = 0;
    int localOffset=0, zOffset=0;
    MapLocalPanelIndices
    ( nx, ny, nz, topDepth, zOffset, cutoff, commRank, log2CommSize, 
      localToNaturalMap_, localRowOffsets_, localOffset );
    for( int i=0; i<numFullInnerPanels; ++i )
        MapLocalPanelIndices
        ( nx, ny, nz, planesPerPanel, zOffset, cutoff, commRank, log2CommSize,
          localToNaturalMap_, localRowOffsets_, localOffset );
    if( haveLeftover )
        MapLocalPanelIndices
        ( nx, ny, nz, leftoverInnerDepth, zOffset, cutoff, 
          commRank, log2CommSize, localToNaturalMap_, localRowOffsets_, 
          localOffset );
    MapLocalPanelIndices
    ( nx, ny, nz, bottomOrigDepth, zOffset, cutoff, commRank, log2CommSize,
      localToNaturalMap_, localRowOffsets_, localOffset );

    // Fill in the natural connection indices
    std::vector<int> localConnections( localRowOffsets_.back() );
    localOffset = 0;
    zOffset = 0;
    MapLocalConnectionIndices
    ( nx, ny, nz, topDepth, zOffset, cutoff, commRank, log2CommSize, 
      localConnections, localOffset );
    for( int i=0; i<numFullInnerPanels; ++i )
        MapLocalConnectionIndices
        ( nx, ny, nz, planesPerPanel, zOffset, cutoff, commRank, log2CommSize,
          localConnections, localOffset );
    if( haveLeftover )
        MapLocalConnectionIndices
        ( nx, ny, nz, leftoverInnerDepth, zOffset, cutoff, 
          commRank, log2CommSize, localConnections, localOffset );
    MapLocalConnectionIndices
    ( nx, ny, nz, bottomOrigDepth, zOffset, cutoff, commRank, log2CommSize,
      localConnections, localOffset );
#ifndef RELEASE
    const int numLocalConnections = localConnections.size();
    if( numLocalConnections != localOffset )
        throw std::logic_error("Invalid connection count");
#endif

    // Count the number of indices that we will need to recv from each process.
    actualRecvSizes_.resize( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalRow = localToNaturalMap_[iLocal];
        const int rowOffset = localRowOffsets_[iLocal];
        const int rowSize = localRowOffsets_[iLocal+1]-rowOffset;
        // skip the diagonal value...
        for( int k=1; k<rowSize; ++k )
        {
            const int naturalCol = localConnections[rowOffset+k];
            const int x = naturalCol % nx;
            const int y = (naturalCol/nx) % ny;
            const int z = naturalCol/(nx*ny);
            const int zLocal = 
                LocalZ
                ( z, topDepth, innerDepth, bottomOrigDepth, planesPerPanel );
            const int process = 
                OwningProcess( x, y, zLocal, nx, ny, log2CommSize );
            ++actualRecvSizes_[process];
        }
    }
    const int maxRecvSize = 
        *std::max_element( actualRecvSizes_.begin(), actualRecvSizes_.end() );
    std::vector<int> synchMessageSends( 2*commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        synchMessageSends[2*proc+0] = actualRecvSizes_[proc];
        synchMessageSends[2*proc+1] = maxRecvSize;
    }
    std::vector<int> synchMessageRecvs( 2*commSize );
    clique::mpi::AllToAll
    ( &synchMessageSends[0], 2, &synchMessageRecvs[0], 2, comm );
    synchMessageSends.clear();
    actualSendSizes_.resize( commSize );
    for( int proc=0; proc<commSize; ++proc )
        actualSendSizes_[proc] = synchMessageRecvs[2*proc];
    allToAllSize_ = 0;
    for( int proc=0; proc<commSize; ++proc )
        allToAllSize_ = std::max(allToAllSize_,synchMessageRecvs[2*proc+1]);
    synchMessageRecvs.clear();

    // Pack and send the natural indices
    std::vector<int> packOffsets( commSize );
    std::vector<int> naturalIndexSends( commSize*allToAllSize_ );
    for( int proc=0; proc<commSize; ++proc )
        packOffsets[proc] = proc*allToAllSize_;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalRow = localToNaturalMap_[iLocal];
        const int rowOffset = localRowOffsets_[iLocal];
        const int rowSize = localRowOffsets_[iLocal+1]-rowOffset;
        // skip the diagonal value...
        for( int k=1; k<rowSize; ++k )
        {
            const int naturalCol = localConnections[rowOffset+k];
            const int x = naturalCol % nx;
            const int y = (naturalCol/nx) % ny;
            const int z = naturalCol/(nx*ny);
            const int zLocal = 
                LocalZ
                ( z, topDepth, innerDepth, bottomOrigDepth, planesPerPanel );
            const int process = 
                OwningProcess( x, y, zLocal, nx, ny, log2CommSize );

            naturalIndexSends[packOffsets[process]++] = naturalCol;
        }
    }
    sendIndices_.resize( commSize*allToAllSize_ );
    clique::mpi::AllToAll
    ( &naturalIndexSends[0], allToAllSize_, 
      &sendIndices_[0],      allToAllSize_, comm );
    naturalIndexSends.clear();

    // Invert the local to natural map
    std::map<int,int> naturalToLocalMap;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
        naturalToLocalMap[localToNaturalMap_[iLocal]] = iLocal;

    // Convert the recv indices in place
    for( int proc=0; proc<commSize; ++proc )
    {
        const int actualSendSize = actualSendSizes_[proc];
        int* thisSend = &sendIndices_[proc*allToAllSize_];
        for( int i=0; i<actualSendSize; ++i )
            thisSend[i] = naturalToLocalMap[thisSend[i]];
    }

    // Count the number of local supernodes in each panel
    const int numLocalSupernodes = 
        NumLocalSupernodes( nx, ny, cutoff, commRank, log2CommSize );

    // Create space for the original structures of the panel classes
    clique::symbolic::SymmOrig 
        topSymbolicOrig, 
        fullInnerSymbolicOrig, 
        leftoverInnerSymbolicOrig, 
        bottomSymbolicOrig;
    topSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    topSymbolicOrig.dist.supernodes.resize( log2CommSize );
    fullInnerSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    fullInnerSymbolicOrig.dist.supernodes.resize( log2CommSize );
    leftoverInnerSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    leftoverInnerSymbolicOrig.dist.supernodes.resize( log2CommSize );
    bottomSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    bottomSymbolicOrig.dist.supernodes.resize( log2CommSize );

    // Fill the original structures (in the nested-dissection ordering)
    //
    // In order to minimize the number of symbolic factorizations that have to 
    // be performed, and to simplify distribution issues, the leading PML region
    // on each inner panel will always be ordered _LAST_ within that panel.
    FillOrigPanelStruct
    ( nx, ny, topDepth, cutoff, comm, log2CommSize, topSymbolicOrig );
    FillOrigPanelStruct
    ( nx, ny, planesPerPanel+bzCeil, cutoff, comm, log2CommSize, 
      fullInnerSymbolicOrig );
    if( haveLeftover )
        FillOrigPanelStruct
        ( nx, ny, leftoverInnerDepth+bzCeil, cutoff, comm, log2CommSize,
          leftoverInnerSymbolicOrig );
    FillOrigPanelStruct
    ( nx, ny, bottomOrigDepth+bzCeil, cutoff, comm, log2CommSize,
      bottomSymbolicOrig );

    // Perform the parallel symbolic factorizations
    clique::symbolic::SymmetricFactorization
    ( topSymbolicOrig, topSymbolicFact_, true );
    clique::symbolic::SymmetricFactorization
    ( fullInnerSymbolicOrig, fullInnerSymbolicFact_, true );
    if( haveLeftover )
        clique::symbolic::SymmetricFactorization
        ( leftoverInnerSymbolicOrig, leftoverInnerSymbolicFact_, true );
    clique::symbolic::SymmetricFactorization
    ( bottomSymbolicOrig, bottomSymbolicFact_, true );
}

template<typename F>
void 
psp::DistHelmholtz<F>::RecursiveReordering
( int nx, int xOffset, int xSize, int yOffset, int ySize,
  int cutoff, int depthTilSerial, int* reordering ) 
{
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
    {
        // Write the leaf
        for( int x=xOffset; x<xOffset+xSize; ++x )
            for( int y=yOffset; y<yOffset+ySize; ++y )
                reordering[(x-xOffset)*ySize+(y-yOffset)] = x+y*nx;
    }
    else if( xSize >= ySize )
    {
        // Cut the x dimension and write the separator
        const int xLeftSize = (xSize-1) / 2;
        const int separatorSize = ySize;
        int* separatorSection = &reordering[xSize*ySize-separatorSize];
        for( int y=yOffset; y<yOffset+ySize; ++y )
            separatorSection[y-yOffset] = (xOffset+xLeftSize)+y*nx;
        // Recurse on the left side of the x cut
        RecursiveReordering
        ( nx, xOffset, xLeftSize, yOffset, ySize, 
          cutoff, std::max(depthTilSerial-1,0), reordering );
        // Recurse on the right side of the x cut
        RecursiveReordering
        ( nx, xOffset+(xLeftSize+1), xSize-(xLeftSize+1), yOffset, ySize,
          cutoff, std::max(depthTilSerial-1,0), &reordering[xLeftSize*ySize] );
    }
    else
    {
        // Cut the y dimension and write the separator
        const int yLeftSize = (ySize-1) / 2;
        const int separatorSize = xSize;
        int* separatorSection = &reordering[xSize*ySize-separatorSize];
        for( int x=xOffset; x<xOffset+xSize; ++x )
            separatorSection[x-xOffset] = x+(yOffset+yLeftSize)*nx;
        // Recurse on the left side of the y cut
        RecursiveReordering
        ( nx, xOffset, xSize, yOffset, yLeftSize, 
          cutoff, std::max(depthTilSerial-1,0), reordering );
        // Recurse on the right side of the y cut
        RecursiveReordering
        ( nx, xOffset, xSize, yOffset+(yLeftSize+1), ySize-(yLeftSize+1),
          cutoff, std::max(depthTilSerial-1,0), &reordering[xSize*yLeftSize] );
    }
}

template<typename F>
int
psp::DistHelmholtz<F>::LocalPanelHeight
( int xSize, int ySize, int zSize, int cutoff, 
  unsigned commRank, unsigned log2CommSize ) 
{
    int localHeight = 0;
    LocalPanelHeightRecursion
    ( xSize, ySize, zSize, cutoff, commRank, log2CommSize, localHeight );
    return localHeight;
}

template<typename F>
void
psp::DistHelmholtz<F>::LocalPanelHeightRecursion
( int xSize, int ySize, int zSize, int cutoff, 
  unsigned commRank, unsigned log2CommSize, int& localHeight ) 
{
    if( log2CommSize == 0 && xSize*ySize <= cutoff )
    {
        // Add the leaf
        localHeight += xSize*ySize;
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        // Add our local portion of the partition
        const int commSize = 1u<<log2CommSize;
        localHeight += 
            elemental::LocalLength( ySize*zSize, commRank, commSize );

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xLeftSize, ySize, zSize, cutoff, 0, 0, localHeight );
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize-(xLeftSize+1), ySize, zSize, cutoff, 0, 0, localHeight );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize-(xLeftSize+1), ySize, zSize, cutoff,
              commRank/2, log2CommSize-1, localHeight );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xLeftSize, ySize, zSize, cutoff,
              commRank/2, log2CommSize-1, localHeight );
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //

        // Add our local portion of the partition
        const int commSize = 1u<<log2CommSize;
        localHeight +=
            elemental::LocalLength( xSize*zSize, commRank, commSize );

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xSize, yLeftSize, zSize, cutoff, 0, 0, localHeight );
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize, ySize-(yLeftSize+1), zSize, cutoff, 0, 0, localHeight );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize, ySize-(yLeftSize+1), zSize, cutoff, 
              commRank/2, log2CommSize-1, localHeight );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xSize, yLeftSize, zSize, cutoff,
              commRank/2, log2CommSize-1, localHeight );
        }
    }
}

template<typename F>
int
psp::DistHelmholtz<F>::NumLocalSupernodes
( int xSize, int ySize, int cutoff, unsigned commRank, unsigned log2CommSize )
{
    int numLocalSupernodes = 0;
    NumLocalSupernodesRecursion
    ( xSize, ySize, cutoff, commRank, log2CommSize, numLocalSupernodes );
    return numLocalSupernodes;
}

template<typename F>
void
psp::DistHelmholtz<F>::NumLocalSupernodesRecursion
( int xSize, int ySize, int cutoff, unsigned commRank, unsigned log2CommSize, 
  int& numLocal ) 
{
    if( log2CommSize == 0 && xSize*ySize <= cutoff )
    {
        ++numLocal;
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        // Add our local portion of the partition
        if( log2CommSize == 0 )
            ++numLocal;

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            NumLocalSupernodesRecursion
            ( xLeftSize, ySize, cutoff, 0, 0, numLocal );
            // Add the right side
            NumLocalSupernodesRecursion
            ( xSize-(xLeftSize+1), ySize, cutoff, 0, 0, numLocal );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            NumLocalSupernodesRecursion
            ( xSize-(xLeftSize+1), ySize, cutoff, commRank/2, log2CommSize-1, 
              numLocal );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            NumLocalSupernodesRecursion
            ( xLeftSize, ySize, cutoff, commRank/2, log2CommSize-1, numLocal );
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //

        // Add our local portion of the partition
        if( log2CommSize == 0 )
            ++numLocal;

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            NumLocalSupernodesRecursion
            ( xSize, yLeftSize, cutoff, 0, 0, numLocal );
            // Add the right side
            NumLocalSupernodesRecursion
            ( xSize, ySize-(yLeftSize+1), cutoff, 0, 0, numLocal );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            NumLocalSupernodesRecursion
            ( xSize, ySize-(yLeftSize+1), cutoff, commRank/2, log2CommSize-1, 
              numLocal );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            NumLocalSupernodesRecursion
            ( xSize, yLeftSize, cutoff, commRank/2, log2CommSize-1, numLocal );
        }
    }
}

template<typename F>
int 
psp::DistHelmholtz<F>::LocalZ
( int z, int topDepth, int innerDepth, int bottomOrigDepth, int planesPerPanel )
{
    if( z < topDepth )
    {
        return z;
    }
    else if( z < topDepth + innerDepth )
    {
        return (z-topDepth) % planesPerPanel;
    }
    else // z in [topDepth+innerDepth,topDepth+innerDepth+bottomOrigDepth)
    {
#ifndef RELEASE
        if( z < topDepth+innerDepth || 
            z >= topDepth+innerDepth+bottomOrigDepth )
            throw std::logic_error("z is out of bounds");
#endif
        return z - (topDepth+innerDepth);
    }
}

template<typename F>
int 
psp::DistHelmholtz<F>::LocalPanelOffset
( int z,       
  int topDepth, int innerDepth, int planesPerPanel, int bottomOrigDepth,
  int numFullInnerPanels,
  int localTopHeight, int localInnerHeight, int localLeftoverHeight,
  int localBottomHeight )
{
    if( z < topDepth )
    {
        return 0;
    }
    else if( z < topDepth + innerDepth )
    {
        return localTopHeight + ((z-topDepth)/planesPerPanel)*localInnerHeight;
    }
    else
    {
        return localTopHeight + numFullInnerPanels*localInnerHeight + 
               localLeftoverHeight;
    }
}

template<typename F>
void
psp::DistHelmholtz<F>::MapLocalPanelIndices
( int nx, int ny, int nz, int zSize, int& zOffset, int cutoff, 
  unsigned commRank, unsigned log2CommSize,
  std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
  int& localOffset )
{
    MapLocalPanelIndicesRecursion
    ( nx, ny, nz, nx, ny, zSize, 0, 0, zOffset, cutoff, commRank, log2CommSize, 
      localToNaturalMap, localRowOffsets, localOffset );
    zOffset += zSize;
}

template<typename F>
void
psp::DistHelmholtz<F>::MapLocalPanelIndicesRecursion
( int nx, int ny, int nz, int xSize, int ySize, int zSize, 
  int xOffset, int yOffset, int zOffset, int cutoff, 
  unsigned commRank, unsigned log2CommSize,
  std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
  int& localOffset )
{
    if( log2CommSize == 0 && xSize*ySize <= cutoff )
    {
        // Add the leaf
        for( int zDelta=0; zDelta<zSize; ++zDelta )
        {
            const int z = zOffset + zDelta;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                for( int xDelta=0; xDelta<xSize; ++xDelta )
                {
                    const int x = xOffset + xDelta;

                    // Map this local entry to the global natural index
                    localToNaturalMap[localOffset] = x + y*nx + z*nx*ny;

                    // Compute the number of connections from this row
                    int numConnections = 1; // always count diagonal
                    if( x > 0 )
                        ++numConnections;
                    if( x < nx ) 
                        ++numConnections;
                    if( y > 0 )
                        ++numConnections;
                    if( y < ny )
                        ++numConnections;
                    if( z > 0 )
                        ++numConnections;
                    if( z < nz )
                        ++numConnections;
                    localRowOffsets[localOffset+1] = 
                        localRowOffsets[localOffset] + numConnections;

                    ++localOffset;
                }
            }
        }
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the y dimension
        //

        // Add the left and/or right sides
        const int commSize = 1u<<log2CommSize;
        const int yLeftSize = (ySize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, xOffset, yOffset, zOffset,
              cutoff, commRank/2, log2CommSize-1, 
              localToNaturalMap, localRowOffsets, localOffset );
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize, 
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff, 
              commRank/2, log2CommSize-1, localToNaturalMap, localRowOffsets,
              localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize,
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff,
              0, 0, localToNaturalMap, localRowOffsets, localOffset );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, xOffset, yOffset, zOffset,
              cutoff, 0, 0, localToNaturalMap, localRowOffsets, localOffset );
        }
        
        // Add our local portion of the partition
        const int localHeight = 
            elemental::LocalLength( xSize*zSize, commRank, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = commRank + iLocal*commSize;
            const int xDelta = i % xSize;
            const int zDelta = i / xSize;
            const int x = xOffset + xDelta;
            const int y = yOffset + yLeftSize;
            const int z = zOffset + zDelta;

            // Map this local entry to the global natrual index
            localToNaturalMap[localOffset] = x + y*nx + z*nx*ny;

            // Compute the number of connections from this row
            int numConnections = 1; // always count diagonal
            if( x > 0 )
                ++numConnections;
            if( x < nx )
                ++numConnections;
            if( y > 0 )
                ++numConnections;
            if( y < ny )
                ++numConnections;
            if( z > 0 )
                ++numConnections;
            if( z < nz )
                ++numConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numConnections;

            ++localOffset;
        }
    }
    else
    {
        //
        // Cut the x dimension
        //

        // Add the left and/or right sides
        const int commSize = 1u<<log2CommSize;
        const int xLeftSize = (xSize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xLeftSize, xSize, zSize, xOffset, yOffset, zOffset,
              cutoff, commRank/2, log2CommSize-1, 
              localToNaturalMap, localRowOffsets, localOffset );
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize, 
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff, 
              commRank/2, log2CommSize-1, localToNaturalMap, localRowOffsets,
              localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize,
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff,
              0, 0, localToNaturalMap, localRowOffsets, localOffset );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xLeftSize, ySize, zSize, xOffset, yOffset, zOffset,
              cutoff, 0, 0, localToNaturalMap, localRowOffsets, localOffset );
        }
        
        // Add our local portion of the partition
        const int localHeight = 
            elemental::LocalLength( ySize*zSize, commRank, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = commRank + iLocal*commSize;
            const int yDelta = i % ySize;
            const int zDelta = i / ySize;
            const int x = xOffset + xLeftSize;
            const int y = yOffset + yDelta;
            const int z = zOffset + zDelta;

            // Map this local entry to the global natrual index
            localToNaturalMap[localOffset] = x + y*nx + z*nx*ny;

            // Compute the number of connections from this row
            int numConnections = 1; // always count diagonal
            if( x > 0 )
                ++numConnections;
            if( x < nx )
                ++numConnections;
            if( y > 0 )
                ++numConnections;
            if( y < ny )
                ++numConnections;
            if( z > 0 )
                ++numConnections;
            if( z < nz )
                ++numConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numConnections;

            ++localOffset;
        }
    }
}

template<typename F>
void
psp::DistHelmholtz<F>::MapLocalConnectionIndices
( int nx, int ny, int nz, int zSize, int& zOffset, int cutoff, 
  unsigned commRank, unsigned log2CommSize,
  std::vector<int>& localConnections, int& localOffset )
{
    MapLocalConnectionIndicesRecursion
    ( nx, ny, nz, nx, ny, zSize, 0, 0, zOffset, cutoff, commRank, log2CommSize, 
      localConnections, localOffset );
    zOffset += zSize;
}

template<typename F>
void
psp::DistHelmholtz<F>::MapLocalConnectionIndicesRecursion
( int nx, int ny, int nz, int xSize, int ySize, int zSize, 
  int xOffset, int yOffset, int zOffset, int cutoff, 
  unsigned commRank, unsigned log2CommSize,
  std::vector<int>& localConnections, int& localOffset )
{
    if( log2CommSize == 0 && xSize*ySize <= cutoff )
    {
        // Add the leaf
        for( int zDelta=0; zDelta<zSize; ++zDelta )
        {
            const int z = zOffset + zDelta;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                for( int xDelta=0; xDelta<xSize; ++xDelta )
                {
                    const int x = xOffset + xDelta;

                    localConnections[localOffset++] = x + y*nx + z*nx*ny;
                    if( x > 0 )
                        localConnections[localOffset++] = (x-1)+y*nx+z*nx*ny;
                    if( x < nx ) 
                        localConnections[localOffset++] = (x+1)+y*nx+z*nx*ny;
                    if( y > 0 )
                        localConnections[localOffset++] = x+(y-1)*nx+z*nx*ny;
                    if( y < ny )
                        localConnections[localOffset++] = x+(y+1)*nx+z*nx*ny;
                    if( z > 0 )
                        localConnections[localOffset++] = x+y*nx+(z-1)*nx*ny;
                    if( z < nz )
                        localConnections[localOffset++] = x+y*nx+(z+1)*nx*ny;
                }
            }
        }
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the y dimension
        //

        // Add the left and/or right sides
        const int commSize = 1u<<log2CommSize;
        const int yLeftSize = (ySize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, xOffset, yOffset, zOffset,
              cutoff, commRank/2, log2CommSize-1, localConnections, 
              localOffset );
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize, 
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff, 
              commRank/2, log2CommSize-1, localConnections, localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize,
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff,
              0, 0, localConnections, localOffset );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, xOffset, yOffset, zOffset,
              cutoff, 0, 0, localConnections, localOffset );
        }
        
        // Add our local portion of the partition
        const int localHeight = 
            elemental::LocalLength( xSize*zSize, commRank, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = commRank + iLocal*commSize;
            const int xDelta = i % xSize;
            const int zDelta = i / xSize;
            const int x = xOffset + xDelta;
            const int y = yOffset + yLeftSize;
            const int z = zOffset + zDelta;

            localConnections[localOffset++] = x + y*nx + z*nx*ny;
            if( x > 0 )
                localConnections[localOffset++] = (x-1) + y*nx + z*nx*ny;
            if( x < nx )
                localConnections[localOffset++] = (x+1) + y*nx + z*nx*ny;
            if( y > 0 )
                localConnections[localOffset++] = x + (y-1)*nx + z*nx*ny;
            if( y < ny )
                localConnections[localOffset++] = x + (y+1)*nx + z*nx*ny;
            if( z > 0 )
                localConnections[localOffset++] = x + y*nx + (z-1)*nx*ny;
            if( z < nz )
                localConnections[localOffset++] = x + y*nx + (z+1)*nx*ny;
        }
    }
    else
    {
        //
        // Cut the x dimension
        //

        // Add the left and/or right sides
        const int commSize = 1u<<log2CommSize;
        const int xLeftSize = (xSize-1) / 2;
        if( log2CommSize == 0 )
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xLeftSize, xSize, zSize, xOffset, yOffset, zOffset,
              cutoff, commRank/2, log2CommSize-1, 
              localConnections, localOffset );
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize, 
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff, 
              commRank/2, log2CommSize-1, localConnections, localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize,
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff,
              0, 0, localConnections, localOffset );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xLeftSize, ySize, zSize, xOffset, yOffset, zOffset,
              cutoff, 0, 0, localConnections, localOffset );
        }
        
        // Add our local portion of the partition
        const int localHeight = 
            elemental::LocalLength( ySize*zSize, commRank, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = commRank + iLocal*commSize;
            const int yDelta = i % ySize;
            const int zDelta = i / ySize;
            const int x = xOffset + xLeftSize;
            const int y = yOffset + yDelta;
            const int z = zOffset + zDelta;

            localConnections[localOffset++] = x+y*nx+z*nx*ny;
            if( x > 0 )
                localConnections[localOffset++] = (x-1) + y*nx + z*nx*ny;
            if( x < nx )
                localConnections[localOffset++] = (x+1) + y*nx + z*nx*ny;
            if( y > 0 )
                localConnections[localOffset++] = x + (y-1)*nx + z*nx*ny;
            if( y < ny )
                localConnections[localOffset++] = x + (y+1)*nx + z*nx*ny;
            if( z > 0 )
                localConnections[localOffset++] = x + y*nx + (z-1)*nx*ny;
            if( z < nz )
                localConnections[localOffset++] = x + y*nx + (z+1)*nx*ny;
        }
    }
}

template<typename F>
int
psp::DistHelmholtz<F>::OwningProcess
( int x, int y, int zLocal, int xSize, int ySize, unsigned depthToSerial )
{
    int process = 0;
    OwningProcessRecursion
    ( x, y, zLocal, xSize, ySize, depthToSerial, process );
    return process;
}

template<typename F>
void
psp::DistHelmholtz<F>::OwningProcessRecursion
( int x, int y, int zLocal, int xSize, int ySize, unsigned depthToSerial, 
  int& process )
{
    if( depthToSerial == 0 )
        return;

    if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        const int xLeftSize = (xSize-1) / 2;
        if( x == xLeftSize )
        {
            process <<= depthToSerial;
            process |= (y+zLocal*ySize) % (1u<<depthToSerial);
        }
        else if( x > xLeftSize )
        { 
            // Continue down the right side
            process <<= 1;
            process |= 1;
            OwningProcessRecursion
            ( x-(xLeftSize+1), y, zLocal, xSize-(xLeftSize+1), ySize, 
              depthToSerial-1, process );
        }
        else // x < leftSize
        {
            // Continue down the left side
            process <<= 1;
            OwningProcessRecursion
            ( x, y, zLocal, xLeftSize, ySize, depthToSerial-1, process );
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //

        const int yLeftSize = (ySize-1) / 2;
        if( y == yLeftSize )
        {
            process <<= depthToSerial;
            process |= (x+zLocal*xSize) % (1u<<depthToSerial);
        }
        else if( y > yLeftSize )
        { 
            // Continue down the right side
            process <<= 1;
            process |= 1;
            OwningProcessRecursion
            ( x, y-(yLeftSize+1), zLocal, xSize, ySize-(yLeftSize+1), 
              depthToSerial-1, process );
        }
        else // x < leftSize
        {
            // Continue down the left side
            process <<= 1;
            OwningProcessRecursion
            ( x, y, zLocal, xSize, yLeftSize, depthToSerial-1, process );
        }
    }
}

template<typename F>
int
psp::DistHelmholtz<F>::ReorderedIndex
( int x, int y, int z, int nx, int ny, int nz, int log2CommSize, int cutoff )
{
    int index = 
        ReorderedIndexRecursion( x, y, z, nx, ny, nz, log2CommSize, cutoff, 0 );
    return index;
}

template<typename F>
int
psp::DistHelmholtz<F>::ReorderedIndexRecursion
( int x, int y, int z, int nx, int ny, int nz,
  int log2CommSize, int cutoff, int offset )
{
    const int nextLog2CommSize = std::max(log2CommSize-1,0);
    if( log2CommSize == 0 && nx*ny <= cutoff )
    {
        // We have satisfied the nested dissection constraints
        return offset + (x+y*nx+z*nx*ny);
    }
    else if( nx >= ny )
    {
        // Partition the X dimension
        const int middle = (nx-1)/2;
        if( x < middle )
        {
            return ReorderedIndexRecursion
            ( x, y, z, middle, ny, nz, nextLog2CommSize, cutoff, offset );
        }
        else if( x == middle )
        {
            return offset + std::max(nx-1,0)*ny*nz + (y+z*ny);
        }
        else // x > middle
        {
            return ReorderedIndexRecursion
            ( x-middle-1, y, z, std::max(nx-middle-1,0), ny, nz,
              nextLog2CommSize, cutoff, offset+middle*ny*nz );
        }
    }
    else
    {
        // Partition the Y dimension
        const int middle = (ny-1)/2;
        if( y < middle )
        {
            return ReorderedIndexRecursion
            ( x, y, z, nx, middle, nz, nextLog2CommSize, cutoff, offset );
        }
        else if( y == middle )
        {
            return offset + nx*std::max(ny-1,0)*nz + (x+z*nx);
        }
        else // y > middle 
        {
            return ReorderedIndexRecursion
            ( x, y-middle-1, z, nx, std::max(ny-middle-1,0), nz,
              nextLog2CommSize, cutoff, offset+nx*middle*nz );
        }
    }
}

template<typename F>
void
psp::DistHelmholtz<F>::FillOrigPanelStruct
( int nx, int ny, int nz, int cutoff, clique::mpi::Comm comm, 
  unsigned log2CommSize, clique::symbolic::SymmOrig& S )
{
    int nxSub=nx, nySub=ny, xOffset=0, yOffset=0;    
    FillDistOrigPanelStruct
    ( nx, ny, nz, nxSub, nySub, xOffset, yOffset, cutoff, comm, log2CommSize, 
      S );
    FillLocalOrigPanelStruct
    ( nx, ny, nz, nxSub, nySub, xOffset, yOffset, cutoff, log2CommSize, S );
}

template<typename F>
void
psp::DistHelmholtz<F>::FillDistOrigPanelStruct
( int nx, int ny, int nz, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
  int cutoff, clique::mpi::Comm comm, unsigned log2CommSize, 
  clique::symbolic::SymmOrig& S )
{
    const unsigned commRank = clique::mpi::CommRank( comm );
    S.dist.comm = comm;
    // Fill the distributed nodes
    for( int s=log2CommSize; s>0; --s )
    {
        clique::symbolic::DistSymmOrigSupernode& sn = S.dist.supernodes[s];
        const int powerOfTwo = 1u<<(s-1);
        const bool onLeft = (commRank&powerOfTwo) == 0;
        if( nxSub >= nySub )
        {
            // Form the structure of a partition of the X dimension
            const int middle = (nxSub-1)/2;
            sn.size = nySub*nz;
            sn.offset = 
                ReorderedIndex
                ( xOffset+middle, yOffset, 0, nx, ny, nz, 
                  log2CommSize, cutoff );

            // Allocate space for the lower structure
            int numJoins = 0;
            if( yOffset-1 >= 0 )
                ++numJoins;
            if( yOffset+nySub < ny )
                ++numJoins;
            sn.lowerStruct.resize( numJoins*nz );

            // Fill the (unsorted) lower structure
            int joinOffset = 0;
            if( yOffset-1 >= 0 )
            {
                for( int i=0; i<nz; ++i )
                    sn.lowerStruct[i] = ReorderedIndex
                    ( xOffset+middle, yOffset-1, i, nx, ny, nz,
                      log2CommSize, cutoff );
                joinOffset += nz;
            }
            if( yOffset+nySub < ny )
            {
                for( int i=0; i<nz; ++i )
                    sn.lowerStruct[joinOffset+i] = ReorderedIndex
                    ( xOffset+middle, yOffset+nySub, i, nx, ny, nz,
                      log2CommSize, cutoff );
            }

            // Sort the lower structure
            std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );

            // Pick the new offsets and sizes based upon our rank
            if( onLeft )
            {
                xOffset = xOffset;
                nxSub = middle;
            }
            else
            {
                xOffset = xOffset+middle+1;
                nxSub = std::max(nxSub-middle-1,0);
            }
        }
        else
        {
            // Form the structure of a partition of the Y dimension
            const int middle = (nySub-1)/2;
            sn.size = nxSub*nz;
            sn.offset =
                ReorderedIndex
                ( xOffset, yOffset+middle, 0, nx, ny, nz,
                  log2CommSize, cutoff );

            // Allocate space for the lower structure
            int numJoins = 0;
            if( xOffset-1 >= 0 )
                ++numJoins;
            if( xOffset+nxSub < nx )
                ++numJoins;
            sn.lowerStruct.resize( numJoins*nz );

            // Fill the (unsorted) lower structure
            int joinOffset = 0;
            if( xOffset-1 >= 0 )
            {
                for( int i=0; i<nz; ++i )
                    sn.lowerStruct[i] = ReorderedIndex
                    ( xOffset-1, yOffset+middle, i, nx, ny, nz,
                      log2CommSize, cutoff );
                joinOffset += nz;
            }
            if( xOffset+nxSub < nx )
            {
                for( int i=0; i<nz; ++i )
                    sn.lowerStruct[joinOffset+i] = ReorderedIndex
                    ( xOffset+nxSub, yOffset+middle, i, nx, ny, nz,
                      log2CommSize, cutoff );
            }

            // Sort the lower structure
            std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );

            // Pick the new offsets and sizes based upon our rank
            if( onLeft )
            {
                yOffset = yOffset;
                nySub = middle;
            }
            else
            {
                yOffset = yOffset+middle+1;
                nySub = std::max(nySub-middle-1,0);
            }
        }
    }

    // Fill the bottom node, which is only owned by a single process
    clique::symbolic::DistSymmOrigSupernode& sn = S.dist.supernodes[0];
    if( nxSub*nySub <= cutoff )
    {
        sn.size = nxSub*nySub*nz;
        sn.offset = ReorderedIndex
            ( xOffset, yOffset, 0, nx, ny, nz, log2CommSize, cutoff );

        // Count, allocate, and fill the lower struct
        int joinSize = 0;
        if( xOffset-1 >= 0 )
            joinSize += nySub*nz;
        if( xOffset+nxSub < nx )
            joinSize += nySub*nz;
        if( yOffset-1 >= 0 )
            joinSize += nxSub*nz;
        if( yOffset+nySub < ny )
            joinSize += nxSub*nz;
        sn.lowerStruct.resize( joinSize );

        int joinOffset = 0;
        if( xOffset-1 >= 0 )
        {
            for( int i=0; i<nz; ++i )
                for( int j=0; j<nySub; ++j )
                    sn.lowerStruct[i*nySub+j] = ReorderedIndex
                    ( xOffset-1, yOffset+j, i,
                      nx, ny, nz, log2CommSize, cutoff );
            joinOffset += nySub*nz;
        }
        if( xOffset+nxSub < nx )
        {
            for( int i=0; i<nz; ++i )
                for( int j=0; j<nySub; ++j )
                    sn.lowerStruct[joinOffset+i*nySub+j] = ReorderedIndex
                    ( xOffset+nxSub, yOffset+j, i,
                      nx, ny, nz, log2CommSize, cutoff );
            joinOffset += nySub*nz;
        }
        if( yOffset-1 >= 0 )
        {
            for( int i=0; i<nz; ++i )
                for( int j=0; j<nxSub; ++j )
                    sn.lowerStruct[joinOffset+i*nxSub+j] = ReorderedIndex
                    ( xOffset+j, yOffset-1, i,
                      nx, ny, nz, log2CommSize, cutoff );
            joinOffset += nxSub*nz;
        }
        if( yOffset+nySub < ny )
        {
            for( int i=0; i<nz; ++i )
                for( int j=0; j<nxSub; ++j )
                    sn.lowerStruct[joinOffset+i*nxSub+j] = ReorderedIndex
                    ( xOffset+j, yOffset+nySub, i,
                      nx, ny, nz, log2CommSize, cutoff );
        }

        // Sort the lower structure
        std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
    }
    else if( nxSub >= nySub )
    {
        // Form the structure of a partition of the X dimension
        const int middle = (nxSub-1)/2;
        sn.size = nySub*nz;
        sn.offset =
            ReorderedIndex
            ( xOffset+middle, yOffset, 0, nx, ny, nz, log2CommSize, cutoff );

        // Allocate space for the lower structure
        int numJoins = 0;
        if( yOffset-1 >= 0 )
            ++numJoins;
        if( yOffset+nySub < ny )
            ++numJoins;
        sn.lowerStruct.resize( numJoins*nz );

        // Fill the (unsorted) lower structure
        int joinOffset = 0;
        if( yOffset-1 >= 0 )
        {
            for( int i=0; i<nz; ++i )
                sn.lowerStruct[i] = ReorderedIndex
                ( xOffset+middle, yOffset-1, i, nx, ny, nz,
                  log2CommSize, cutoff );
            joinOffset += nz;
        }
        if( yOffset+nySub < ny )
        {
            for( int i=0; i<nz; ++i )
                sn.lowerStruct[joinOffset+i] = ReorderedIndex
                ( xOffset+middle, yOffset+nySub, i, nx, ny, nz,
                  log2CommSize, cutoff );
        }

        // Sort the lower structure
        std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
    }
    else
    {
        // Form the structure of a partition of the Y dimension
        const int middle = (nySub-1)/2;
        sn.size = nxSub*nz;
        sn.offset =
            ReorderedIndex
            ( xOffset, yOffset+middle, 0, nx, ny, nz, log2CommSize, cutoff );

        // Allocate space for the lower structure
        int numJoins = 0;
        if( xOffset-1 >= 0 )
            ++numJoins;
        if( xOffset+nxSub < nx )
            ++numJoins;
        sn.lowerStruct.resize( numJoins*nz );

        // Fill the (unsorted) lower structure
        int joinOffset = 0;
        if( xOffset-1 >= 0 )
        {
            for( int i=0; i<nz; ++i )
                sn.lowerStruct[i] = ReorderedIndex
                ( xOffset-1, yOffset+middle, i, nx, ny, nz,
                  log2CommSize, cutoff );
            joinOffset += nz;
        }
        if( xOffset+nxSub < nx )
        {
            for( int i=0; i<nz; ++i )
                sn.lowerStruct[joinOffset+i] = ReorderedIndex
                ( xOffset+nxSub, yOffset+middle, i, nx, ny, nz,
                  log2CommSize, cutoff );
        }

        // Sort the lower structure
        std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
    }
}

template<typename F>
void
psp::DistHelmholtz<F>::FillLocalOrigPanelStruct
( int nx, int ny, int nz, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
  int cutoff, unsigned log2CommSize, clique::symbolic::SymmOrig& S )
{
    const int numLocalSupernodes = S.local.supernodes.size();

    // Initialize with the local root's box
    std::stack<Box> boxStack;
    {
        Box box;
        box.parentIndex = -1;
        box.nx = nxSub;
        box.ny = nySub;
        box.xOffset = xOffset;
        box.yOffset = yOffset;
        box.leftChild = false;
        boxStack.push(box);
    }

    // Fill the local tree
    for( int s=numLocalSupernodes-1; s>=0; --s )
    {
        Box box = boxStack.top();
        boxStack.pop();

        clique::symbolic::LocalSymmOrigSupernode& sn = S.local.supernodes[s];
        sn.parent = box.parentIndex;
        if( sn.parent != -1 )
        {
            if( box.leftChild )
                S.local.supernodes[sn.parent].children[0] = s;
            else
                S.local.supernodes[sn.parent].children[1] = s;
        }

        if( box.nx*box.ny*nz <= cutoff )
        {
            sn.size = box.nx*box.ny*nz;
            sn.offset = ReorderedIndex
                ( box.xOffset, box.yOffset, 0, nx, ny, nz,
                  log2CommSize, cutoff );
            sn.children.clear();

            // Count, allocate, and fill the lower struct
            int joinSize = 0;
            if( box.xOffset-1 >= 0 )
                joinSize += box.ny*nz;
            if( box.xOffset+box.nx < nx )
                joinSize += box.ny*nz;
            if( box.yOffset-1 >= 0 )
                joinSize += box.nx*nz;
            if( box.yOffset+box.ny < ny )
                joinSize += box.nx*nz;
            sn.lowerStruct.resize( joinSize );

            int joinOffset = 0;
            if( box.xOffset-1 >= 0 )
            {
                for( int i=0; i<nz; ++i )
                    for( int j=0; j<box.ny; ++j )
                        sn.lowerStruct[i*box.ny+j] = ReorderedIndex
                        ( box.xOffset-1, box.yOffset+j, i,
                          nx, ny, nz, log2CommSize, cutoff );
                joinOffset += box.ny*nz;
            }
            if( box.xOffset+box.nx < nx )
            {
                for( int i=0; i<nz; ++i )
                    for( int j=0; j<box.ny; ++j )
                        sn.lowerStruct[joinOffset+i*box.ny+j] = ReorderedIndex
                        ( box.xOffset+box.nx, box.yOffset+j, i,
                          nx, ny, nz, log2CommSize, cutoff );
                joinOffset += box.ny*nz;
            }
            if( box.yOffset-1 >= 0 )
            {
                for( int i=0; i<nz; ++i )
                    for( int j=0; j<box.nx; ++j )
                        sn.lowerStruct[joinOffset+i*box.nx+j] = ReorderedIndex
                        ( box.xOffset+j, box.yOffset-1, i,
                          nx, ny, nz, log2CommSize, cutoff );
                joinOffset += box.nx*nz;
            }
            if( box.yOffset+box.ny < ny )
            {
                for( int i=0; i<nz; ++i )
                    for( int j=0; j<box.nx; ++j )
                        sn.lowerStruct[joinOffset+i*box.nx+j] = ReorderedIndex
                        ( box.xOffset+j, box.yOffset+box.ny, i,
                          nx, ny, nz, log2CommSize, cutoff );
            }


            // Sort the lower structure
            std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
        }
        else
        {
            sn.children.resize(2);
            if( box.nx >= box.ny )
            {
                // Partition the X dimension (this is the separator)
                const int middle = (box.nx-1)/2;
                sn.size = box.ny*nz;
                sn.offset = ReorderedIndex
                    ( box.xOffset+middle, box.yOffset, 0, nx, ny, nz,
                      log2CommSize, cutoff );

                // Count, allocate, and fill the lower struct
                int numJoins = 0;
                if( box.yOffset-1 >= 0 )
                    ++numJoins;
                if( box.yOffset+box.ny < ny )
                    ++numJoins;
                sn.lowerStruct.resize( numJoins*nz );

                int joinOffset = 0;
                if( box.yOffset-1 >= 0 )
                {
                    for( int i=0; i<nz; ++i )
                        sn.lowerStruct[i] = ReorderedIndex
                        ( box.xOffset+middle, box.yOffset-1, i, nx, ny, nz,
                          log2CommSize, cutoff );
                    joinOffset += nz;
                }
                if( box.yOffset+box.ny < ny )
                {
                    for( int i=0; i<nz; ++i )
                        sn.lowerStruct[joinOffset+i] = ReorderedIndex
                        ( box.xOffset+middle, box.yOffset+box.ny, i, nx, ny, nz,
                          log2CommSize, cutoff );
                }

                // Sort the lower structure
                std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );

                // Push the left child box onto the stack
                Box leftBox;
                leftBox.parentIndex = s;
                leftBox.nx = middle;
                leftBox.ny = box.ny;
                leftBox.xOffset = box.xOffset;
                leftBox.yOffset = box.yOffset;
                leftBox.leftChild = true;
                boxStack.push( leftBox );

                // Push the right child box onto the stack
                Box rightBox;
                rightBox.parentIndex = s;
                rightBox.nx = std::max(box.nx-middle-1,0);
                rightBox.ny = box.ny;
                rightBox.xOffset = box.xOffset+middle+1;
                rightBox.yOffset = box.yOffset;
                rightBox.leftChild = false;
                boxStack.push( rightBox );
            }
            else
            {
                // Partition the Y dimension (this is the separator)
                const int middle = (box.ny-1)/2;
                sn.size = box.nx*nz;
                sn.offset = ReorderedIndex
                    ( box.xOffset, box.yOffset+middle, 0, nx, ny, nz,
                      log2CommSize, cutoff );


                // Count, allocate, and fill the lower struct
                int numJoins = 0;
                if( box.xOffset-1 >= 0 )
                    ++numJoins;
                if( box.xOffset+box.nx < nx )
                    ++numJoins;
                sn.lowerStruct.resize( numJoins*nz );

                int joinOffset = 0;
                if( box.xOffset-1 >= 0 )
                {
                    for( int i=0; i<nz; ++i )
                        sn.lowerStruct[i] = ReorderedIndex
                        ( box.xOffset-1, box.yOffset+middle, i, nx, ny, nz,
                          log2CommSize, cutoff );
                    joinOffset += nz;
                }
                if( box.xOffset+box.nx < nx )
                {
                    for( int i=0; i<nz; ++i )
                        sn.lowerStruct[joinOffset+i] = ReorderedIndex
                        ( box.xOffset+box.nx, box.yOffset+middle, i, nx, ny, nz,
                          log2CommSize, cutoff );
                }

                // Sort the lower structure
                std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );

                // Push the left child box onto the stack
                Box leftBox;
                leftBox.parentIndex = s;
                leftBox.nx = box.nx;
                leftBox.ny = middle;
                leftBox.xOffset = box.xOffset;
                leftBox.yOffset = box.yOffset;
                leftBox.leftChild = true;
                boxStack.push( leftBox );

                // Push the right child box onto the stack
                Box rightBox;
                rightBox.parentIndex = s;
                rightBox.nx = box.nx;
                rightBox.ny = std::max(box.ny-middle-1,0);
                rightBox.xOffset = box.xOffset;
                rightBox.yOffset = box.yOffset+middle+1;
                rightBox.leftChild = false;
                boxStack.push( rightBox );
            }
        }
    }
}

template class psp::DistHelmholtz<float>;
template class psp::DistHelmholtz<double>;
template class psp::DistHelmholtz<std::complex<float> >;
template class psp::DistHelmholtz<std::complex<double> >;
