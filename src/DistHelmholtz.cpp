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

    const int cutoff = 100;
    // Create the 2d reordering structure
    /*
    const int planeSize = nx*ny;
    std::vector<int> reordering( planeSize );
    RecursiveReordering( nx, 0, nx, 0, ny, cutoff, 0, &reordering[0] );

    // Construct the inverse map
    std::vector<int> inverseReordering( planeSize );
    for( int i=0; i<planeSize; ++i )
        inverseReordering[reordering[i]] = i;
    */

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
    const int numTotalPanels = 1 + numFullInnerPanels + haveLeftover + 1;

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

    // Compute the indices of the RHS that each process will need to send/recv 
    // from every other process in order to perform its portion of a mat-vec
    //
    //localEntries_.resize( localRowOffsets_.back() );
    // TODO

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

    // Fill the original structures
    //
    // In order to minimize the number of symbolic factorizations that have to 
    // be performed, and to simplify distribution issues, the leading PML region
    // on each inner panel will always be ordered _LAST_ within that panel.
    //
    // TODO

    // Perform the parallel symbolic factorizations
    clique::symbolic::SymmetricFactorization
    ( topSymbolicOrig, topSymbolicFact_, true );
    clique::symbolic::SymmetricFactorization
    ( fullInnerSymbolicOrig, fullInnerSymbolicFact_, true );
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
( int z, int topDepth, int innerDepth, int bottomOrigDepth, 
  int bzPadded, int planesPerPanel )
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
                    int numForwardConnections = 1; // always count diagonal
                    if( x < nx ) 
                        ++numForwardConnections;
                    if( y < ny )
                        ++numForwardConnections;
                    if( z < nz )
                        ++numForwardConnections;
                    localRowOffsets[localOffset+1] = 
                        localRowOffsets[localOffset] + numForwardConnections;

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
            int numForwardConnections = 1; // always count diagonal
            if( x < nx )
                ++numForwardConnections;
            if( y < ny )
                ++numForwardConnections;
            if( z < nz )
                ++numForwardConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numForwardConnections;

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
            int numForwardConnections = 1; // always count diagonal
            if( x < nx )
                ++numForwardConnections;
            if( y < ny )
                ++numForwardConnections;
            if( z < nz )
                ++numForwardConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numForwardConnections;

            ++localOffset;
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

template class psp::DistHelmholtz<float>;
template class psp::DistHelmholtz<double>;
template class psp::DistHelmholtz<std::complex<float> >;
template class psp::DistHelmholtz<std::complex<double> >;
