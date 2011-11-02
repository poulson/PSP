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

#include "./DistHelmholtz/InitializeFinalize-incl.hpp"
#include "./DistHelmholtz/Multiply-incl.hpp"
#include "./DistHelmholtz/Precondition-incl.hpp"

template<typename R>
psp::DistHelmholtz<R>::DistHelmholtz
( const FiniteDiffControl<R>& control, elemental::mpi::Comm comm )
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
    const int cutoff = control.cutoff;
    const int numPlanesPerPanel = control.numPlanesPerPanel;
    const int bzCeil = bzCeil_;

    // Pull out some information about our communicator
    const int commRank = elemental::mpi::CommRank( comm );
    const int commSize = elemental::mpi::CommSize( comm );
    unsigned temp = commSize;
    log2CommSize_ = 0;
    while( temp >>= 1 )
        ++log2CommSize_;

    // Decide if the domain is sufficiently deep to warrant sweeping
    const bool bottomHasPML = (control.bottomBC == PML);
    topDepth_ = bzCeil+numPlanesPerPanel;
    bottomOrigDepth_ = 
        (bottomHasPML ? bzCeil+numPlanesPerPanel : numPlanesPerPanel );
    if( nz <= topDepth_+bottomOrigDepth_ )
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
    innerDepth_ = nz-(topDepth_+bottomOrigDepth_);
    leftoverInnerDepth_ = innerDepth_ % numPlanesPerPanel;
    haveLeftover_ = ( leftoverInnerDepth_ != 0 );
    numFullInnerPanels_ = innerDepth_ / numPlanesPerPanel;

    // Compute the number of rows we own of the sparse distributed matrix
    localTopHeight_ = LocalPanelHeight( topDepth_, 0, commRank );
    localFullInnerHeight_ = 
        LocalPanelHeight( numPlanesPerPanel, bzCeil_, commRank );
    localLeftoverInnerHeight_ = 
        LocalPanelHeight( leftoverInnerDepth_, bzCeil_, commRank );
    localBottomHeight_ = 
        LocalPanelHeight( bottomOrigDepth_, bzCeil_, commRank );
    localHeight_ = localTopHeight_ + numFullInnerPanels_*localFullInnerHeight_ +
                   localLeftoverInnerHeight_ + localBottomHeight_;

    // Compute the natural indices of our local indices and fill in the offsets
    // for the store of each local rows indices.
    localToNaturalMap_.resize( localHeight_ );
    localRowOffsets_.resize( localHeight_+1 );
    localRowOffsets_[0] = 0;
    int localOffset=0, zOffset=0;
    MapLocalPanelIndices( topDepth_, 0, zOffset, commRank, localOffset );
    for( int i=0; i<numFullInnerPanels_; ++i )
        MapLocalPanelIndices
        ( numPlanesPerPanel, bzCeil_, zOffset, commRank, localOffset );
    if( haveLeftover_ )
        MapLocalPanelIndices
        ( leftoverInnerDepth_, bzCeil_, zOffset, commRank, localOffset );
    MapLocalPanelIndices
    ( bottomOrigDepth_, bzCeil_, zOffset, commRank, localOffset );

    // Fill in the natural connection indices
    std::vector<int> localConnections( localRowOffsets_.back() );
    localOffset = 0;
    zOffset = 0;
    MapLocalConnectionIndices
    ( topDepth_, 0, zOffset, commRank, localConnections, localOffset );
    for( int i=0; i<numFullInnerPanels_; ++i )
        MapLocalConnectionIndices
        ( numPlanesPerPanel, bzCeil_, zOffset, commRank, localConnections, 
          localOffset );
    if( haveLeftover_ )
        MapLocalConnectionIndices
        ( leftoverInnerDepth_, bzCeil_, zOffset, commRank, localConnections, 
          localOffset );
    MapLocalConnectionIndices
    ( bottomOrigDepth_, bzCeil_, zOffset, commRank, localConnections, 
      localOffset );
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
            const int zLocal = LocalZ( z );
            const int process = OwningProcess( x, y, zLocal );
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
            const int zLocal = LocalZ( z );
            const int process = OwningProcess( x, y, zLocal );

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
    const int numLocalSupernodes = NumLocalSupernodes( commRank );

    // Create space for the original structures of the panel classes
    clique::symbolic::SymmOrig 
        topSymbolicOrig, 
        fullInnerSymbolicOrig, 
        leftoverInnerSymbolicOrig, 
        bottomSymbolicOrig;
    topSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    topSymbolicOrig.dist.supernodes.resize( log2CommSize_ );
    fullInnerSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    fullInnerSymbolicOrig.dist.supernodes.resize( log2CommSize_ );
    leftoverInnerSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    leftoverInnerSymbolicOrig.dist.supernodes.resize( log2CommSize_ );
    bottomSymbolicOrig.local.supernodes.resize( numLocalSupernodes );
    bottomSymbolicOrig.dist.supernodes.resize( log2CommSize_ );

    // Fill the original structures (in the nested-dissection ordering)
    //
    // In order to minimize the number of symbolic factorizations that have to 
    // be performed, and to simplify distribution issues, the leading PML region
    // on each inner panel will always be ordered _LAST_ within that panel.
    FillOrigPanelStruct( topDepth_, topSymbolicOrig );
    FillOrigPanelStruct( numPlanesPerPanel+bzCeil, fullInnerSymbolicOrig );
    if( haveLeftover_ )
        FillOrigPanelStruct
        ( leftoverInnerDepth_+bzCeil, leftoverInnerSymbolicOrig );
    FillOrigPanelStruct( bottomOrigDepth_+bzCeil, bottomSymbolicOrig );

    // Perform the parallel symbolic factorizations
    clique::symbolic::SymmetricFactorization
    ( topSymbolicOrig, topSymbolicFact_, true );
    clique::symbolic::SymmetricFactorization
    ( fullInnerSymbolicOrig, fullInnerSymbolicFact_, true );
    if( haveLeftover_ )
        clique::symbolic::SymmetricFactorization
        ( leftoverInnerSymbolicOrig, leftoverInnerSymbolicFact_, true );
    clique::symbolic::SymmetricFactorization
    ( bottomSymbolicOrig, bottomSymbolicFact_, true );
}

template<typename R>
int
psp::DistHelmholtz<R>::LocalPanelHeight
( int zSize, int zPadding, unsigned commRank ) const
{
    int localHeight = 0;
    LocalPanelHeightRecursion
    ( control_.nx, control_.ny, zSize, zPadding, control_.cutoff, 
      commRank, log2CommSize_, localHeight );
    return localHeight;
}

template<typename R>
void
psp::DistHelmholtz<R>::LocalPanelHeightRecursion
( int xSize, int ySize, int zSize, int zPadding, int cutoff, 
  unsigned commRank, unsigned depthTilSerial, int& localHeight ) 
{
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
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
        const int commSize = 1u<<depthTilSerial;
        const int alignment = (ySize*zPadding) % commSize;
        localHeight += 
            elemental::LocalLength( ySize*zSize, alignment, commSize );

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( depthTilSerial == 0 )
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xLeftSize, ySize, zSize, zPadding, cutoff, 0, 0, localHeight );
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize-(xLeftSize+1), ySize, zSize, zPadding, cutoff, 0, 0, 
              localHeight );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize-(xLeftSize+1), ySize, zSize, zPadding, cutoff,
              commRank/2, depthTilSerial-1, localHeight );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xLeftSize, ySize, zSize, zPadding, cutoff,
              commRank/2, depthTilSerial-1, localHeight );
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //

        // Add our local portion of the partition
        const int commSize = 1u<<depthTilSerial;
        const int alignment = (xSize*zPadding) % commSize;
        localHeight +=
            elemental::LocalLength( xSize*zSize, alignment, commSize );

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( depthTilSerial == 0 )
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xSize, yLeftSize, zSize, zPadding, cutoff, 0, 0, localHeight );
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize, ySize-(yLeftSize+1), zSize, zPadding, cutoff, 0, 0, 
              localHeight );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            LocalPanelHeightRecursion
            ( xSize, ySize-(yLeftSize+1), zSize, zPadding, cutoff, 
              commRank/2, depthTilSerial-1, localHeight );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            LocalPanelHeightRecursion
            ( xSize, yLeftSize, zSize, zPadding, cutoff,
              commRank/2, depthTilSerial-1, localHeight );
        }
    }
}

template<typename R>
int
psp::DistHelmholtz<R>::NumLocalSupernodes( unsigned commRank ) const
{
    int numLocalSupernodes = 0;
    NumLocalSupernodesRecursion
    ( control_.nx, control_.ny, control_.cutoff, commRank, log2CommSize_, 
      numLocalSupernodes );
    return numLocalSupernodes;
}

template<typename R>
void
psp::DistHelmholtz<R>::NumLocalSupernodesRecursion
( int xSize, int ySize, int cutoff, unsigned commRank, unsigned depthTilSerial, 
  int& numLocal ) 
{
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
    {
        ++numLocal;
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        // Add our local portion of the partition
        if( depthTilSerial == 0 )
            ++numLocal;

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( depthTilSerial  == 0 )
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
            ( xSize-(xLeftSize+1), ySize, cutoff, commRank/2, depthTilSerial-1, 
              numLocal );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            NumLocalSupernodesRecursion
            ( xLeftSize, ySize, cutoff, commRank/2, depthTilSerial-1, 
              numLocal );
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //

        // Add our local portion of the partition
        if( depthTilSerial == 0 )
            ++numLocal;

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( depthTilSerial == 0 )
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
            ( xSize, ySize-(yLeftSize+1), cutoff, commRank/2, depthTilSerial-1, 
              numLocal );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            NumLocalSupernodesRecursion
            ( xSize, yLeftSize, cutoff, commRank/2, depthTilSerial-1, 
              numLocal );
        }
    }
}

template<typename R>
int 
psp::DistHelmholtz<R>::LocalZ( int z ) const
{
    if( z < topDepth_ )
    {
        return z;
    }
    else if( z < topDepth_ + innerDepth_ )
    {
        return ((z-topDepth_) % control_.numPlanesPerPanel) + bzCeil_;
    }
    else // z in [topDepth+innerDepth,topDepth+innerDepth+bottomOrigDepth)
    {
#ifndef RELEASE
        if( z < topDepth_+innerDepth_ || 
            z >= topDepth_+innerDepth_+bottomOrigDepth_ )
            throw std::logic_error("z is out of bounds");
#endif
        return (z - (topDepth_+innerDepth_)) + bzCeil_;
    }
}

// Return the local offset into the global sparse matrix for the panel 
// containing the given z index
template<typename R>
int 
psp::DistHelmholtz<R>::LocalPanelOffset( int z ) const
{
    if( z < topDepth_ )
    {
        return 0;
    }
    else if( z < topDepth_ + innerDepth_ )
    {
        return localTopHeight_ + 
               ((z-topDepth_)/control_.numPlanesPerPanel)*localFullInnerHeight_;
    }
    else
    {
        return localTopHeight_ + numFullInnerPanels_*localFullInnerHeight_ + 
               localLeftoverInnerHeight_;
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::MapLocalPanelIndices
( int zSize, int zPadding, int& zOffset, unsigned commRank, int& localOffset ) 
{
    MapLocalPanelIndicesRecursion
    ( control_.nx, control_.ny, control_.nz, control_.nx, control_.ny, zSize, 
      zPadding, 0, 0, zOffset, control_.cutoff, commRank, log2CommSize_, 
      localToNaturalMap_, localRowOffsets_, localOffset );
    zOffset += zSize;
}

template<typename R>
void
psp::DistHelmholtz<R>::MapLocalPanelIndicesRecursion
( int nx, int ny, int nz, int xSize, int ySize, int zSize, int zPadding,
  int xOffset, int yOffset, int zOffset, int cutoff, 
  unsigned commRank, unsigned depthTilSerial,
  std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
  int& localOffset )
{
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
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
                    if( x+1 < nx ) 
                        ++numConnections;
                    if( y > 0 )
                        ++numConnections;
                    if( y+1 < ny )
                        ++numConnections;
                    if( z > 0 )
                        ++numConnections;
                    if( z+1 < nz )
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
        const int commSize = 1u<<depthTilSerial;
        const int yLeftSize = (ySize-1) / 2;
        if( depthTilSerial == 0 )
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, 0, 0, localToNaturalMap, 
              localRowOffsets, localOffset );
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize, zPadding,
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff, 
              0, 0, localToNaturalMap, localRowOffsets, localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize, zPadding,
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff,
              commRank/2, depthTilSerial-1, localToNaturalMap, 
              localRowOffsets, localOffset );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, commRank/2, depthTilSerial-1, 
              localToNaturalMap, localRowOffsets, localOffset );
        }
        
        // Add our local portion of the partition
        const int alignment = (xSize*zPadding) % commSize;
        const int localHeight = 
            elemental::LocalLength( xSize*zSize, alignment, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = alignment + iLocal*commSize;
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
            if( x+1 < nx )
                ++numConnections;
            if( y > 0 )
                ++numConnections;
            if( y+1 < ny )
                ++numConnections;
            if( z > 0 )
                ++numConnections;
            if( z+1 < nz )
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
        const int commSize = 1u<<depthTilSerial;
        const int xLeftSize = (xSize-1) / 2;
        if( depthTilSerial == 0 )
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xLeftSize, xSize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, 0, 0, localToNaturalMap, 
              localRowOffsets, localOffset );
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize, zPadding,
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff, 
              0, 0, localToNaturalMap, localRowOffsets, localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize, zPadding,
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff,
              commRank/2, depthTilSerial-1, localToNaturalMap, localRowOffsets, 
              localOffset );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xLeftSize, ySize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, commRank/2, depthTilSerial-1, 
              localToNaturalMap, localRowOffsets, localOffset );
        }
        
        // Add our local portion of the partition
        const int alignment = (ySize*zPadding) % commSize;
        const int localHeight = 
            elemental::LocalLength( ySize*zSize, alignment, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = alignment + iLocal*commSize;
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
            if( x+1 < nx )
                ++numConnections;
            if( y > 0 )
                ++numConnections;
            if( y+1 < ny )
                ++numConnections;
            if( z > 0 )
                ++numConnections;
            if( z+1 < nz )
                ++numConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numConnections;

            ++localOffset;
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::MapLocalConnectionIndices
( int zSize, int zPadding, int& zOffset, unsigned commRank, 
  std::vector<int>& localConnections, int& localOffset ) const
{
    MapLocalConnectionIndicesRecursion
    ( control_.nx, control_.ny, control_.nz, control_.nx, control_.ny, zSize, 
      zPadding, 0, 0, zOffset, control_.cutoff, commRank, log2CommSize_, 
      localConnections, localOffset );
    zOffset += zSize;
}

template<typename R>
void
psp::DistHelmholtz<R>::MapLocalConnectionIndicesRecursion
( int nx, int ny, int nz, int xSize, int ySize, int zSize, int zPadding,
  int xOffset, int yOffset, int zOffset, int cutoff, 
  unsigned commRank, unsigned depthTilSerial,
  std::vector<int>& localConnections, int& localOffset )
{
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
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
                    if( x+1 < nx ) 
                        localConnections[localOffset++] = (x+1)+y*nx+z*nx*ny;
                    if( y > 0 )
                        localConnections[localOffset++] = x+(y-1)*nx+z*nx*ny;
                    if( y+1 < ny )
                        localConnections[localOffset++] = x+(y+1)*nx+z*nx*ny;
                    if( z > 0 )
                        localConnections[localOffset++] = x+y*nx+(z-1)*nx*ny;
                    if( z+1 < nz )
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
        const int commSize = 1u<<depthTilSerial;
        const int yLeftSize = (ySize-1) / 2;
        if( depthTilSerial == 0 )
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, 0, 0, localConnections, 
              localOffset );
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize, zPadding,
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff, 
              0, 0, localConnections, localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), zSize, zPadding,
              xOffset, yOffset+(yLeftSize+1), zOffset, cutoff,
              commRank/2, depthTilSerial-1, localConnections, localOffset );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, commRank/2, depthTilSerial-1, 
              localConnections, localOffset );
        }
        
        // Add our local portion of the partition
        const int alignment = (xSize*zPadding) % commSize;
        const int localHeight = 
            elemental::LocalLength( xSize*zSize, alignment, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = alignment + iLocal*commSize;
            const int xDelta = i % xSize;
            const int zDelta = i / xSize;
            const int x = xOffset + xDelta;
            const int y = yOffset + yLeftSize;
            const int z = zOffset + zDelta;

            localConnections[localOffset++] = x + y*nx + z*nx*ny;
            if( x > 0 )
                localConnections[localOffset++] = (x-1) + y*nx + z*nx*ny;
            if( x+1 < nx )
                localConnections[localOffset++] = (x+1) + y*nx + z*nx*ny;
            if( y > 0 )
                localConnections[localOffset++] = x + (y-1)*nx + z*nx*ny;
            if( y+1 < ny )
                localConnections[localOffset++] = x + (y+1)*nx + z*nx*ny;
            if( z > 0 )
                localConnections[localOffset++] = x + y*nx + (z-1)*nx*ny;
            if( z+1 < nz )
                localConnections[localOffset++] = x + y*nx + (z+1)*nx*ny;
        }
    }
    else
    {
        //
        // Cut the x dimension
        //

        // Add the left and/or right sides
        const int commSize = 1u<<depthTilSerial;
        const int xLeftSize = (xSize-1) / 2;
        if( depthTilSerial == 0 )
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xLeftSize, xSize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, 0, 0, localConnections, 
              localOffset );
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize, zPadding,
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff, 
              0, 0, localConnections, localOffset );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, zSize, zPadding,
              xOffset+(xLeftSize+1), yOffset, zOffset, cutoff,
              commRank/2, depthTilSerial-1, localConnections, localOffset );
        }
        else // depthTilSerial != 0 && commRank & 1 == 0
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xLeftSize, ySize, zSize, zPadding, 
              xOffset, yOffset, zOffset, cutoff, commRank/2, depthTilSerial-1, 
              localConnections, localOffset );
        }
        
        // Add our local portion of the partition
        const int alignment = (ySize*zPadding) % commSize;
        const int localHeight = 
            elemental::LocalLength( ySize*zSize, alignment, commSize );
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const int i = alignment + iLocal*commSize;
            const int yDelta = i % ySize;
            const int zDelta = i / ySize;
            const int x = xOffset + xLeftSize;
            const int y = yOffset + yDelta;
            const int z = zOffset + zDelta;

            localConnections[localOffset++] = x+y*nx+z*nx*ny;
            if( x > 0 )
                localConnections[localOffset++] = (x-1) + y*nx + z*nx*ny;
            if( x+1 < nx )
                localConnections[localOffset++] = (x+1) + y*nx + z*nx*ny;
            if( y > 0 )
                localConnections[localOffset++] = x + (y-1)*nx + z*nx*ny;
            if( y+1 < ny )
                localConnections[localOffset++] = x + (y+1)*nx + z*nx*ny;
            if( z > 0 )
                localConnections[localOffset++] = x + y*nx + (z-1)*nx*ny;
            if( z+1 < nz )
                localConnections[localOffset++] = x + y*nx + (z+1)*nx*ny;
        }
    }
}

template<typename R>
int
psp::DistHelmholtz<R>::OwningProcess( int x, int y, int zLocal ) const
{
    int process = 0;
    OwningProcessRecursion
    ( x, y, zLocal, control_.nx, control_.ny, log2CommSize_, process );
    return process;
}

template<typename R>
void
psp::DistHelmholtz<R>::OwningProcessRecursion
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

template<typename R>
void psp::DistHelmholtz<R>::LocalReordering
( std::map<int,int>& reordering, int zSize ) const
{
    int offset = 0;
    LocalReorderingRecursion
    ( reordering, offset, 
      0, 0, control_.nx, control_.ny, zSize, control_.nx, control_.ny, 
      log2CommSize_, control_.cutoff, mpi::CommRank(comm_) );
}

template<typename R>
void psp::DistHelmholtz<R>::LocalReorderingRecursion
( std::map<int,int>& reordering, int offset,
  int xOffset, int yOffset, int xSize, int ySize, int zSize, int nx, int ny, 
  int depthTilSerial, int cutoff, int commRank )
{
    const int nextDepthTilSerial = std::max(depthTilSerial-1,0);
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
    {
        for( int zDelta=0; zDelta<zSize; ++zDelta )
        {
            const int z = zDelta;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                for( int xDelta=0; xDelta<xSize; ++xDelta )
                {
                    const int x = xOffset + xDelta;
                    const int index = x + y*nx + z*nx*ny;
                    reordering[offset++] = index;
                }
            }
        }
    }
    else if( xSize >= ySize )
    {
        //
        // Partition the X dimension
        //
        const int middle = (xSize-1)/2;

        // Recurse on the left side
        if( depthTilSerial == 0 || !(commRank&1) )
            LocalReorderingRecursion
            ( reordering, offset, 
              xOffset, yOffset, middle, ySize, zSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += middle*ySize*zSize;

        // Recurse on the right side
        if( depthTilSerial == 0 || commRank&1 )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset+middle+1, yOffset, 
              std::max(xSize-middle-1,0), ySize, zSize, nx, ny, 
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += std::max(xSize-middle-1,0)*ySize*zSize;

        // Store the separator
        const int x = xOffset + middle;
        for( int zDelta=0; zDelta<zSize; ++zDelta )
        {
            const int z = zDelta;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                const int index = x + y*nx + z*nx*ny;
                reordering[offset++] = index;
            }
        }
    }
    else
    {
        //
        // Partition the Y dimension
        //
        const int middle = (ny-1)/2;

        // Recurse on the left side
        if( depthTilSerial == 0 || !(commRank&1) )
            LocalReorderingRecursion
            ( reordering, offset, 
              xOffset, yOffset, xSize, middle, zSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += xSize*middle*zSize;

        // Recurse on the right side
        if( depthTilSerial == 0 || commRank&1 )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset+middle+1, 
              xSize, std::max(ySize-middle-1,0), zSize, nx, ny, 
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += xSize*std::max(ySize-middle-1,0)*zSize;

        // Store the separator
        const int y = yOffset + middle;
        for( int zDelta=0; zDelta<zSize; ++zDelta )
        {
            const int z = zDelta;
            for( int xDelta=0; xDelta<xSize; ++xDelta )
            {
                const int x = xOffset + xDelta;
                const int index = x + y*nx + z*nx*ny;
                reordering[offset++] = index;
            }
        }
    }
}

template<typename R>
int
psp::DistHelmholtz<R>::ReorderedIndex
( int x, int y, int z, int zSize ) const
{
    int index = 
        ReorderedIndexRecursion
        ( x, y, z, control_.nx, control_.ny, zSize, log2CommSize_, 
          control_.cutoff, 0 );
    return index;
}

template<typename R>
int
psp::DistHelmholtz<R>::ReorderedIndexRecursion
( int x, int y, int z, int xSize, int ySize, int zSize,
  int depthTilSerial, int cutoff, int offset )
{
    const int nextDepthTilSerial = std::max(depthTilSerial-1,0);
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
    {
        // We have satisfied the nested dissection constraints
        return offset + (x+y*xSize+z*xSize*ySize);
    }
    else if( xSize >= ySize )
    {
        // Partition the X dimension
        const int middle = (xSize-1)/2;
        if( x < middle )
        {
            return ReorderedIndexRecursion
            ( x, y, z, middle, ySize, zSize, nextDepthTilSerial, cutoff,
              offset );
        }
        else if( x == middle )
        {
            return offset + std::max(xSize-1,0)*ySize*zSize + (y+z*ySize);
        }
        else // x > middle
        {
            return ReorderedIndexRecursion
            ( x-middle-1, y, z, std::max(xSize-middle-1,0), ySize, zSize,
              nextDepthTilSerial, cutoff, offset+middle*ySize*zSize );
        }
    }
    else
    {
        // Partition the Y dimension
        const int middle = (ySize-1)/2;
        if( y < middle )
        {
            return ReorderedIndexRecursion
            ( x, y, z, xSize, middle, zSize, nextDepthTilSerial, cutoff, 
              offset );
        }
        else if( y == middle )
        {
            return offset + xSize*std::max(ySize-1,0)*zSize + (x+z*xSize);
        }
        else // y > middle 
        {
            return ReorderedIndexRecursion
            ( x, y-middle-1, z, xSize, std::max(ySize-middle-1,0), zSize,
              nextDepthTilSerial, cutoff, offset+xSize*middle*zSize );
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::FillOrigPanelStruct
( int zSize, clique::symbolic::SymmOrig& S ) const
{
    int nxSub=control_.nx, nySub=control_.ny, xOffset=0, yOffset=0;    
    FillDistOrigPanelStruct( zSize, nxSub, nySub, xOffset, yOffset, S );
    FillLocalOrigPanelStruct( zSize, nxSub, nySub, xOffset, yOffset, S );
}

template<typename R>
void
psp::DistHelmholtz<R>::FillDistOrigPanelStruct
( int zSize, int& nxSub, int& nySub, int& xOffset, int& yOffset,
  clique::symbolic::SymmOrig& S ) const
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int cutoff = control_.cutoff;
    const unsigned commRank = clique::mpi::CommRank( comm_ );
    S.dist.comm = comm_;
    // Fill the distributed nodes
    for( int s=log2CommSize_; s>0; --s )
    {
        clique::symbolic::DistSymmOrigSupernode& sn = S.dist.supernodes[s];
        const int powerOfTwo = 1u<<(s-1);
        const bool onLeft = (commRank&powerOfTwo) == 0;
        if( nxSub >= nySub )
        {
            // Form the structure of a partition of the X dimension
            const int middle = (nxSub-1)/2;
            sn.size = nySub*zSize;
            sn.offset = ReorderedIndex( xOffset+middle, yOffset, 0, zSize );

            // Allocate space for the lower structure
            int numJoins = 0;
            if( yOffset-1 >= 0 )
                ++numJoins;
            if( yOffset+nySub < ny )
                ++numJoins;
            sn.lowerStruct.resize( numJoins*zSize );

            // Fill the (unsorted) lower structure
            int joinOffset = 0;
            if( yOffset-1 >= 0 )
            {
                for( int i=0; i<zSize; ++i )
                    sn.lowerStruct[i] = ReorderedIndex
                    ( xOffset+middle, yOffset-1, i, zSize );
                joinOffset += zSize;
            }
            if( yOffset+nySub < ny )
            {
                for( int i=0; i<zSize; ++i )
                    sn.lowerStruct[joinOffset+i] = ReorderedIndex
                    ( xOffset+middle, yOffset+nySub, i, zSize );
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
            sn.size = nxSub*zSize;
            sn.offset = ReorderedIndex( xOffset, yOffset+middle, 0, zSize );

            // Allocate space for the lower structure
            int numJoins = 0;
            if( xOffset-1 >= 0 )
                ++numJoins;
            if( xOffset+nxSub < nx )
                ++numJoins;
            sn.lowerStruct.resize( numJoins*zSize );

            // Fill the (unsorted) lower structure
            int joinOffset = 0;
            if( xOffset-1 >= 0 )
            {
                for( int i=0; i<zSize; ++i )
                    sn.lowerStruct[i] = ReorderedIndex
                    ( xOffset-1, yOffset+middle, i, zSize );
                joinOffset += zSize;
            }
            if( xOffset+nxSub < nx )
            {
                for( int i=0; i<zSize; ++i )
                    sn.lowerStruct[joinOffset+i] = ReorderedIndex
                    ( xOffset+nxSub, yOffset+middle, i, zSize );
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
        sn.size = nxSub*nySub*zSize;
        sn.offset = ReorderedIndex( xOffset, yOffset, 0, zSize );

        // Count, allocate, and fill the lower struct
        int joinSize = 0;
        if( xOffset-1 >= 0 )
            joinSize += nySub*zSize;
        if( xOffset+nxSub < nx )
            joinSize += nySub*zSize;
        if( yOffset-1 >= 0 )
            joinSize += nxSub*zSize;
        if( yOffset+nySub < ny )
            joinSize += nxSub*zSize;
        sn.lowerStruct.resize( joinSize );

        int joinOffset = 0;
        if( xOffset-1 >= 0 )
        {
            for( int i=0; i<zSize; ++i )
                for( int j=0; j<nySub; ++j )
                    sn.lowerStruct[i*nySub+j] = ReorderedIndex
                    ( xOffset-1, yOffset+j, i, zSize );
            joinOffset += nySub*zSize;
        }
        if( xOffset+nxSub < nx )
        {
            for( int i=0; i<zSize; ++i )
                for( int j=0; j<nySub; ++j )
                    sn.lowerStruct[joinOffset+i*nySub+j] = ReorderedIndex
                    ( xOffset+nxSub, yOffset+j, i, zSize );
            joinOffset += nySub*zSize;
        }
        if( yOffset-1 >= 0 )
        {
            for( int i=0; i<zSize; ++i )
                for( int j=0; j<nxSub; ++j )
                    sn.lowerStruct[joinOffset+i*nxSub+j] = ReorderedIndex
                    ( xOffset+j, yOffset-1, i, zSize );
            joinOffset += nxSub*zSize;
        }
        if( yOffset+nySub < ny )
        {
            for( int i=0; i<zSize; ++i )
                for( int j=0; j<nxSub; ++j )
                    sn.lowerStruct[joinOffset+i*nxSub+j] = ReorderedIndex
                    ( xOffset+j, yOffset+nySub, i, zSize );
        }

        // Sort the lower structure
        std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
    }
    else if( nxSub >= nySub )
    {
        // Form the structure of a partition of the X dimension
        const int middle = (nxSub-1)/2;
        sn.size = nySub*zSize;
        sn.offset = ReorderedIndex( xOffset+middle, yOffset, 0, zSize );

        // Allocate space for the lower structure
        int numJoins = 0;
        if( yOffset-1 >= 0 )
            ++numJoins;
        if( yOffset+nySub < ny )
            ++numJoins;
        sn.lowerStruct.resize( numJoins*zSize );

        // Fill the (unsorted) lower structure
        int joinOffset = 0;
        if( yOffset-1 >= 0 )
        {
            for( int i=0; i<zSize; ++i )
                sn.lowerStruct[i] = ReorderedIndex
                ( xOffset+middle, yOffset-1, i, zSize );
            joinOffset += zSize;
        }
        if( yOffset+nySub < ny )
        {
            for( int i=0; i<zSize; ++i )
                sn.lowerStruct[joinOffset+i] = ReorderedIndex
                ( xOffset+middle, yOffset+nySub, i, zSize );
        }

        // Sort the lower structure
        std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
    }
    else
    {
        // Form the structure of a partition of the Y dimension
        const int middle = (nySub-1)/2;
        sn.size = nxSub*zSize;
        sn.offset = ReorderedIndex( xOffset, yOffset+middle, 0, zSize );

        // Allocate space for the lower structure
        int numJoins = 0;
        if( xOffset-1 >= 0 )
            ++numJoins;
        if( xOffset+nxSub < nx )
            ++numJoins;
        sn.lowerStruct.resize( numJoins*zSize );

        // Fill the (unsorted) lower structure
        int joinOffset = 0;
        if( xOffset-1 >= 0 )
        {
            for( int i=0; i<zSize; ++i )
                sn.lowerStruct[i] = ReorderedIndex
                ( xOffset-1, yOffset+middle, i, zSize );
            joinOffset += zSize;
        }
        if( xOffset+nxSub < nx )
        {
            for( int i=0; i<zSize; ++i )
                sn.lowerStruct[joinOffset+i] = ReorderedIndex
                ( xOffset+nxSub, yOffset+middle, i, zSize );
        }

        // Sort the lower structure
        std::sort( sn.lowerStruct.begin(), sn.lowerStruct.end() );
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::FillLocalOrigPanelStruct
( int zSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
  clique::symbolic::SymmOrig& S ) const
{
    const int numLocalSupernodes = S.local.supernodes.size();
    const int cutoff = control_.cutoff;
    const int nx = control_.nx;
    const int ny = control_.ny;

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

        if( box.nx*box.ny*zSize <= cutoff )
        {
            sn.size = box.nx*box.ny*zSize;
            sn.offset = ReorderedIndex( box.xOffset, box.yOffset, 0, zSize );
            sn.children.clear();

            // Count, allocate, and fill the lower struct
            int joinSize = 0;
            if( box.xOffset-1 >= 0 )
                joinSize += box.ny*zSize;
            if( box.xOffset+box.nx < nx )
                joinSize += box.ny*zSize;
            if( box.yOffset-1 >= 0 )
                joinSize += box.nx*zSize;
            if( box.yOffset+box.ny < ny )
                joinSize += box.nx*zSize;
            sn.lowerStruct.resize( joinSize );

            int joinOffset = 0;
            if( box.xOffset-1 >= 0 )
            {
                for( int i=0; i<zSize; ++i )
                    for( int j=0; j<box.ny; ++j )
                        sn.lowerStruct[i*box.ny+j] = ReorderedIndex
                        ( box.xOffset-1, box.yOffset+j, i, zSize );
                joinOffset += box.ny*zSize;
            }
            if( box.xOffset+box.nx < nx )
            {
                for( int i=0; i<zSize; ++i )
                    for( int j=0; j<box.ny; ++j )
                        sn.lowerStruct[joinOffset+i*box.ny+j] = ReorderedIndex
                        ( box.xOffset+box.nx, box.yOffset+j, i, zSize );
                joinOffset += box.ny*zSize;
            }
            if( box.yOffset-1 >= 0 )
            {
                for( int i=0; i<zSize; ++i )
                    for( int j=0; j<box.nx; ++j )
                        sn.lowerStruct[joinOffset+i*box.nx+j] = ReorderedIndex
                        ( box.xOffset+j, box.yOffset-1, i, zSize );
                joinOffset += box.nx*zSize;
            }
            if( box.yOffset+box.ny < ny )
            {
                for( int i=0; i<zSize; ++i )
                    for( int j=0; j<box.nx; ++j )
                        sn.lowerStruct[joinOffset+i*box.nx+j] = ReorderedIndex
                        ( box.xOffset+j, box.yOffset+box.ny, i, zSize );
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
                sn.size = box.ny*zSize;
                sn.offset = ReorderedIndex
                    ( box.xOffset+middle, box.yOffset, 0, zSize );

                // Count, allocate, and fill the lower struct
                int numJoins = 0;
                if( box.yOffset-1 >= 0 )
                    ++numJoins;
                if( box.yOffset+box.ny < ny )
                    ++numJoins;
                sn.lowerStruct.resize( numJoins*zSize );

                int joinOffset = 0;
                if( box.yOffset-1 >= 0 )
                {
                    for( int i=0; i<zSize; ++i )
                        sn.lowerStruct[i] = ReorderedIndex
                        ( box.xOffset+middle, box.yOffset-1, i, zSize );
                    joinOffset += zSize;
                }
                if( box.yOffset+box.ny < ny )
                {
                    for( int i=0; i<zSize; ++i )
                        sn.lowerStruct[joinOffset+i] = ReorderedIndex
                        ( box.xOffset+middle, box.yOffset+box.ny, i, zSize );
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
                sn.size = box.nx*zSize;
                sn.offset = ReorderedIndex
                    ( box.xOffset, box.yOffset+middle, 0, zSize );

                // Count, allocate, and fill the lower struct
                int numJoins = 0;
                if( box.xOffset-1 >= 0 )
                    ++numJoins;
                if( box.xOffset+box.nx < nx )
                    ++numJoins;
                sn.lowerStruct.resize( numJoins*zSize );

                int joinOffset = 0;
                if( box.xOffset-1 >= 0 )
                {
                    for( int i=0; i<zSize; ++i )
                        sn.lowerStruct[i] = ReorderedIndex
                        ( box.xOffset-1, box.yOffset+middle, i, zSize );
                    joinOffset += zSize;
                }
                if( box.xOffset+box.nx < nx )
                {
                    for( int i=0; i<zSize; ++i )
                        sn.lowerStruct[joinOffset+i] = ReorderedIndex
                        ( box.xOffset+box.nx, box.yOffset+middle, i, zSize );
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
