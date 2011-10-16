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

    // Create the 2d reordering structure
    const int planeSize = nx*ny;
    std::vector<int> reordering( planeSize );
    const int cutoff = 100;
    RecursiveReordering( nx, 0, nx, 0, ny, cutoff, 0, &reordering[0] );

    // Construct the inverse map
    std::vector<int> inverseReordering( planeSize );
    for( int i=0; i<planeSize; ++i )
        inverseReordering[reordering[i]] = i;

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
    int localHeight = 0;
    AddLocalHeight
    ( nx, ny, planesPerPanel, cutoff, commRank, log2CommSize, localHeight );
    localHeight *= numFullInnerPanels;
    AddLocalHeight
    ( nx, ny, topDepth, cutoff, commRank, log2CommSize, localHeight );
    if( haveLeftover )
        AddLocalHeight
        ( nx, ny, leftoverInnerDepth, cutoff, commRank, log2CommSize,
          localHeight );
    AddLocalHeight
    ( nx, ny, bottomOrigDepth, cutoff, commRank, log2CommSize, localHeight );
    localHeight_ = localHeight;

    // TODO: Fill the original structures of the panels
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
void
psp::DistHelmholtz<F>::AddLocalHeight
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
            AddLocalHeight
            ( xLeftSize, ySize, zSize, cutoff, 0, 0, localHeight );
            // Add the right side
            AddLocalHeight
            ( xSize-(xLeftSize+1), ySize, zSize, cutoff, 0, 0, localHeight );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            AddLocalHeight
            ( xSize-(xLeftSize+1), ySize, zSize, cutoff,
              commRank/2, log2CommSize-1, localHeight );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            AddLocalHeight
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
            AddLocalHeight
            ( xSize, yLeftSize, zSize, cutoff, 0, 0, localHeight );
            // Add the right side
            AddLocalHeight
            ( xSize, ySize-(yLeftSize+1), zSize, cutoff, 0, 0, localHeight );
        }
        else if( commRank & 1 )
        {
            // Add the right side
            AddLocalHeight
            ( xSize, ySize-(yLeftSize+1), zSize, cutoff, 
              commRank/2, log2CommSize-1, localHeight );
        }
        else // log2CommSize != 0 && commRank & 1 == 0
        {
            // Add the left side
            AddLocalHeight
            ( xSize, yLeftSize, zSize, cutoff,
              commRank/2, log2CommSize-1, localHeight );
        }
    }
}

template class psp::DistHelmholtz<float>;
template class psp::DistHelmholtz<double>;
template class psp::DistHelmholtz<std::complex<float> >;
template class psp::DistHelmholtz<std::complex<double> >;
