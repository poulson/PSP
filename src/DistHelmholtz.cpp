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
    RecursiveReordering( 0, nx, 0, ny, cutoff, &reordering[0] );

    // Construct the inverse map
    std::vector<int> inverseReordering( planeSize );
    for( int i=0; i<planeSize; ++i )
        inverseReordering[reordering[i]] = i;

    // Compute the depths of each interior panel class
    const int innerDepth = nz-(topDepth+bottomOrigDepth);
    const int leftoverInnerDepth = innerDepth % planesPerPanel;

    // Compute the number of full-sized interior artificially padded panels
    const int numFullInnerPanels = innerDepth / planesPerPanel;
    
    // Count the total number of panels:
    //    -----------
    //   | Top       |
    //   | Inner     |
    //       ...     
    //   | Leftover? |
    //   | Bottom    |
    //    -----------
    const bool haveLeftover = ( leftoverInnerDepth != 0 );
    const int numTotalPanels = 2 + numFullInnerPanels + haveLeftover;

    // TODO: Compute our local height of the distributed sparse matrix

    // TODO: Fill the original structures of the panels
}

template<typename F>
void 
psp::DistHelmholtz<F>::RecursiveReordering
( int xOffset, int xSize, int yOffset, int ySize, int cutoff,
  int* reordering ) const
{
    if( xSize*ySize <= cutoff )
    {
        // Write the leaf
        for( int x=xOffset; x<xOffset+xSize; ++x )
            for( int y=yOffset; y<yOffset+ySize; ++y )
                reordering[(x-xOffset)*ySize+(y-yOffset)] = x+y*control_.nx;
    }
    else if( xSize >= ySize )
    {
        // Cut the x dimension and write the separator
        const int xLeftSize = (xSize-1) / 2;
        const int separatorSize = ySize;
        int* separatorSection = &reordering[xSize*ySize-separatorSize];
        for( int y=yOffset; y<yOffset+ySize; ++y )
            separatorSection[y-yOffset] = (xOffset+xLeftSize)+y*control_.nx;
        // Recurse on the left side of the x cut
        RecursiveReordering
        ( xOffset, xLeftSize, yOffset, ySize, cutoff, reordering );
        // Recurse on the right side of the x cut
        RecursiveReordering
        ( xOffset+(xLeftSize+1), xSize-(xLeftSize+1), yOffset, ySize,
          cutoff, &reordering[xLeftSize*ySize] );
    }
    else
    {
        // Cut the y dimension and write the separator
        const int yLeftSize = (ySize-1) / 2;
        const int separatorSize = xSize;
        int* separatorSection = &reordering[xSize*ySize-separatorSize];
        for( int x=xOffset; x<xOffset+xSize; ++x )
            separatorSection[x-xOffset] = x+(yOffset+yLeftSize)*control_.nx;
        // Recurse on the left side of the y cut
        RecursiveReordering
        ( xOffset, xSize, yOffset, yLeftSize, cutoff, reordering );
        // Recurse on the right side of the y cut
        RecursiveReordering
        ( xOffset, xSize, yOffset+(yLeftSize+1), ySize-(yLeftSize+1),
          cutoff, &reordering[xSize*yLeftSize] );
    }
}

template class psp::DistHelmholtz<float>;
template class psp::DistHelmholtz<double>;
template class psp::DistHelmholtz<std::complex<float> >;
template class psp::DistHelmholtz<std::complex<double> >;
