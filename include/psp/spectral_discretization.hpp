/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_SPECTRAL_DISCRETIZATION_HPP
#define PSP_SPECTRAL_DISCRETIZATION_HPP

namespace psp {

// Our control structure for defining the basic parameters for the problem 
// to be solved. The domain is assumed to be of the form:
//
//                 _______________ (wx,wy,0)
//                /              /|
//            x  /              / |
//  sweep dir   /              /  |
//     /\      /______________/   |
//     ||      |              |   |
//     ||      |              |   / (wx,wy,wz)
//     ||    z |              |  /  
//     ||      |              | /  
//             |______________|/
//          (0,0,wz)    y    (0,wy,wz)
//
// PML must be enforced at least on the bottom face, and the remaining faces 
// may be set as either PML or a zero Dirichlet boundary condition.
//
template<typename R>
struct SpectralDiscretization
{
    typedef Complex<R> C;

    R omega;         // frequency of problem [rad/sec]
    int polyOrder;   // the polynomial order for the Gauss Labatto grid
    int nx, ny, nz;  // number of grid points in each direction
                     // NOTE: must have nx-1 % polyOrder = 0
                     //                 ny-1 % polyOrder = 0
                     //                 nz-1 % polyOrder = 0
    R wx, wy, wz;    // width of the PML-padded box in each direction

    Boundary frontBC, rightBC, backBC, leftBC, topBC;
    // NOTE: the bottom boundary condition must be PML since we are 
    //       sweeping from it.
    int bx, by, bz; // number of grid points of PML in each direction
                    // NOTE: must have bx % polyOrder = 0
                    //                 by % polyOrder = 0
                    //                 bz % polyOrder = 0
    R sigmax, sigmay, sigmaz;   // coefficient for PML in each direction

    SpectralDiscretization
    ( R frequency, int pOrder,
      int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth )
    : omega(frequency), polyOrder(pOrder),
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(PML), rightBC(PML), backBC(PML), leftBC(PML), topBC(PML),
      bx(polyOrder), by(polyOrder), bz(polyOrder)
    {
        const R maxDim = std::max(wx,std::max(wy,wz));
        sigmax = sigmay = sigmaz = 1.2*maxDim;
    }
    
    SpectralDiscretization
    ( R frequency, int pOrder,
      int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth,
      Boundary front, Boundary right, Boundary back,
      Boundary left, Boundary top )
    : omega(frequency), polyOrder(pOrder),
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(front), rightBC(right), backBC(back), leftBC(left), topBC(top),
      bx(polyOrder), by(polyOrder), bz(polyOrder)
    {
        const R maxDim = std::max(wx,std::max(wy,wz));
        sigmax = sigmay = sigmaz = 1.2*maxDim;
    }

    SpectralDiscretization
    ( R frequency, int pOrder,
      int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth,
      Boundary front, Boundary right, Boundary back,
      Boundary left, Boundary top,
      int xPMLSize, int yPMLSize, int zPMLSize )
    : omega(frequency), polyOrder(pOrder),
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(front), rightBC(right), backBC(back), leftBC(left), topBC(top),
      bx(xPMLSize), by(yPMLSize), bz(zPMLSize)
    {
        const R maxDim = std::max(wx,std::max(wy,wz));
        sigmax = sigmay = sigmaz = 1.2*maxDim;
    }

    SpectralDiscretization
    ( R frequency, int pOrder,
      int xSize, int ySize, int zSize, 
      R xWidth, R yWidth, R zWidth,
      Boundary front, Boundary right, Boundary back,
      Boundary left, Boundary top,
      int pmlSize, double sigma )
    : omega(frequency), polyOrder(pOrder),
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(front), rightBC(right), backBC(back), leftBC(left), topBC(top),
      bx(pmlSize), by(pmlSize), bz(pmlSize),
      sigmax(sigma), sigmay(sigma), sigmaz(sigma)
    { }
};

} // namespace psp

#endif // PSP_SPECTRAL_DISCRETIZATION_HPP
