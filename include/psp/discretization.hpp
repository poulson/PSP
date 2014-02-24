/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_DISCRETIZATION_HPP
#define PSP_DISCRETIZATION_HPP

namespace psp {

enum Boundary { PML, DIRICHLET };
enum Stencil { SEVEN_POINT }; // Not very useful yet...

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
template<typename Real>
struct Discretization
{
    typedef Complex<Real> C;

    Stencil stencil; // only 7-point supported so far
    Real omega;      // frequency of problem [rad/sec]
    int nx, ny, nz;  // number of grid points in each direction
    Real wx, wy, wz; // width of the PML-padded box in each direction

    Boundary frontBC, rightBC, backBC, leftBC, topBC;
    // NOTE: the bottom boundary condition must be PML since we are 
    //       sweeping from it.
    int bx, by, bz; // number of grid points of PML in each direction
    Real sigmax, sigmay, sigmaz;   // coefficient for PML in each direction

    Discretization
    ( Real frequency, 
      int xSize, int ySize, int zSize, Real xWidth, Real yWidth, Real zWidth )
    : stencil(SEVEN_POINT), omega(frequency), 
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(PML), rightBC(PML), backBC(PML), leftBC(PML), topBC(PML),
      bx(5), by(5), bz(5)
    {
        const Real maxDim = std::max(wx,std::max(wy,wz));
        sigmax = sigmay = sigmaz = 1.2*maxDim;
    }
    
    Discretization
    ( Real frequency, 
      int xSize, int ySize, int zSize, Real xWidth, Real yWidth, Real zWidth,
      Boundary front, Boundary right, Boundary back,
      Boundary left, Boundary top )
    : stencil(SEVEN_POINT), omega(frequency), 
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(front), rightBC(right), backBC(back), leftBC(left), topBC(top),
      bx(5), by(5), bz(5)
    {
        const Real maxDim = std::max(wx,std::max(wy,wz));
        sigmax = sigmay = sigmaz = 1.2*maxDim;
    }

    Discretization
    ( Real frequency, 
      int xSize, int ySize, int zSize, Real xWidth, Real yWidth, Real zWidth,
      Boundary front, Boundary right, Boundary back,
      Boundary left, Boundary top,
      int xPMLSize, int yPMLSize, int zPMLSize )
    : stencil(SEVEN_POINT), omega(frequency), 
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(front), rightBC(right), backBC(back), leftBC(left), topBC(top),
      bx(xPMLSize), by(yPMLSize), bz(zPMLSize)
    {
        const Real maxDim = std::max(wx,std::max(wy,wz));
        sigmax = sigmay = sigmaz = 1.2*maxDim;
    }

    Discretization
    ( Real frequency,
      int xSize, int ySize, int zSize, 
      Real xWidth, Real yWidth, Real zWidth,
      Boundary front, Boundary right, Boundary back,
      Boundary left, Boundary top,
      int pmlSize, double sigma )
    : stencil(SEVEN_POINT), omega(frequency), 
      nx(xSize), ny(ySize), nz(zSize),
      wx(xWidth), wy(yWidth), wz(zWidth),
      frontBC(front), rightBC(right), backBC(back), leftBC(left), topBC(top),
      bx(pmlSize), by(pmlSize), bz(pmlSize),
      sigmax(sigma), sigmay(sigma), sigmaz(sigma)
    { }
};

} // namespace psp

#endif // ifndef PSP_DISCRETIZATION_HPP
