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
#ifndef PSP_FINITE_DIFF_CONTROL_HPP
#define PSP_FINITE_DIFF_CONTROL_HPP 1
#include "petscvec.h"

namespace psp {

enum BoundaryCondition { PML, DIRICHLET };
enum Stencil { SEVEN_POINT, TWENTY_SEVEN_POINT };

// Our control structure for defining the basic parameters for the problem 
// to be solved. The domain is assumed to be of the form:
//
//                 _______________ (wx,wy,0)
//                /              /|
//            x  /              / |
//              /              /  |
// sweep dir.  /______________/   |
//     ||      |              |   |
//     ||      |              |   / (wx,wy,wz)
//     ||    z |              |  /  
//     ||      |              | /  
//     \/      |______________|/
//          (0,0,wz)    y    (0,wy,wz)
//
// PML must be enforced at least on the bottom face, and the remaining faces 
// may be set as either PML or a zero Dirichlet boundary condition.
//
// The implied ordering of the nodes is that node (i,j,k) is in position 
//    i+j*nx+k*nx*ny
// so that the x direction is the shallowest and the z direction is the deepest.
// Our sweeping procedure occurs within the z dimensions so that each x-y plane
// is contiguous.
//
struct FiniteDiffControl
{
    Stencil stencil; // 7-point planned, only 7-point supported so far
    PetscInt nx, ny, nz; // number of grid points in each direction
    PetscReal wx, wy, wz; // width of the PML-padded box in each direction
    PetscReal omega; // frequency of problem

    PetscReal Cx, Cy, Cz; // coefficient for PML in each direction
    PetscReal etax, etay, etaz; // widths of PML in each direction

    PetscReal imagShift; // stabilizing shift for preconditioner
    PetscInt planesPerPanel; // number of xy planes to put into each panel

    BoundaryCondition frontBC;
    BoundaryCondition rightBC;
    BoundaryCondition backBC;
    BoundaryCondition leftBC;
    BoundaryCondition bottomBC;
    // The top boundary condition must be PML since we are sweeping from it.
};

} // namespace psp

#endif // PSP_FINITE_DIFF_CONTROL_HPP
