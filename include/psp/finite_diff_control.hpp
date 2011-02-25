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

   Additional permissions under GNU GPL version 3 section 7

   If you modify this Program, or any covered work, by linking or combining it
   with MUMPS and/or ParMetis (or modified versions of those libraries),
   containing parts covered by the terms of the respective licenses of MUMPS
   and ParMetis, the licensors of this Program grant you additional permission
   to convey the resulting work. {Corresponding Source for a non-source form of
   such a combination shall include the source code for the parts of MUMPS and
   ParMetis used as well as that of the covered work.}
*/
#ifndef PSP_FINITE_DIFF_CONTROL_H
#define PSP_FINITE_DIFF_CONTROL_H 1
#include <vector>

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
    int nx; // number of grid points in x direction
    int ny; // number of grid points in y direction
    int nz; // number of grid points in z direction
    double wx; // width in x direction (left side at x=0)
    double wy; // width in y direction (left side at y=0)
    double wz; // width in z direction (left side at z=0)

    double omega; // angular frequency of problem
    double C; // coefficient for PML: basic form is C/eta * ((t-eta)/eta)^2
    int b; // width of PML is b*(wx/nx), b*(wy/ny), b*(wz/nz) in each direction
    int d; // number of layers to process at a time

    BoundaryCondition frontBC;
    BoundaryCondition rightBC;
    BoundaryCondition backBC;
    BoundaryCondition leftBC;
    BoundaryCondition bottomBC;
    // The top boundary condition must be PML since we are sweeping from it.
};

void
FiniteDiffInfo
( MPI_Comm comm, const FiniteDiffControl& control,
  int& firstLocalVertex, int& numLocalVertices, int& numVertices,
  int& numLocalNonzeros, int& numNonzeros );

void
FiniteDiffConnectivity
( const FiniteDiffControl& control, const int numNonzeros,
  std::vector<int>& rowIndices, std::vector<int>& colIndices );

} // namespace psp

#endif // PSP_FINITE_DIFF_CONTROL_H
