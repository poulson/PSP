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
#ifndef PSP_FINITE_DIFF_SWEEPING_PC_HPP
#define PSP_FINITE_DIFF_SWEEPING_PC_HPP 1
#include "petscpc.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace psp {

enum SparseDirectSolver { MUMPS, MUMPS_SYMMETRIC };

// This class is being designed to serve as the context for a PETSc PCShell 
// preconditioner. 
//
// In order to simplify the interface without compromising efficiency, the 
// slowness data is ordered over the grid such that each process owns a
// box in the xy plane that is extruded in the z dimension. Recalling that 
// our domain is:
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
// we split each xy plane into a 2d grid of dimensions
// numProcessRows x numProcessCols. For instance, with a 2 x 3 grid where the
// x and y dimensions are 9 and 22, chunks of each xy plane would be owned as 
// follows (notice that the last row/column gets the extra data):
//    ________________________
//   |       |       |        |
//   |Process|Process|Process |
//   |   3   |   4   |   5    |
//   |       |       |        |
// x |_______|_______|________|
//   |       |       |        |
//   |Process|Process|Process |
//   |   0   |   1   |   2    |
//   |_______|_______|________|
//               y
//
// In order to ensure compatibility with PETSc's 1d matrix distribution
// approach, we order our matrices so that all of process 0's data comes first,
// in an x-y-z ordering, then process 1's, etc.
//
// In the above case, if the z dimension was 10, process 0's local x, y, and z 
// dimensions would be 4, 7, and 10, respectively. Thus it would own the first
// 280 rows of the matrix, process 1 would own the next 280, and process 2 
// would own the next 320, since its y dimension is 8 (the last column also gets
// the remainder of 22/3). Proceding, processes 3 and 4 will each own 350 rows,
// and process 5 will own 400. The load balance is clearly not very good for 
// this small example problem, but it will be much better for 1000^3 grids.
//

class FiniteDiffSweepingPC
{
    MPI_Comm _comm;
    PetscInt _rank, _numProcesses;
    const PetscInt _numProcessRows, _numProcessCols;
    PetscInt _myProcessRow, _myProcessCol;
    PetscInt _xChunkSize, _yChunkSize;
    PetscInt _myXOffset, _myYOffset;
    PetscInt _myXPortion, _myYPortion;

    const FiniteDiffControl _control;
    const SparseDirectSolver _solver;

    const PetscReal _hx, _hy, _hz; // grid spacings
    const PetscReal _bx, _by, _bz; // eta/h
    const PetscInt _bzPadded; // ceil(eta/h)

    Vec* _slowness;
    PetscScalar* _localSlownessData;
    std::vector<PetscInt> _zSizesOfPanels;
    std::vector<PetscInt> _zSizesOfPaddedPanels;
    std::vector<PetscInt> _zOffsetsOfPanels;
    std::vector<PetscInt> _zOffsetsOfPaddedPanels;
    std::vector<PetscReal> _zPmlSizesOfPanels;
    std::vector<PetscReal> _zPmlSizesOfPaddedPanels;
    std::vector<Mat> _paddedFactors;
    std::vector<Mat> _offDiagBlocks;

    bool _initialized; // do we have an active set of factorizations?

    // These functions return the inverse of the 's' functions associated with
    // PML in each of the three dimensions.
    PetscScalar s1Inv( PetscInt x ) const;
    PetscScalar s2Inv( PetscInt y ) const;
    PetscScalar s3Inv( PetscInt z ) const;

    // Since the PML will often be pushed down the z direction, we will need
    // a means of specifying where the artificial PML is in the z direction.
    PetscScalar s3InvArtificial
    ( PetscInt z, PetscInt zOffset, PetscReal sizeOfPml ) const;

    // Return the maximum number of nonzeros in the right side of each row
    PetscInt GetSymmetricRowSize() const;

    // Return the indices and values for each nonzero in the right side of the
    // row defined by node (x,y,z) in the panel defined by its top z-location
    // (zOffset), its z size (zSize), and its PML size on the top (pmlSize).
    //
    // In the cases where a connection is invalid, fill its index with -1 so 
    // that PETSc's MatSetValues will ignore it.
    PetscInt FormSymmetricRow
    ( PetscReal imagShift,
      PetscInt x, PetscInt y, PetscInt z, 
      PetscInt zOffset, PetscInt zSize, PetscReal pmlSize,
      std::vector<PetscScalar>& nonzeros, std::vector<PetscInt>& colIndices ) 
    const;

    // Return the maximum number of nonzeros in a row
    PetscInt GetRowSize() const;

    // Return the indices and values for each nonzero in the 
    // row defined by node (x,y,z) in the panel defined by its top z-location
    // (zOffset), its z size (zSize), and its PML size at the top (pmlSize).
    //
    // In the cases where a connection is invalid, fill its index with -1 so 
    // that PETSc's MatSetValues will ignore it.
    PetscInt FormRow
    ( PetscReal imagShift,
      PetscInt x, PetscInt y, PetscInt z, 
      PetscInt zOffset, PetscInt zSize, PetscReal pmlSize,
      std::vector<PetscScalar>& nonzeros, std::vector<PetscInt>& colIndices ) 
    const;
    
    // Return the maximum number of nonzeros in the connections from a node
    // in one panel to nodes in the next panel.
    PetscInt GetPanelConnectionSize() const;

    // Return the column indices for the nonzero entries in the row of the 
    // off-diagonal block defined by node (x,y,z). The ordering of the 
    // off-diagonal block is constrained by the diagonal blocks being ordered 
    // in the fashion described at the top of this file. 
    PetscInt FormPanelConnections
    ( PetscInt x, PetscInt y, PetscInt z, 
      PetscInt zSize, PetscInt zSizeNext, PetscInt rowIndex,
      std::vector<PetscScalar>& nonzeros, std::vector<PetscInt>& colIndices ) 
    const;

public:
    FiniteDiffSweepingPC
    ( MPI_Comm comm, PetscInt numProcessRows, PetscInt numProcessCols, 
      FiniteDiffControl& control, SparseDirectSolver solver );

    ~FiniteDiffSweepingPC();

    // Return the number of entries our process will own of parallel vectors
    // that are conformal with our forward operator
    PetscInt GetLocalSize() const;

    // Return properties about our local rod of data
    PetscInt GetMyXOffset() const;
    PetscInt GetMyYOffset() const;
    PetscInt GetMyXPortion() const;
    PetscInt GetMyYPortion() const;

    // Return the mesh spacings
    PetscReal GetXSpacing() const;
    PetscReal GetYSpacing() const;
    PetscReal GetZSpacing() const;

    // Form the forward operator A and set up the preconditioner
    void Init( Vec& slowness, Mat& A );

    // Free all of the resources allocated during Init
    void Destroy();

    // Apply sweeping preconditioner to x, producing y
    void Apply( Vec& x, Vec& y ) const;

    // Form a Parallel VTK VTI file from a conformal vector.
    void WriteParallelVtkFile( Vec& v, const char* basename ) const;
};

} // namespace psp

#endif // PSP_FINITE_DIFF_SWEEPING_PC_HPP
