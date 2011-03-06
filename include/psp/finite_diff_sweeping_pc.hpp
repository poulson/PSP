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
#include <stdexcept>
#include <vector>

namespace psp {

enum SparseDirectSolver { MUMPS, MUMPS_SYMMETRIC, SUPERLU_DIST };

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
    FiniteDiffControl _control;
    SparseDirectSolver _solver;

    Vec& _slowness;
    std::vector<Mat> _panels;
    std::vector<Mat> _factors;

    MPI_Comm _comm;
    PetscInt _rank;
    PetscInt _numProcesses;
    PetscInt _numProcessRows; 
    PetscInt _numProcessCols; 
    PetscInt _myProcessRow;
    PetscInt _myProcessCol;
    PetscInt _xChunkSize;
    PetscInt _yChunkSize;
    PetscInt _myXOffset;
    PetscInt _myYOffset;
    PetscInt _myXPortion;
    PetscInt _myYPortion;

    bool _initialized;

public:
    FiniteDiffSweepingPC
    ( FiniteDiffControl& control, SparseDirectSolver solver, Vec slowness,
      PetscInt numProcessRows, PetscInt numProcessCols );

    ~FiniteDiffSweepingPC();

    void Init();
    void Destroy();

    void FormSymmetricRow
    ( PetscInt x, PetscInt y, PetscInt z, 
      PetscInt heightOffset, PetscInt panelHeight,
      std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const;

    void FormRow
    ( PetscInt x, PetscInt y, PetscInt z, 
      PetscInt heightOffset, PetscInt panelHeight,
      std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const;
};

} // namespace psp

// Implementation begins here

psp::FiniteDiffSweepingPC::FiniteDiffSweepingPC
( psp::FiniteDiffControl& control, 
  psp::SparseDirectSolver solver, 
  Vec slowness,
  PetscInt numProcessRows, 
  PetscInt numProcessCols )
: _control(control), _solver(solver), _slowness(slowness), 
  _numProcessRows(numProcessRows), _numProcessCols(numProcessCols), 
  _initialized(false)
{
    // Check that the slowness vector is the right length
    const PetscInt N = VecGetSize( _slowness );
    if( N != _control.nx*_control.ny*_control.nz )
        throw std::logic_error("Slowness must be of size nx*ny*nz");

    // Ensure that we have the right number of local entries in the slowness
    // vector. We distribute our box using a 2d block distribution over each 
    // panel. The process grid is specified to be r x c.
    PetscObjectGetComm( _slowness, &_comm );
    MPI_Comm_size( _comm, &_numProcesses );
    MPI_Comm_rank( _comm, &_rank );
    if( _numProcessRows*_numProcessCols != _numProcesses )
        throw std::logic_error("Invalid process grid dimensions");
    _myProcessRow = rank / _numProcessCols;
    _myProcessCol = rank % _numProcessCols;
    _xChunkSize = _control.nx / _numProcessCols;
    _yChunkSize = _control.ny / _numProcessRows;
    _myXOffset = _myProcessCol*_xChunkSize;
    _myYOffset = _myProcessRow*_yChunkSize;
    _myXPortion = ( _myProcessCol==_numProcessCols-1 ? 
                    _xChunkSize : 
                    _xChunkSize+(_control.nx%_xChunkSize) );
    _myYPortion = ( _myProcessRow==_numProcessRows-1 ?
                    _yChunkSize :
                    _yChunkSize+(_control.ny%_yChunkSize) );
    const PetscInt localN = VecGetLocalSize( _slowness );
    if( localN != _myXPortion*_myYPortion*_control.nz )
        throw std::runtime_error("Slowness is not properly distributed");
}

// Return the 4 nonzeros in the left half of the row for node (x,y,z) of the 
// 7-point finite difference approximation defined by the information in 
// _control. 
//
// In the cases where one of these four connections does not exist, fill its 
// index in as -1 so that PETSc will ignore it.
//
// We will later turn this into a wrapper for generating the row defined by 
// the particular stencil defined in _control.
void
psp::FiniteDiffSweepingPC::FormSymmetricRow
( PetscInt x, PetscInt y, PetscInt z, 
  PetscInt heightOffset, PetscInt panelHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*panelHeight;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*panelHeight + 
        _myProcessCol*pillarSize;
    const PetscInt localX = x - _myXOffset;
    const PetscInt localY = y - _myYOffset;
    const PetscInt localZ = z - heightOffset;
    const PetscInt rowIdx = myOffset + 
        localX + localY*_myXPortion + localZ*_myXPortion*_myYPortion;

    // Fill in value and local index for the diagonal entry in this panel + PML
    PetscInt entry = 0;
    colIndices[0] = rowIdx;
    // TODO: Fill finite diff approx into row[0]
    ++entry;

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        if( localX != 0 )
        {
            colIndices[entry] = rowIdx - 1;
        }
        else
        {
            const PetscInt frontProcessOffset = myOffset - pillarSize;
            colIndices[entry] = frontProcessOffset + (_xChunkSize-1) + 
                localY*_xChunkSize + localZ*_xChunkSize*_yChunkSize;
        }
        // TODO: Fill finite diff approx into row[1]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        if( localY != 0 )
        {
            colIndices[entry] = rowIdx - _myXPortion;
        }
        else
        {
            const PetscInt leftProcessOffset = 
                myOffset - _control.nx*_yChunkSize*panelHeight;
            colIndices[entry] = leftProcessOffset + localX + 
                (_yChunkSize-1)*_xChunkSize + localZ*_xChunkSize*_yChunkSize;
        }
        // TODO: Fill finite diff approx into row[2]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Top connection to (x,y,z-1)
    if( localZ != 0 )
    {
        colIndices[entry] = rowIdx - _myXPortion*_myYPortion;
        // TODO: Fill finite diff approx into row[3]
    }
    else
    {
        colIndices[entry] = -1;
    }
}

// Return the 7 nonzeros in the row for node (x,y,z) of the 
// 7-point finite difference approximation defined by the information in 
// _control. 
//
// In the cases where one of these seven connections does not exist, fill its 
// index in as -1 so that PETSc will ignore it.
//
// We will later turn this into a wrapper for generating the row defined by 
// the particular stencil defined in _control.
void
psp::FiniteDiffSweepingPC::FormRow
( PetscInt x, PetscInt y, PetscInt z, 
  PetscInt heightOffset, PetscInt panelHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*panelHeight;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*panelHeight + 
        _myProcessCol*pillarSize;
    const PetscInt localX = x - _myXOffset;
    const PetscInt localY = y - _myYOffset;
    const PetscInt localZ = z - heightOffset;
    const PetscInt rowIdx = myOffset + 
        localX + localY*_myXPortion + localZ*_myXPortion*_myYPortion;

    // Fill in value and local index for the diagonal entry in this panel + PML
    PetscInt entry = 0;
    colIndices[entry] = rowIdx;
    // TODO: Fill finite diff approx into row[0]
    ++entry;

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        if( localX != 0 )
        {
            colIndices[entry] = rowIdx - 1;
        }
        else
        {
            const PetscInt frontProcessOffset = myOffset - pillarSize;
            colIndices[entry] = frontProcessOffset + (_xChunkSize-1) + 
                localY*_xChunkSize + localZ*_xChunkSize*_yChunkSize;
        }
        // TODO: Fill finite diff approx into row[1]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Back connection to (x+1,y,z)
    if( x != _control.nx-1 )
    {
        if( localX != _myXPortion-1 )
        {
            colIndices[entry] = rowIdx+1;
        }
        else
        {
            const PetscInt backProcessOffset = myOffset + pillarSize;
            const PetscInt backProcessXPortion = 
                std::min( _xChunkSize, _control.nx-(_myXOffset+_xChunkSize) );
            colIndices[entry] = backProcessOffset + localY*backProcessXPortion +
                localZ*backProcessXPortion*_myYPortion;
        }
        // TODO: Fill finite diff approx into row[2]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        if( localY != 0 )
        {
            colIndices[entry] = rowIdx - _myXPortion;
        }
        else
        {
            const PetscInt leftProcessOffset = 
                myOffset - _control.nx*_yChunkSize*panelHeight;
            colIndices[entry] = leftProcessOffset + localX + 
                (_yChunkSize-1)*_xChunkSize + localZ*_xChunkSize*_yChunkSize;
        }
        // TODO: Fill finite diff approx into row[3]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Right connection to (x,y+1,z)
    if( y != _control.ny )
    {
        if( localY != _myYPortion )
        {
            colIndices[entry] = rowIdx + _myXPortion;
        }
        else
        {
            const PetscInt rightProcessYPortion =
                std::min( _yChunkSize, _control.ny-(_myYOffset+_yChunkSize) );
            const PetscInt rightProcessOffset = 
                myOffset + _control.nx*rightProcessYPortion*panelHeight;
            colIndices[entry] = rightProcessOffset + localX + 
                localZ*_xChunkSize*rightProcessYPortion;
        }
        // TODO: Fill finite diff approx into row[4]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Top connection to (x,y,z-1)
    if( localZ != 0 )
    {
        colIndices[entry] = rowIdx - _myXPortion*_myYPortion;
        // TODO: Fill finite diff approx into row[5]
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Bottom connection to (x,y,z+1)
    if( localZ != panelHeight-1 )
    {
        colIndices[entry] = rowIdx + _myXPortion*_myYPortion;
        // TODO: Fill finite diff approx into row[6]
    }
    else
    {
        colIndices[entry] = -1;
    }
}

void
psp::FiniteDiffSweepingPC::Init()
{
    // Initialize each panel
    const PetscInt nzMinusTopPML = _control.nz - _control.b;
    const PetscInt numPanels = 
        ( nzMinusTopPML>0 ? (nzMinusTopPML-1)/_control.b+1 : 0 );
    _panels.resize( numPanels );
    _factors.resize( numPanels );
    if( _solver == MUMPS_SYMMETRIC )
    {
        // Our solver supports symmetry, so only fill the lower triangles
        for( PetscInt panel=0; panel<numPanels; ++panel )
        {
            Mat& A = _panels[j];
            MatCreate( _comm, &A );

            const PetscInt panelHeight = 
                ( panel==numPanels-1 ? 
                  _control.b + _control.d : 
                  _control.b + (_control.nz-panel*_control.d) );
            const PetscInt heightOffset = panel*(_control.b+_control.d);
            const PetscInt heightOfPML = heightOffset + _control.b;

            const PetscInt localSize = _myXPortion*_myYPortion*panelHeight;
            const PetscInt size = _control.nx*_control.ny*panelHeight;
            MatSetSizes( A, localSize, size, size, size );

            // SBAIJ is required for symmetry support
            MatSetType( A, MATMPISBAIJ ); 
            MatSetBlocksize( A, 1 );

            // Preallocate memory. This should be an upper bound for 7-point.
            MatMPISBAIJSetPreallocation( A, 1, 3, PETSC_NULL, 3, PETSC_NULL );

            // Fill with 7-point stencil
            std::vector<PetscScalar> row(4);
            std::vector<PetscInt> colIndices(4);
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( A, &iStart, &iEnd );
            for( PetscInt i=iStart; i<iEnd; ++i )
            {
                const PetscInt localX = (i-iStart) % _myXPortion;
                const PetscInt localY = ((i-iStart)/_myXPortion)%_myYPortion;
                const PetscInt localZ = (i-iStart) / (_myXPortion*_myYPortion);
                const PetscInt x = _myXOffset + localX;
                const PetscInt y = _myYOffset + localY;
                const PetscInt z = heightOffset + localZ;

                // Compute this row. The data is filled into the four entries
                // as follows:
                //   [0]: diagonal entry
                //   [1]: left connection (if it exists)
                //   [2]: front connection (if it exists)
                //   [3]: bottom connection (if it exists)
                FormSymmetricRow
                ( x, y, z, heightOffset, row, colIndices );

                MatSetValues
                ( A, &row[0], 1, &i, 4, &colIndices[0], INSERT_VALUES );
            }
            MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _factors[j];
            MatGetFactor( A, MAT_SOLVER_MUMPS, MAT_FACTOR_CHOLESKY, &F );
            MatFactorInfo cholInfo;
            cholInfo.fill = 3.0; // TODO: Tweak/expose this
            cholInfo.dtcol = 0.5; // TODO: Tweak/expose this
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            MatCholeskyFactorSymbolic( F, A, perm, &cholInfo );
            MatCholeskyFactorNumeric( F, A, &cholInfo );
        }
    }
    else // solver == MUMPS or solver == SUPERLU_DIST
    {
        // Our solver does not support symmetry, fill the entire matrices
        for( PetscInt j=0; j<numPanels; ++j )
        {
            Mat& A = _panels[j];
            MatCreate( _comm, &A );

            const PetscInt height = 
                ( j==numPanels-1 ? 
                  _control.b + _control.d : 
                  _control.b + (_control.nz-j*_control.d) );

            const PetscInt localSize = _mySlicePortion*height;
            const PetscInt size = _sliceSize*height;
            MatSetSizes( A, localSize, size, size, size );

            MatSetType( A, MATMPIAIJ );

            // Preallocate memory
            MatMPIAIJSetPreallocation( A, 5, PETSC_NULL, 5, PETSC_NULL );

            // Fill with 7-point stencil
            std::vector<PetscScalar> row(7);
            std::vector<PetscInt> colIndices(7);
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( A, &iStart, &iEnd );
            for( PetscInt i=iStart; i<iEnd; ++i )
            {
                const PetscInt localX = (i-iStart) % _myXPortion;
                const PetscInt localY = ((i-iStart)/_myXPortion)%_myYPortion;
                const PetscInt localZ = (i-iStart) / (_myXPortion*_myYPortion);
                const PetscInt x = _myXOffset + localX;
                const PetscInt y = _myYOffset + localY;
                const PetscInt z = heightOffset + localZ;

                // Compute this entire row.
                FormRow
                ( x, y, z, heightOffset, row, colIndices );

                MatSetValues
                ( A, &row[0], 1, &i, 7, &colIndices[0], INSERT_VALUES );
            }
            MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _factors[j];
            if( solver == MUMPS )
                MatGetFactor( A, MAT_SOLVER_MUMPS, MAT_FACTOR_LU, &F );
            else
                MatGetFactor( A, MAT_SOLVER_SUPERLU_DIST, MAT_FACTOR_LU, &F );
            MatFactorInfo luInfo;
            luInfo.fill = 3.0; // TODO: Tweak/expose this
            luInfo.dtcol = 0.5; // TODO: Tweak/expose this
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            MatLUFactorSymbolic( F, A, perm, &luInfo );
            MatLUFactorNumeric( F, A, &luInfo );
        }
    }
    
    _initialized = true;
}

psp::FiniteDiffSweepingPC::Destroy()
{
    if( !_initialized )
    {
        throw std::logic_error
        ("Cannot destroy preconditioner without initializing");
    }
    for( std::size_t j=0; j<_panels.size(); ++j )
    {
        MatDestroy( _panels[j] );
        MatDestroy( _factors[j] );
    }
    _initialized = false;
}

psp::FiniteDiffSweepingPC::~FiniteDiffSweepingPC()
{
    if( _initialized )
        Destroy();
}

#endif // PSP_FINITE_DIFF_SWEEPING_PC_HPP
