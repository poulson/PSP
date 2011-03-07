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

    Vec* _slowness;
    std::vector<Mat> _paddedPanels;
    std::vector<Mat> _paddedFactors;
    std::vector<Mat> _subdiagonalBlocks;

    bool _initialized;

public:
    FiniteDiffSweepingPC
    ( FiniteDiffControl& control, SparseDirectSolver solver, 
      PetscInt numProcessRows, PetscInt numProcessCols );

    ~FiniteDiffSweepingPC();

    PetscInt GetLocalSize() const;

    void Init( Vec& slowness, Mat& A );
    void Destroy();

    // Apply sweeping preconditioner to x, producing y
    void Apply( Vec& x, Vec& y ) const;

    // Return the maximum number of nonzeros in the left side of each row
    PetscInt GetSymmetricRowSize() const;

    // Return the indices and values for each nonzero in the left side of the
    // row defined by node (x,y,z) in the panel defined by its bottom z-location
    // (bottomOfPanel), its height (panelHeight), and its PML height from the 
    // bottom (pmlHeight).
    //
    // In the cases where a connection is invalid, fill its index with -1 so 
    // that PETSc's MatSetValues will ignore it.
    void FormSymmetricRow
    ( PetscReal imagShift,
      PetscInt x, PetscInt y, PetscInt z, 
      PetscInt bottomOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
      std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const;

    // Return the maximum number of nonzeros in a row
    PetscInt GetRowSize() const;

    // Return the indices and values for each nonzero in the 
    // row defined by node (x,y,z) in the panel defined by its bottom z-location
    // (bottomOfPanel), its height (panelHeight), and its PML height from the 
    // bottom (pmlHeight)
    //
    // In the cases where a connection is invalid, fill its index with -1 so 
    // that PETSc's MatSetValues will ignore it.
    void FormRow
    ( PetscReal imagShift,
      PetscInt x, PetscInt y, PetscInt z, 
      PetscInt bottomOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
      std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const;
    
    // Return the maximum number of nonzeros in the connections from a node
    // in one panel to nodes in the next panel.
    PetscInt GetPanelConnectionSize() const;

    // Return the column indices for the nonzero entries in the row of the 
    // off-diagonal block defined by node (x,y,z). The ordering of the 
    // off-diagonal block is constrained by the diagonal blocks being ordered 
    // in the fashion described at the top of this file. 
    void FormPanelConnections
    ( PetscInt x, PetscInt y, PetscInt z, 
      PetscInt thisPanelHeight, PetscInt nextPanelHeight,
      std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const;
};

} // namespace psp

// Implementation begins here

psp::FiniteDiffSweepingPC::FiniteDiffSweepingPC
( MPI_Comm comm,
  psp::FiniteDiffControl& control, 
  psp::SparseDirectSolver solver, 
  PetscInt numProcessRows, 
  PetscInt numProcessCols )
: _control(control), _solver(solver), _comm(comm),
  _numProcessRows(numProcessRows), _numProcessCols(numProcessCols), 
  _initialized(false)
{
    // We distribute our box using a 2d block distribution over each 
    // panel. The process grid is specified to be r x c.
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
}

psp::FiniteDiffSweepingPC::~FiniteDiffSweepingPC()
{
    if( _initialized )
        Destroy();
}

PetscInt
psp::GetLocalSize() const
{
    return _myXPortion*_myYPortion*_control.nz;
}

void
psp::FiniteDiffSweepingPC::Init( Vec& slowness, Mat& A )
{
    _slowness = &slowness;

    // Check that the slowness vector is the right length
    const PetscInt N = VecGetSize( *_slowness );
    if( N != _control.nx*_control.ny*_control.nz )
        throw std::logic_error("Slowness must be of size nx*ny*nz");

    // Check that we have the right-sizes local portion of the slowness
    const PetscInt localN = VecGetLocalSize( *_slowness );
    if( localN != _myXPortion*_myYPortion*_control.nz )
        throw std::runtime_error("Slowness is not properly distributed");

    //--------------------------------//
    // Form the full forward operator //
    //--------------------------------//
    const PetscInt localSize = GetLocalSize();
    // HERE

    //----------------------------------------------------------------//
    // Form each of the PML-padded diagonal blocks and then factor it //
    //----------------------------------------------------------------//
    // TODO: Consider allowing the artificial panel PML to be of a different 
    //       size than the outer fixed PML.
    const PetscInt nzMinusTopPML = _control.nz - _control.sizeOfPML;
    const PetscInt numPanels = 
      ( nzMinusTopPML>0 ? (nzMinusTopPML-1)/_control.sizeOfPML+1 : 0 );
    _paddedPanels.resize( numPanels );
    _paddedFactors.resize( numPanels );
    if( _solver == MUMPS_SYMMETRIC )
    {
        // Our solver supports symmetry, so only fill the lower triangles
        for( PetscInt panel=0; panel<numPanels; ++panel )
        {
            Mat& D = _paddedPanels[panel];
            MatCreate( _comm, &D );

            const PetscInt panelHeight = _control.sizeOfPML + 
              ( panel==numPanels-1 ? 
                _control.nz-(panel*_control.planesPerPanel+_control.sizeOfPML) :
                _control.planesPerPanel );
            const PetscInt bottomOfPanel = 
              ( panel==0 ? 0 : panel*_control.planesPerPanel );

            const PetscInt localSize = _myXPortion*_myYPortion*panelHeight;
            const PetscInt size = _control.nx*_control.ny*panelHeight;
            MatSetSizes( D, localSize, size, size, size );

            // SBAIJ is required for symmetry support
            MatSetType( D, MATMPISBAIJ ); 
            MatSetBlocksize( D, 1 );

            // Preallocate memory. This should be an upper bound for 7-point.
            // TODO: Generalize this for more general stencils.
            MatMPISBAIJSetPreallocation( D, 1, 3, PETSC_NULL, 3, PETSC_NULL );

            // Fill our portion of the distributed matrix.
            // TODO: Batch several rows together for each MatSetValues
            const PetscInt symmRowSize = GetSymmetricRowSize();
            std::vector<PetscScalar> row(symmRowSize);
            std::vector<PetscInt> colIndices(symmRowSize);
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( D, &iStart, &iEnd );
            for( PetscInt i=iStart; i<iEnd; ++i )
            {
                const PetscInt localX = (i-iStart) % _myXPortion;
                const PetscInt localY = ((i-iStart)/_myXPortion)%_myYPortion;
                const PetscInt localZ = (i-iStart) / (_myXPortion*_myYPortion);
                const PetscInt x = _myXOffset + localX;
                const PetscInt y = _myYOffset + localY;
                const PetscInt z = begOfPML + localZ;

                // Compute the left half of this row
                FormSymmetricRow
                ( _control.imagShift, x, y, z, 
                  bottomOfPanel, panelHeight, _control.sizeOfPML, 
                  row, colIndices );

                // Store the values into the distributed matrix
                MatSetValues
                ( D, &row[0], 1, &i, symmRowSize, &colIndices[0], 
                  INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _paddedFactors[panel];
            MatGetFactor( D, MAT_SOLVER_MUMPS, MAT_FACTOR_CHOLESKY, &F );
            MatFactorInfo cholInfo;
            cholInfo.fill = 3.0; // TODO: Tweak/expose this
            cholInfo.dtcol = 0.5; // TODO: Tweak/expose this
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            MatCholeskyFactorSymbolic( F, D, perm, &cholInfo );
            MatCholeskyFactorNumeric( F, D, &cholInfo );
        }
    }
    else // solver == MUMPS or solver == SUPERLU_DIST
    {
        // Our solver does not support symmetry, fill the entire matrices
        for( PetscInt panel=0; panel<numPanels; ++panel )
        {
            Mat& D = _paddedPanels[panel];
            MatCreate( _comm, &D );

            const PetscInt panelHeight = _control.sizeOfPML + 
              ( panel==numPanels-1 ? 
                _control.nz-(panel*_control.planesPerPanel+_control.sizeOfPML) :
                _control.planesPerPanel );
            const PetscInt bottomOfPanel = 
              ( panel==0 ? 0 : panel*_control.planesPerPanel );

            const PetscInt localSize = _myXPortion*_myYPortion*panelHeight;
            const PetscInt size = _control.nx*_control.ny*panelHeight;
            MatSetSizes( D, localSize, size, size, size );

            MatSetType( D, MATMPIAIJ );

            // Preallocate memory
            // TODO: Generalize this for more general stencils.
            MatMPIAIJSetPreallocation( D, 5, PETSC_NULL, 5, PETSC_NULL );

            // Fill our portion of the diagonal block
            // TODO: Batch several rows together for each MatSetValues
            const PetscInt rowSize = GetRowSize();
            std::vector<PetscScalar> row(rowSize);
            std::vector<PetscInt> colIndices(rowSize);
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( D, &iStart, &iEnd );
            for( PetscInt i=iStart; i<iEnd; ++i )
            {
                const PetscInt localX = (i-iStart) % _myXPortion;
                const PetscInt localY = ((i-iStart)/_myXPortion)%_myYPortion;
                const PetscInt localZ = (i-iStart) / (_myXPortion*_myYPortion);
                const PetscInt x = _myXOffset + localX;
                const PetscInt y = _myYOffset + localY;
                const PetscInt z = begOfPML + localZ;

                // Compute this entire row.
                FormRow
                ( _control.imagShift, x, y, z, 
                  bottomOfPanel, panelHeight, _control.sizeOfPML, 
                  row, colIndices );

                // Store the row in the distributed matrix
                MatSetValues
                ( D, &row[0], 1, &i, rowSize, &colIndices[0], INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _paddedFactors[panel];
            if( solver == MUMPS )
                MatGetFactor( D, MAT_SOLVER_MUMPS, MAT_FACTOR_LU, &F );
            else
                MatGetFactor( D, MAT_SOLVER_SUPERLU_DIST, MAT_FACTOR_LU, &F );
            MatFactorInfo luInfo;
            luInfo.fill = 3.0; // TODO: Tweak/expose this
            luInfo.dtcol = 0.5; // TODO: Tweak/expose this
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            MatLUFactorSymbolic( F, D, perm, &luInfo );
            MatLUFactorNumeric( F, D, &luInfo );
        }
    }

    //---------------------------------------//
    // Form the unpadded off-diagonal blocks //
    //---------------------------------------//
    _subdiagonalBlocks.resize( numPanels-1 );
    for( PetscInt panel=0; panel<numPanels-1; ++panel )
    {
        Mat& B = _subdiagonalBlocks[panel];
        MatCreate( _comm, &B );

        const PetscInt bottomOfPanel = 
          ( panel==0 ? 0 : _control.sizeOfPML + panel*_control.planesPerPanel );

        const PetscInt thisPanelHeight = _control.planesPerPanel;
        const PetscInt nextPanelHeight = 
            ( panel==numPanels-2 ? 
              _control.planesPerPanel : 
              _control.nz-(panel+1)*_control.planesPerPanel-_control.sizeOfPML 
            );
        const PetscInt blockHeight = _control.nx*_control.ny*thisPanelHeight;
        const PetscInt blockWidth = _control.nx*_control.ny*nextPanelHeight;
        const PetscInt localBlockHeight = 
            _myXPortion*_myYPortion*thisPanelHeight;
        MatSetSizes( B, localBlockHeight, blockWidth, blockHeight, blockWidth );

        MatSetType( B, MATMPIAIJ );

        // Preallocate memory (go ahead and use one diagonal + one off-diagonal)
        // even though this is a gross overestimate (roughly factor of 10).
        MatMPIAIJSetPreallocation( B, 1, PETSC_NULL, 1, PETSC_NULL );

        // Fill the connections between our nodes in the last xy plane of 
        // unpadded panel 'panel' and the first xy plane of unpadded panel 
        // 'panel+1'.
        //
        // TODO: Batch together several MatSetValues calls
        const PetscInt rowSize = GetPanelConnectionSize();
        std::vector<PetscScalar> subdiagRow(rowSize);
        std::vector<PetscInt> subdiagColIndices(rowSize);
        PetscInt iStart, iEnd;
        MatGetOwnershipRange( B, &iStart, &iEnd );
        const PetscInt iStartLastPlane = 
            iStart + (thisPanelHeight-1)*_myXPortion*_myYPortion;
        for( PetscInt i=iStartLastPlane; i<iEnd; ++i )
        {
            const PetscInt localX = (i-iStart) % _myXPortion;
            const PetscInt localY = ((i-iStart)/_myXPortion)%_myYPortion;
            const PetscInt localZ = thisPanelHeight-1;
            const PetscInt x = _myXOffset + localX;
            const PetscInt y = _myYOffset + localY;
            const PetscInt z = bottomOfPanel + localZ;

            // Compute this entire row.
            FormPanelConnections
            ( x, y, z, thisPanelHeight, nextPanelHeight,
              row, colIndices );

            // Store the row in the distributed matrix
            MatSetValues
            ( B, &row[0], 1, &i, rowSize, &colIndices[0], INSERT_VALUES );
        }
        MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY );
    }
    
    _initialized = true;
}

void
psp::FiniteDiffSweepingPC::Destroy()
{
    if( !_initialized )
    {
        throw std::logic_error
        ("Cannot destroy preconditioner without initializing");
    }
    PetscInt numPanels = _paddedPanels.size();
    for( std::size_t panel=0; panel<numPanels; ++panel )
    {
        MatDestroy( _paddedPanels[panel] );
        MatDestroy( _paddedFactors[panel] );
    }
    for( PetscInt panel=0; panel<numPanels-1; ++panel )
        MatDestroy( _subdiagonalBlocks[panel] );
    _initialized = false;
}

void
psp::FiniteDiffSweepingPC::Apply( Vec& x, Vec& y ) const
{
    // TODO: Write the sweeping preconditioner application on x, into y
}

PetscInt
psp::FiniteDiffSweepingPC::GetSymmetricRowSize()
{
    // TODO: Support more than just the 7-point stencil
    return 4;
}

void
psp::FiniteDiffSweepingPC::FormSymmetricRow
( PetscReal imagShift, 
  PetscInt x, PetscInt y, PetscInt z, 
  PetscInt bottomOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*panelHeight;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*panelHeight + 
        _myProcessCol*pillarSize;
    const PetscInt localX = x - _myXOffset;
    const PetscInt localY = y - _myYOffset;
    const PetscInt localZ = z - bottomOfPanel;
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

PetscInt
psp::FiniteDiffSweepingPC::GetRowSize()
{
    // TODO: Support more than just the 7-point stencil
    return 7;
}

void
psp::FiniteDiffSweepingPC::FormRow
( PetscReal imagShift,
  PetscInt x, PetscInt y, PetscInt z, 
  PetscInt bottomOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*panelHeight;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*panelHeight + 
        _myProcessCol*pillarSize;
    const PetscInt localX = x - _myXOffset;
    const PetscInt localY = y - _myYOffset;
    const PetscInt localZ = z - bottomOfPanel;
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

PetscInt
psp::FiniteDiffSweepingPC::GetPanelConnectionSize()
{
    // TODO: Support more than just the 7-point stencil
    return 1;
}

void
psp::FiniteDiffSweepingPC::FormPanelConnections
( PetscInt x, PetscInt y, PetscInt z,
  PetscInt thisPanelHeight, PetscInt nextPanelHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices )
{
    // Connection to (x,y,z+1)
    const PetscInt firstRowInNextPanel = 
        (_myXOffset+_myYOffset*_control.nx)*nextPanelHeight;
    colIndices[0] = 
        firstRowInNextPanel + (x-_myXOffset) + (y-_myYOffset)*_myXPortion;
    // TODO: Fill in finite diff approx of (x,y,z)'s connection to (x,y,z+1) 
    // into row[0]
}

#endif // PSP_FINITE_DIFF_SWEEPING_PC_HPP
