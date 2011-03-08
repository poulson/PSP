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
    MPI_Comm _comm;
    PetscInt _rank;
    PetscInt _numProcesses;
    const PetscInt _numProcessRows;
    const PetscInt _numProcessCols;
    PetscInt _myProcessRow;
    PetscInt _myProcessCol;
    PetscInt _xChunkSize;
    PetscInt _yChunkSize;
    PetscInt _myXOffset;
    PetscInt _myYOffset;
    PetscInt _myXPortion;
    PetscInt _myYPortion;

    const FiniteDiffControl _control;
    const SparseDirectSolver _solver;

    const PetscInt _pmlSizeCubed;
    const PetscReal _hx;
    const PetscReal _hy;
    const PetscReal _hz;

    Vec* _slowness;
    std::vector<Mat> _paddedFactors;
    std::vector<Mat> _subdiagonalBlocks;

    bool _initialized;

    PetscScalar s1Inv( PetscInt x ) const;
    PetscScalar s2Inv( PetscInt y ) const;
    PetscScalar s3Inv( PetscInt z ) const;
    // Since the PML will often be pushed down the z direction, we will need
    // a means of specifying when the PML ends somewhere different than 
    // z=_control.pmlSize-1;
    PetscScalar s3Inv( PetscInt z, PetscInt endOfPml ) const;

public:
    FiniteDiffSweepingPC
    ( MPI_Comm comm, PetscInt numProcessRows, PetscInt numProcessCols, 
      FiniteDiffControl& control, SparseDirectSolver solver );

    ~FiniteDiffSweepingPC();

    PetscInt GetLocalSize() const;

    void Init( Vec& slowness, Mat& A );
    void Destroy();

    // Apply sweeping preconditioner to x, producing y
    void Apply( Vec& x, Vec& y ) const;

    // Return the maximum number of nonzeros in the left side of each row
    PetscInt GetSymmetricRowSize() const;

    // Return the indices and values for each nonzero in the left side of the
    // row defined by node (x,y,z) in the panel defined by its top z-location
    // (topOfPanel), its height (panelHeight), and its PML height on the top
    // (pmlHeight).
    //
    // In the cases where a connection is invalid, fill its index with -1 so 
    // that PETSc's MatSetValues will ignore it.
    void FormSymmetricRow
    ( PetscReal imagShift,
      PetscInt x, PetscInt y, PetscInt z, 
      PetscInt topOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
      std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const;

    // Return the maximum number of nonzeros in a row
    PetscInt GetRowSize() const;

    // Return the indices and values for each nonzero in the 
    // row defined by node (x,y,z) in the panel defined by its top z-location
    // (topOfPanel), its height (panelHeight), and its PML height at the top
    // (pmlHeight)
    //
    // In the cases where a connection is invalid, fill its index with -1 so 
    // that PETSc's MatSetValues will ignore it.
    void FormRow
    ( PetscReal imagShift,
      PetscInt x, PetscInt y, PetscInt z, 
      PetscInt topOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
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

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

// Private utility functions

inline PetscScalar
psp::FiniteDiffSweepingPC::s1Inv( PetscInt x ) const
{
    if( (x+1)<_control.pmlSize && _control.frontBC==PML )
    {
        const PetscInt delta = _control.pmlSize-(x+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hx*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( x>(_control.nx-_control.pmlSize) && _control.backBC==PML )
    {
        const PetscInt delta = x-(_control.nx-_control.pmlSize);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hx*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

inline PetscScalar
psp::FiniteDiffSweepingPC::s2Inv( PetscInt y ) const
{
    if( (y+1)<_control.pmlSize && _control.leftBC==PML )
    {
        const PetscInt delta = _control.pmlSize-(y+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hy*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( y >(_control.ny-_control.pmlSize) && _control.rightBC==PML )
    {
        const PetscInt delta = y-(_control.ny-_control.pmlSize);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hy*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

inline PetscScalar
psp::FiniteDiffSweepingPC::s3Inv( PetscInt z ) const
{
    if( (z+1)<_control.pmlSize )
    {
        const PetscInt delta = _control.pmlSize-(z+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( z>(_control.nz-_control.pmlSize) && _control.bottomBC==PML )
    {
        const PetscInt delta = z-(_control.nz-_control.pmlSize);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

inline PetscScalar
psp::FiniteDiffSweepingPC::s3Inv( PetscInt z, PetscInt endOfPml ) const
{
    if( z<endOfPml )
    {
        const PetscInt delta = endOfPml-z;
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( z>(_control.nz-_control.pmlSize) && _control.bottomBC==PML )
    {
        const PetscInt delta = z-(_control.nz-_control.pmlSize);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.C*delta*delta/(_pmlSizeCubed*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

// Public member functions

psp::FiniteDiffSweepingPC::FiniteDiffSweepingPC
( MPI_Comm comm, PetscInt numProcessRows, PetscInt numProcessCols,
  psp::FiniteDiffControl& control, psp::SparseDirectSolver solver )
: _comm(comm),
  _numProcessRows(numProcessRows), _numProcessCols(numProcessCols),
  _control(control), _solver(solver),
  _pmlSizeCubed(_control.pmlSize*_control.pmlSize*_control.pmlSize),
  _hx(control.wx/(control.nx-1)),
  _hy(control.wy/(control.ny-1)),
  _hz(control.wz/(control.nz-1)),
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

    //-----------------------------------//
    // Form the full forward operator, A //
    //-----------------------------------//
    {
        const PetscInt localSize = GetLocalSize();
        const PetscInt size = _control.nx*_control.ny*_control.nz;

        MatCreate( _comm, &A );
        MatSetSizes( A, localSize, localSize, size, size 0 );
        MatSetType( A, MATMPISBAIJ );
        MatSetBlocksize( D, 1 );
        // TODO: Generalize this step for more stencils.
        MatMPISBAIJSetPreallocation( A, 1, 3, PETSC_NULL, 3, PETSC_NULL );

        const PetscInt symmRowSize = GetSymmetricRowSize();
        std::vector<PetscScalar> row(symmRowSize);
        std::vector<PetscInt> colIndices(symmRowSize);
        PetscInt iStart, iEnd;
        MatGetOwnershipRange( A, &iStart, &iEnd );
        for( PetscInt i=iStart; i<iEnd; ++i )
        {
            const PetscInt localX = (i-iStart) % _myXPortion;
            const PetscInt localY = ((i-iStart)/_myXPortion) % _myYPortion;
            const PetscInt localZ = (i-iStart) / (_myXPortion*_myYPortion);
            const PetscInt x = _myXOffset + localX;
            const PetscInt y = _myYOffset + localY;
            const PetscInt z = localZ;

            // Compute the left half of this row
            FormSymmetricRow
            ( 0.0, x, y, z, 0, _control.nz, _control.pmlSize, row, colIndices );

            // Put this row into the distributed matrix
            MatSetValues
            ( A, &row[0], 1, &i, symmRowSize, &colIndices[0], INSERT_VALUES );
        }
        MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
    }

    //----------------------------------------------------------------//
    // Form each of the PML-padded diagonal blocks and then factor it //
    //----------------------------------------------------------------//
    // TODO: Consider allowing the artificial panel PML to be of a different 
    //       size than the outer fixed PML.
    const PetscInt nzMinusTopPml = _control.nz - (_control.pmlSize-1);
    const PetscInt numPanels = 
      ( nzMinusTopPml>0 ? (nzMinusTopPml-1)/_control.planesPerPanel+1 : 0 );
    _paddedFactors.resize( numPanels );
    if( _solver == MUMPS_SYMMETRIC )
    {
        // Our solver supports symmetry, so only fill the lower triangles
        for( PetscInt m=0; m<numPanels; ++m )
        {
            Mat D;
            MatCreate( _comm, &D );

            const PetscInt panelHeight = (_control.pmlSize-1) + 
              ( m==numPanels-1 ? 
                _control.nz-(m*_control.planesPerPanel+(_control.pmlSize-1)) :
                _control.planesPerPanel );
            const PetscInt topOfPanel = 
              ( m==0 ? 0 : (_control.pmlSize-1)+m*_control.planesPerPanel );

            const PetscInt localSize = _myXPortion*_myYPortion*panelHeight;
            const PetscInt size = _control.nx*_control.ny*panelHeight;
            MatSetSizes( D, localSize, localSize, size, size );

            // SBAIJ is required for symmetry support
            MatSetType( D, MATMPISBAIJ ); 
            MatSetBlocksize( D, 1 );

            // Preallocate memory. This should be an upper bound for 7-point.
            // TODO: Generalize this for more stencils.
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
                const PetscInt z = topOfPanel + localZ;

                // Compute the left half of this row
                FormSymmetricRow
                ( _control.imagShift, x, y, z, 
                  topOfPanel, panelHeight, _control.pmlSize, row, colIndices );

                // Store the values into the distributed matrix
                MatSetValues
                ( D, &row[0], 1, &i, symmRowSize, &colIndices[0], 
                  INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _paddedFactors[m];
            MatGetFactor( D, MAT_SOLVER_MUMPS, MAT_FACTOR_CHOLESKY, &F );
            MatFactorInfo cholInfo;
            cholInfo.fill = 3.0; // TODO: Tweak/expose this
            cholInfo.dtcol = 0.5; // TODO: Tweak/expose this
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            MatCholeskyFactorSymbolic( F, D, perm, &cholInfo );
            MatCholeskyFactorNumeric( F, D, &cholInfo );
            MatDestroy( D );
        }
    }
    else // solver == MUMPS or solver == SUPERLU_DIST
    {
        // Our solver does not support symmetry, fill the entire matrices
        for( PetscInt m=0; m<numPanels; ++m )
        {
            Mat D;
            MatCreate( _comm, &D );

            const PetscInt panelHeight = (_control.pmlSize-1) + 
              ( m==numPanels-1 ? 
                _control.nz-(m*_control.planesPerPanel+(_control.pmlSize-1)) :
                _control.planesPerPanel );
            const PetscInt topOfPanel = 
              ( m==0 ? 0 : (_control.pmlSize-1)+m*_control.planesPerPanel );

            const PetscInt localSize = _myXPortion*_myYPortion*panelHeight;
            const PetscInt size = _control.nx*_control.ny*panelHeight;
            MatSetSizes( D, localSize, localSize, size, size );

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
                const PetscInt z = topOfPanel + localZ;

                // Compute this entire row.
                FormRow
                ( _control.imagShift, x, y, z, 
                  topOfPanel, panelHeight, _control.pmlSize, row, colIndices );

                // Store the row in the distributed matrix
                MatSetValues
                ( D, &row[0], 1, &i, rowSize, &colIndices[0], INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _paddedFactors[m];
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
            MatDestroy( D );
        }
    }

    //---------------------------------------//
    // Form the unpadded off-diagonal blocks //
    //---------------------------------------//
    _offDiagBlocks.resize( numPanels-1 );
    for( PetscInt m=0; m<numPanels-1; ++m )
    {
        Mat& B = _offDiagBlocks[m];
        MatCreate( _comm, &B );

        const PetscInt topOfPanel = 
          ( m==0 ? 0 : (_control.pmlSize-1) + m*_control.planesPerPanel );

        const PetscInt thisPanelHeight = _control.planesPerPanel;
        const PetscInt nextPanelHeight = 
            ( m==numPanels-2 ? 
              _control.planesPerPanel : 
              _control.nz-(m+1)*_control.planesPerPanel-(_control.pmlSize-1) );
        const PetscInt blockHeight = _control.nx*_control.ny*thisPanelHeight;
        const PetscInt blockWidth = _control.nx*_control.ny*nextPanelHeight;
        const PetscInt localBlockHeight = 
            _myXPortion*_myYPortion*thisPanelHeight;
        const PetscInt localBlockWidth = 
            _myXPortion*_myYPortion*nextPanelHeight;
        MatSetSizes
        ( B, localBlockHeight, localBlockWidth, blockHeight, blockWidth );

        MatSetType( B, MATMPIAIJ );

        // Preallocate memory (go ahead and use one diagonal + one off-diagonal)
        // even though this is a gross overestimate (roughly factor of 10).
        //
        // TODO: Generalize beyond 7-point stencil
        MatMPIAIJSetPreallocation( B, 1, PETSC_NULL, 1, PETSC_NULL );

        // Fill the connections between our nodes in the last xy plane of 
        // unpadded panel 'm' and the first xy plane of unpadded panel 
        // 'm+1'.
        //
        // TODO: Batch together several MatSetValues calls
        const PetscInt rowSize = GetPanelConnectionSize();
        std::vector<PetscScalar> offDiagRow(rowSize);
        std::vector<PetscInt> offDiagColIndices(rowSize);
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
            const PetscInt z = topOfPanel + localZ;

            // Compute this entire row.
            FormPanelConnections
            ( x, y, z, thisPanelHeight, nextPanelHeight, row, colIndices );

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
    const PetscInt numPanels = _paddedFactors.size();
    for( PetscInt m=0; m<numPanels; ++ml )
        MatDestroy( _paddedFactors[m] );
    for( PetscInt m=0; m<numPanels-1; ++m )
        MatDestroy( _offDiagBlocks[m] );
    _initialized = false;
}

void
psp::FiniteDiffSweepingPC::Apply( Vec& f, Vec& u ) const
{
    const PetscInt numPanels = _paddedFactors.size();
    std::vector<Vec> uPanels(numPanels);

    // for m=0,1,...,n-1
    //   u_m = f_m
    //
    // We first grab a pointer to our local data in f. It points to a rod of 
    // (x,y) dimensions (_myXPortion,_myYPortion) and z dimension _control.nz, 
    // with the storage x-y-z major (i.e., x dimension is contiguous, as are xy 
    // planes of our local data). This allows us to easily memcpy chunks of 
    // our process's rod of data.
    PetscScalar* fLocalData;
    VecGetArray( f, &fLocalData );
    for( PetscInt m=0; m<numPanels; ++m )
    {
        // A_{m,m+1}'s 'left' vector distribution should match u_m's, while 
        // its right vector solution should match u_{m+1}
        MatGetVecs( _offDiagBlocks[m], PETSC_NULL, &uPanels[m] );

        // Get a pointer to this panel solution so that we may memcpy the
        // appropriate chunk of f into it
        PetscScalar* uPanelLocalData;
        VecGetArray( uPanels[m], &uPanelLocalData );

        // Copy this panel's RHS, "f_m", into u[m]
        const PetscInt zOffset = 
            ( m==0 ? 0 : (_control.pmlSize-1)+m*_control.planesPerPanel );
        const PetscInt fOffset = _myXPortion*_myYPortion*zOffset;
        const uPanelLocalSize = VecGetLocalSize( uPanels[m] );
        std::memcpy
        ( uPanelLocalData, &fLocalData[fOffset], 
          uPanelLocalSize*sizeof(PetscScalar) );

        // Give this u panel's pointer back
        VecRestoreArray( uPanels[m], &uPanelLocalData );
    }
    // Give back the pointer to f
    VecRestoreArray( f, &fLocalData );

    // for m=0,...,n-2
    //   u_{m+1} = u_{m+1} - A_{m+1,m}(T_m u_m)
    // where, in our case, we are storing A_{m,m+1} and instead apply
    //   u_{m+1} = u_{m+1} - A_{m,m+1}^T (T_m u_m).
    // It is important to recognize that the (T_m u_m) step involves embedding
    // u_m into a domain padded with PML, solving using the sparse direct 
    // factorization, and then extracting the solution out of the region we 
    // embedded into.
    //
    // Since we are already forming T_m u_m, which is the entire computation of 
    // the next step:
    //   for m=0,...,n-1,
    //     u_m = T_m u_m
    // we will go ahead and fuse these two steps.
    for( PetscInt m=0; m<numPanels; ++m )
    {
        // Grab a pointer to our portion of u_m. Also grab the amount of data.
        PetscScalar* uPanelLocalData;
        VecGetArray( uPanels[m], &uPanelLocalData );
        const PetscInt uPanelLocalSize = VecGetLocalSize( uPanels[m] );

        // Create a temporary buffer that is conformal with our factored 
        // diagonal block. Start by initializing it to zero
        Vec paddedPanelRHS, paddedPanelSol;    
        MatGetVecs( _paddedFactors[m], &paddedPanelRHS, PETSC_NULL );
        VecDuplicate( paddedPanelRHS, &paddedPanelSol );

        if( m == 0 )
        {
            // The first panel is already padded, so it fills the entire space.
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( paddedPanelRHSLocalData, uPanelLocalData, 
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization.
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the entire solution into u_m 
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( uPanelLocalData, paddedPanelRHSLocalData,
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }
        else
        {
            const PetscInt pmlSize = _control.pmlSize;
            const PetscInt paddedOffset = _myXPortion*_myYPortion*(pmlSize-1);

            // Copy our panel's data into the back of the zero-padded buffer
            VecSet( paddedPanelRHS, 0.0 );
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( &paddedPanelRHSLocalData[paddedOffset], uPanelLocalData,
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the relevant portion of the solution and store in u_m
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( uPanelLocalData, &paddedPanelSolLocalData[paddedOffset],
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }

        // Every step but the last must update the next panel solution
        if( m != numPanels-1 )
        {
            // Form A_{m,m+1}^T u_m into a work buffer
            Vec work;
            VecDuplicate( uPanels[m+1], work );
            MatMultTranspose( _offDiagBlocks[m], uPanels[m], work );

            // Substract the work vector from u_{m+1}
            VecAXPY( uPanels[m+1], -1.0, work );

            // Destroy the work vector
            VecDestroy( work );
        }

        // Destroy the padded panel vectors
        VecDestroy( paddedPanelRHS );
        VecDestroy( paddedPanelSol );

        // Release u_m's pointer
        VecRestoreArray( uPanels[m], &uPanelLocalData );
    }

    // for m=n-2,...,0
    //   u_m = u_m - T_m(A_{m,m+1}u_{m+1})
    for( PetscInt m=numPanels-2; m>=0; --m )
    {
        // Form A_{m,m+1} u_{m+1} into a work vector
        Vec work;
        MatGetVecs( _offDiagBlocks[m+1], PETSC_NULL, work );
        MatMult( _offDiagBlocks[m+1], uPanels[m+1], work );

        // Grab a pointer to our portion of 'work'. 
        // Also grab the amount of data.
        PetscScalar* workLocalData;
        VecGetArray( work, &workLocalData );
        const PetscInt workLocalSize = VecGetLocalSize( work );

        // Create a temporary buffer that is conformal with our factored 
        // diagonal block. Start by initializing it to zero
        Vec paddedPanelRHS, paddedPanelSol;    
        MatGetVecs( _paddedFactors[m], &paddedPanelRHS, PETSC_NULL );
        VecDuplicate( paddedPanelRHS, &paddedPanelSol );

        if( m == 0 )
        {
            // The first panel is already padded, so it fills the entire space.
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( paddedPanelRHSLocalData, workLocalData, 
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization.
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the entire solution into work
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( workLocalData, paddedPanelRHSLocalData,
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }
        else
        {
            const PetscInt pmlSize = _control.pmlSize;
            const PetscInt paddedOffset = _myXPortion*_myYPortion*(pmlSize-1);

            // Copy our panel's data into the back of the zero-padded buffer
            VecSet( paddedPanelRHS, 0.0 );
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( &paddedPanelRHSLocalData[paddedOffset], workLocalData,
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the relevant portion of the solution and store in u_m
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( workLocalData, &paddedPanelSolLocalData[paddedOffset],
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }

        // Destroy the padded panel vectors
        VecDestroy( paddedPanelRHS );
        VecDestroy( paddedPanelSol );

        // Substract the work vector from u_m
        VecAXPY( uPanels[m], -1.0, work );

        // Destroy the work vector
        VecDestroy( work );
    }

    // Store the panel solutions into the full vector, u, and destroy the 
    // panel solutions as we go
    PetscScalar* uLocalData;
    VecGetArray( u, &uLocalData );
    for( PetscInt m=0; m<numPanels; ++m )
    {
        // Get a pointer to this panel solution so that we may memcpy from it
        PetscScalar* uPanelLocalData;
        VecGetArray( uPanels[m], &uPanelLocalData );

        // Copy the panel solution into the full vector
        const PetscInt zOffset = 
            ( m==0 ? 0 : (_control.pmlSize-1)+m*_control.planesPerPanel );
        const PetscInt fOffset = _myXPortion*_myYPortion*zOffset;
        const uPanelLocalSize = VecGetLocalSize( uPanels[m] );
        std::memcpy
        ( &uLocalData[fOffset], uPanelLocalData,
          uPanelLocalSize*sizeof(PetscScalar) );

        // Give this u panel's pointer back
        VecRestoreArray( uPanels[m], &uPanelLocalData );

        // Destroy the panel solution
        VecDestroy( uPanels[m] );
    }
    // Give the pointer to the full solution vector back
    VecRestoreArray( u, &uLocalData );
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
  PetscInt topOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*panelHeight;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*panelHeight + 
        _myProcessCol*pillarSize;
    const PetscInt localX = x - _myXOffset;
    const PetscInt localY = y - _myYOffset;
    const PetscInt localZ = z - topOfPanel;
    const PetscInt rowIdx = myOffset + 
        localX + localY*_myXPortion + localZ*_myXPortion*_myYPortion;

    // HERE: We need to carefully consider what the 'size' of the PML is. 
    //       It seems to me that adding 1 layer of PML does not change the 
    //       size of the system since the boundary nodes of a finite difference
    //       system are not explicitly represented. Thus b layers of PML 
    //       adds on (b-1) degrees of freedom on each side of the domain.
    //
    // Evaluate all of our inverse s functions
    const PetscScalar s1InvL = s1Inv(x-1);
    const PetscScalar s1InvM = s1Inv(x);
    const PetscScalar s1InvR = s1Inv(x+1);
    const PetscScalar s2InvL = s2Inv(y-1);
    const PetscScalar s2InvM = s2Inv(y);
    const PetscScalar s2InvR = s2Inv(y+1);
    const PetscScalar s3InvL = s3Inv(z-1,topOfPanel+pmlHeight-1);
    const PetscScalar s3InvM = s3Inv(z,topOfPanel+pmlHeight-1);
    const PetscScalar s3InvR = s3Inv(z+1,topOfPanel+pmlHeight-1);
    // Compute all of the x terms
    const PetscScalar xTempL = s2InvM*s3InvM/s1InvL;
    const PetscScalar xTempM = s2InvM*s3InvM/s1InvM;
    const PetscScalar xTempR = s2InvM*s3InvM/s1InvR;
    const PetscScalar xTermL = (xTempL+xTempM)/(2*_hx*_hx);
    const PetscScalar xTermR = (xTempR+xTempM)/(2*_hx*_hx);
    // Compute all of the y terms
    const PetscScalar yTempL = s1InvM*s3InvM/s2InvL;
    const PetscScalar yTempM = s1InvM*s3InvM/s2InvM;
    const PetscScalar yTempR = s1InvM*s3InvM/s2InvR;
    const PetscScalar yTermL = (yTempL+yTempM)/(2*_hy*_hy);
    const PetscScalar yTermR = (yTempR+yTempM)/(2*_hy*_hy);
    // Compute all of the z terms
    const PetscScalar zTempL = s1InvM*s2InvM/s3InvL;
    const PetscScalar zTempM = s1InvM*s2InvM/s3InvM;
    const PetscScalar zTempR = s1InvM*s2InvM/s3InvR;
    const PetscScalar zTermL = (zTempL+zTempM)/(2*_hz*_hz);
    const PetscScalar zTermR = (zTempR+zTempM)/(2*_hz*_hz);
    // PAUSED HERE...

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
  PetscInt topOfPanel, PetscInt panelHeight, PetscInt pmlHeight,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*panelHeight;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*panelHeight + 
        _myProcessCol*pillarSize;
    const PetscInt localX = x - _myXOffset;
    const PetscInt localY = y - _myYOffset;
    const PetscInt localZ = z - topOfPanel;
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
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    // TODO
}

#endif // PSP_FINITE_DIFF_SWEEPING_PC_HPP
