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

class FiniteDiffSweepingPC
{
    FiniteDiffControl _control;
    SparseDirectSolver _solver;

    Vec& _slowness;
    std::vector<Mat> _panelsPlusPML;

    MPI_Comm _comm;
    PetscInt _r; // number of rows in process grid
    PetscInt _c; // number of cols in process grid
    PetscInt _myProcessRow;
    PetscInt _myProcessCol;
    PetscInt _xChunkSize;
    PetscInt _yChunkSize;
    PetscInt _myXOffset;
    PetscInt _myYOffset;
    PetscInt _myXPortion;
    PetscInt _myYPortion;

public:
    FiniteDiffSweepingPC
    ( FiniteDiffControl& control, SparseDirectSolver solver, Vec slowness,
      PetscInt r, PetscInt c );
};

} // namespace psp

// Implementation begins here

psp::FiniteDiffSweepingPC::FiniteDiffSweepingPC
( psp::FiniteDiffControl& control, 
  psp::SparseDirectSolver solver, 
  Vec slowness,
  PetscInt r, PetscInt c )
: _control(control), _solver(solver), _slowness(slowness), _r(r), _c(c)
{
    // Check that the slowness vector is the right length
    const PetscInt N = VecGetSize( _slowness );
    if( N != _control.nx*_control.ny*_control.nz )
        throw std::logic_error("Slowness must be of size nx*ny*nz");

    // Ensure that we have the right number of local entries in the slowness
    // vector. We distribute our box using a 2d block distribution over each 
    // panel. The process grid is specified to be r x c.
    PetscObjectGetComm( _slowness, &_comm );
    PetscMPIInt p; MPI_Comm_size( _comm, &p );
    PetscMPIInt rank; MPI_Comm_rank( _comm, &rank );
    if( r*c != p )
        throw std::logic_error("Invalid process grid dimensions");
    _myProcessRow = rank / c;
    _myProcessCol = rank % c;
    _xChunkSize = _control.nx / c;
    _yChunkSize = _control.ny / r;
    _myXOffset = _myProcessCol*_xChunkSize;
    _myYOffset = _myProcessRow*_yChunkSize;
    _myXPortion = ( _myProcessCol==c-1 ? 
                    _xChunkSize : 
                    _xChunkSize+(_control.nx%_xChunkSize) );
    _myYPortion = ( _myProcessRow==r-1 ?
                    _yChunkSize :
                    _yChunkSize+(_control.ny%_yChunkSize) );
    const PetscInt localN = VecGetLocalSize( _slowness );
    if( localN != _myXPortion*_myYPortion*_control.nz )
        throw std::runtime_error("Slowness is not properly distributed");

    // Create a vector of Mat's
    const PetscInt nzMinusTopPML = _control.nz - _control.b;
    const PetscInt numPanels = 
        ( nzMinusTopPML>0 ? (nzMinusTopPML-1)/_control.b+1 : 0 );
    _panelsWithPML.resize( numPanels );
    for( PetscInt j=0; j<numPanels; ++j )
    {
        Mat& A = _panelsWithPML[j];
        MatCreate( _comm, &A );
    }
}

psp::FiniteDiffSweepingPC::Init()
{
    // Initialize each panel
    const PetscInt numPanels = _panelsWithPML.size();
    if( _solver == psp::MUMPS_SYMMETRIC )
    {
        // Our solver supports symmetry, so only fill the lower triangles
        for( PetscInt panel=0; panel<numPanels; ++panel )
        {
            Mat& A = _panelsWithPML[j];
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
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( A, &iStart, &iEnd );
            std::vector<PetscScalar> row(4);
            std::vector<PetscInt> colIndices(4);
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
                Form7PointSymmetricRow
                ( x, y, z, heightOffset, row, colIndices );

                MatSetValues
                ( A, &row[0], 1, &i, 4, &colIndices[0], INSERT_VALUES );
            }
            MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat F;
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
            Mat& A = _panelsWithPML[j];
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
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( A, &iStart, &iEnd );
            for( int i=iStart; i<iEnd; ++i )
            {

            }
            MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
        }
    }
}

psp::FiniteDiffSweepingPC::~FiniteDiffSweepingPC()
{
    for( std::size_t j=0; j<_panelsWithPML.size(); ++j )
        VecDestroy( _panelsWithPML[j] );
}

#endif // PSP_FINITE_DIFF_SWEEPING_PC_HPP
