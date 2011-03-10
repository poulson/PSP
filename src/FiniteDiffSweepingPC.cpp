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

psp::FiniteDiffSweepingPC::FiniteDiffSweepingPC
( MPI_Comm comm, PetscInt numProcessRows, PetscInt numProcessCols,
  psp::FiniteDiffControl& control, psp::SparseDirectSolver solver )
: _comm(comm),
  _numProcessRows(numProcessRows), 
  _numProcessCols(numProcessCols),
  _control(control), _solver(solver),
  _hx(control.wx/(control.nx+1)),
  _hy(control.wy/(control.ny+1)),
  _hz(control.wz/(control.nz+1)),
  _bx(control.etax/_hx),
  _by(control.etay/_hy),
  _bz(control.etaz/_hz),
  _bzPadded(std::ceil(control.etaz/_hz)),
  _initialized(false)
{
    // We distribute our box using a 2d block distribution over each 
    // panel. The process grid is specified to be r x c.
    MPI_Comm_size( _comm, &_numProcesses );
    MPI_Comm_rank( _comm, &_rank );
    if( _numProcessRows*_numProcessCols != _numProcesses )
        throw std::logic_error("Invalid process grid dimensions");
    _myProcessRow = _rank / _numProcessCols;
    _myProcessCol = _rank % _numProcessCols;
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
psp::FiniteDiffSweepingPC::GetLocalSize() const
{
    return _myXPortion*_myYPortion*_control.nz;
}

void
psp::FiniteDiffSweepingPC::Init( Vec& slowness, Mat& A )
{
    _slowness = &slowness;
    VecGetArray( *_slowness, &_localSlownessData );

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
        MatSetSizes( A, localSize, localSize, size, size );
        MatSetType( A, MATMPISBAIJ );
        MatSetBlockSize( A, 1 );
        // TODO: Generalize this step for more stencils.
        MatMPISBAIJSetPreallocation( A, 1, 3, PETSC_NULL, 3, PETSC_NULL );

        const PetscInt symmRowSize = GetSymmetricRowSize();
        std::vector<PetscScalar> nonzeros(symmRowSize);
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

            // Compute the right half of this row
            FormSymmetricRow
            ( 0.0, x, y, z, 0, _control.nz, _control.etaz, 
              nonzeros, colIndices );

            // Put this row into the distributed matrix
            MatSetValues
            ( A, 1, &i, symmRowSize, &colIndices[0], &nonzeros[0],
              INSERT_VALUES );
        }
        MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
    }

    //----------------------------------------------------------------//
    // Form each of the PML-padded diagonal blocks and then factor it //
    //----------------------------------------------------------------//
    // TODO: Consider allowing the artificial panel PML to be of a different 
    //       size than the outer fixed PML.
    const PetscInt nzMinusTopPml = _control.nz - (_bzPadded-1);
    const PetscInt numPanels = 
      ( nzMinusTopPml>0 ? (nzMinusTopPml-1)/_control.planesPerPanel+1 : 1 );
    _paddedFactors.resize( numPanels );
    if( _solver == MUMPS_SYMMETRIC )
    {
        // Our solver supports symmetry, so only fill the lower triangles
        for( PetscInt m=0; m<numPanels; ++m )
        {
            Mat D;
            MatCreate( _comm, &D );

            PetscInt zOffset, zSize;
            PetscReal pmlSize;
            if( m == 0 )
            {
                zOffset = 0;
                zSize = 
                    std::min(_control.nz,(_bzPadded-1)+_control.planesPerPanel);
                pmlSize = _bz;
            }
            else if( m == numPanels-1 )
            {
                zOffset = m*_control.planesPerPanel;
                zSize = _control.nz - (m*_control.planesPerPanel+(_bzPadded-1));
                pmlSize = _bzPadded;
            }
            else
            {
                zOffset = m*_control.planesPerPanel;
                zSize = (_bzPadded-1) + _control.planesPerPanel;
                pmlSize = _bzPadded;
            }

            const PetscInt localSize = _myXPortion*_myYPortion*zSize;
            const PetscInt size = _control.nx*_control.ny*zSize;
            MatSetSizes( D, localSize, localSize, size, size );

            // SBAIJ is required for symmetry support
            MatSetType( D, MATMPISBAIJ ); 
            MatSetBlockSize( D, 1 );

            // Preallocate memory. This should be an upper bound for 7-point.
            // TODO: Generalize this for more stencils.
            MatMPISBAIJSetPreallocation( D, 1, 3, PETSC_NULL, 3, PETSC_NULL );

            // Fill our portion of the distributed matrix.
            // TODO: Batch several rows together for each MatSetValues
            const PetscInt symmRowSize = GetSymmetricRowSize();
            std::vector<PetscScalar> nonzeros(symmRowSize);
            std::vector<PetscInt> colIndices(symmRowSize);
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( D, &iStart, &iEnd );
            for( PetscInt i=iStart; i<iEnd; ++i )
            {
                const PetscInt xLocal = (i-iStart) % _myXPortion;
                const PetscInt yLocal = ((i-iStart)/_myXPortion)%_myYPortion;
                const PetscInt zLocal = (i-iStart) / (_myXPortion*_myYPortion);
                const PetscInt x = _myXOffset + xLocal;
                const PetscInt y = _myYOffset + yLocal;
                const PetscInt z = zOffset + zLocal;

                // Compute the right half of this row
                FormSymmetricRow
                ( _control.imagShift, x, y, z, 
                  zOffset, zSize, pmlSize, nonzeros, colIndices );

                // Store the values into the distributed matrix
                MatSetValues
                ( D, 1, &i, symmRowSize, &colIndices[0], &nonzeros[0],
                  INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _paddedFactors[m];
            MatGetFactor( D, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &F );
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
    else // _solver == MUMPS or _solver == SUPERLU_DIST
    {
        // Our solver does not support symmetry, fill the entire matrices
        for( PetscInt m=0; m<numPanels; ++m )
        {
            Mat D;
            MatCreate( _comm, &D );

            PetscInt zOffset, zSize;
            PetscReal pmlSize;
            if( m == 0 )
            {
                zOffset = 0;
                zSize = 
                    std::min(_control.nz,(_bzPadded-1)+_control.planesPerPanel);
                pmlSize = _bz;
            }
            else if( m == numPanels-1 )
            {
                zOffset = m*_control.planesPerPanel;
                zSize = _control.nz - (m*_control.planesPerPanel+(_bzPadded-1));
                pmlSize = _bzPadded;
            }
            else
            {
                zOffset = m*_control.planesPerPanel;
                zSize = (_bzPadded-1) + _control.planesPerPanel;
                pmlSize = _bzPadded;
            }

            const PetscInt localSize = _myXPortion*_myYPortion*zSize;
            const PetscInt size = _control.nx*_control.ny*zSize;
            MatSetSizes( D, localSize, localSize, size, size );

            MatSetType( D, MATMPIAIJ );

            // Preallocate memory
            // TODO: Generalize this for more general stencils.
            MatMPIAIJSetPreallocation( D, 5, PETSC_NULL, 5, PETSC_NULL );

            // Fill our portion of the diagonal block
            // TODO: Batch several rows together for each MatSetValues
            const PetscInt rowSize = GetRowSize();
            std::vector<PetscScalar> nonzeros(rowSize);
            std::vector<PetscInt> colIndices(rowSize);
            PetscInt iStart, iEnd;
            MatGetOwnershipRange( D, &iStart, &iEnd );
            for( PetscInt i=iStart; i<iEnd; ++i )
            {
                const PetscInt xLocal = (i-iStart) % _myXPortion;
                const PetscInt yLocal = ((i-iStart)/_myXPortion)%_myYPortion;
                const PetscInt zLocal = (i-iStart) / (_myXPortion*_myYPortion);
                const PetscInt x = _myXOffset + xLocal;
                const PetscInt y = _myYOffset + yLocal;
                const PetscInt z = zOffset + zLocal;

                // Compute this entire row.
                FormRow
                ( _control.imagShift, x, y, z, 
                  zOffset, zSize, pmlSize, nonzeros, colIndices );

                // Store the row in the distributed matrix
                MatSetValues
                ( D, 1, &i, rowSize, &colIndices[0], &nonzeros[0],
                  INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );

            // Factor the matrix
            Mat& F = _paddedFactors[m];
            if( _solver == MUMPS )
                MatGetFactor( D, MATSOLVERMUMPS, MAT_FACTOR_LU, &F );
            else
                MatGetFactor( D, MATSOLVERSUPERLU_DIST, MAT_FACTOR_LU, &F );
            MatFactorInfo luInfo;
            luInfo.fill = 3.0; // TODO: Tweak/expose this
            luInfo.dtcol = 0.5; // TODO: Tweak/expose this
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            MatLUFactorSymbolic( F, D, perm, perm, &luInfo );
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

        PetscInt zOffset, zSize, zSizeNext;
        if( m == 0 )
        {
            zOffset = 0;
            zSize = 
                std::min(_control.nz,(_bzPadded-1)+_control.planesPerPanel);
        }
        else if( m == numPanels-1 )
        {
            zOffset = m*_control.planesPerPanel + (_bzPadded-1);
            zSize = _control.nz - zOffset;
        }
        else
        {
            zOffset = m*_control.planesPerPanel + (_bzPadded-1);
            zSize = _control.planesPerPanel;
        }
        if( m == numPanels-2 )
        {
            zSizeNext = 
                _control.nz - ((m+1)*_control.planesPerPanel+(_bzPadded-1));
        }
        else
        {
            zSizeNext = _control.planesPerPanel;
        }

        const PetscInt matrixHeight = _control.nx*_control.ny*zSize;
        const PetscInt matrixWidth = _control.nx*_control.ny*zSizeNext;
        const PetscInt localMatrixHeight = _myXPortion*_myYPortion*zSize;
        const PetscInt localMatrixWidth = _myXPortion*_myYPortion*zSizeNext;
        MatSetSizes
        ( B, localMatrixHeight, localMatrixWidth, matrixHeight, matrixWidth );

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
        std::vector<PetscScalar> nonzeros(rowSize);
        std::vector<PetscInt> colIndices(rowSize);
        PetscInt iStart, iEnd;
        MatGetOwnershipRange( B, &iStart, &iEnd );
        const PetscInt iStartLastPlane = 
            iStart + (zSize-1)*_myXPortion*_myYPortion;
        for( PetscInt i=iStartLastPlane; i<iEnd; ++i )
        {
            const PetscInt xLocal = (i-iStart) % _myXPortion;
            const PetscInt yLocal = ((i-iStart)/_myXPortion)%_myYPortion;
            const PetscInt zLocal = zSize-1;
            const PetscInt x = _myXOffset + xLocal;
            const PetscInt y = _myYOffset + yLocal;
            const PetscInt z = zOffset + zLocal;

            // Compute this entire row.
            FormPanelConnections
            ( x, y, z, zSize, zSizeNext, i, nonzeros, colIndices );

            // Store the row in the distributed matrix
            MatSetValues
            ( B, 1, &i, rowSize, &colIndices[0], &nonzeros[0], INSERT_VALUES );
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
    for( PetscInt m=0; m<numPanels; ++m )
        MatDestroy( _paddedFactors[m] );
    for( PetscInt m=0; m<numPanels-1; ++m )
        MatDestroy( _offDiagBlocks[m] );
    VecRestoreArray( *_slowness, &_localSlownessData );
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
            ( m==0 ? 0 : m*_control.planesPerPanel + (_bzPadded-1) );
        const PetscInt fOffset = _myXPortion*_myYPortion*zOffset;
        const PetscInt uPanelLocalSize = VecGetLocalSize( uPanels[m] );
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
            const PetscInt paddingSize = _myXPortion*_myYPortion*(_bzPadded-1);

            // Copy our panel's data into the back of the zero-padded buffer
            VecSet( paddedPanelRHS, 0.0 );
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( &paddedPanelRHSLocalData[paddingSize], uPanelLocalData,
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the relevant portion of the solution and store in u_m
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( uPanelLocalData, &paddedPanelSolLocalData[paddingSize],
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }

        // Every step but the last must update the next panel solution
        if( m != numPanels-1 )
        {
            // Form A_{m,m+1}^T u_m into a work buffer
            Vec work;
            VecDuplicate( uPanels[m+1], &work );
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
        MatGetVecs( _offDiagBlocks[m+1], PETSC_NULL, &work );
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
            const PetscInt paddingSize = _myXPortion*_myYPortion*(_bzPadded-1);

            // Copy our panel's data into the back of the zero-padded buffer
            VecSet( paddedPanelRHS, 0.0 );
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( &paddedPanelRHSLocalData[paddingSize], workLocalData,
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the relevant portion of the solution and store in u_m
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( workLocalData, &paddedPanelSolLocalData[paddingSize],
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
            ( m==0 ? 0 : m*_control.planesPerPanel + (_bzPadded-1) );
        const PetscInt fOffset = _myXPortion*_myYPortion*zOffset;
        const PetscInt uPanelLocalSize = VecGetLocalSize( uPanels[m] );
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
psp::FiniteDiffSweepingPC::GetSymmetricRowSize() const
{
    // TODO: Support more than just the 7-point stencil
    return 4;
}

void
psp::FiniteDiffSweepingPC::FormSymmetricRow
( PetscReal imagShift, 
  PetscInt x, PetscInt y, PetscInt z, 
  PetscInt zOffset, PetscInt zSize, PetscReal pmlSize,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*zSize;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*zSize + 
        _myProcessCol*pillarSize;
    const PetscInt xLocal = x - _myXOffset;
    const PetscInt yLocal = y - _myYOffset;
    const PetscInt zLocal = z - zOffset;
    const PetscInt rowIdx = myOffset + 
        xLocal + yLocal*_myXPortion + zLocal*_myXPortion*_myYPortion;

    // Evaluate all of our inverse s functions
    const PetscScalar s1InvL = s1Inv(x-1);
    const PetscScalar s1InvM = s1Inv(x);
    const PetscScalar s1InvR = s1Inv(x+1);
    const PetscScalar s2InvL = s2Inv(y-1);
    const PetscScalar s2InvM = s2Inv(y);
    const PetscScalar s2InvR = s2Inv(y+1);
    const PetscScalar s3InvL = s3InvArtificial(z-1,zOffset,pmlSize);
    const PetscScalar s3InvM = s3InvArtificial(z,zOffset,pmlSize);
    const PetscScalar s3InvR = s3InvArtificial(z+1,zOffset,pmlSize);
    // Compute all of the x-shifted terms
    const PetscScalar xTempL = s2InvM*s3InvM/s1InvL;
    const PetscScalar xTempM = s2InvM*s3InvM/s1InvM;
    const PetscScalar xTempR = s2InvM*s3InvM/s1InvR;
    const PetscScalar xTermL = (xTempL+xTempM)/(2*_hx*_hx);
    const PetscScalar xTermR = (xTempR+xTempM)/(2*_hx*_hx);
    // Compute all of the y-shifted terms
    const PetscScalar yTempL = s1InvM*s3InvM/s2InvL;
    const PetscScalar yTempM = s1InvM*s3InvM/s2InvM;
    const PetscScalar yTempR = s1InvM*s3InvM/s2InvR;
    const PetscScalar yTermL = (yTempL+yTempM)/(2*_hy*_hy);
    const PetscScalar yTermR = (yTempR+yTempM)/(2*_hy*_hy);
    // Compute all of the z-shifted terms
    const PetscScalar zTempL = s1InvM*s2InvM/s3InvL;
    const PetscScalar zTempM = s1InvM*s2InvM/s3InvM;
    const PetscScalar zTempR = s1InvM*s2InvM/s3InvR;
    const PetscScalar zTermL = (zTempL+zTempM)/(2*_hz*_hz);
    const PetscScalar zTermR = (zTempR+zTempM)/(2*_hz*_hz);
    // Compute the center term
    const PetscScalar shiftedOmega = 
        std::complex<PetscReal>(_control.omega,imagShift);
    const PetscScalar alpha = 
        _localSlownessData[xLocal+_myXPortion*yLocal+_myXPortion*_myYPortion*z];
    const PetscScalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR)+
        (shiftedOmega*alpha)*(shiftedOmega*alpha)/(xTempM*yTempM*zTempM);

    // Fill in value and local index for the diagonal entry in this panel + PML
    PetscInt entry = 0;
    colIndices[entry] = rowIdx;
    row[entry] = centerTerm;
    ++entry;

    // Back connection to (x+1,y,z)
    if( x != _control.nx-1 )
    {
        if( xLocal != _myXPortion-1 )
        {
            colIndices[entry] = rowIdx+1;
        }
        else
        {
            const PetscInt backProcessOffset = myOffset + pillarSize;
            const PetscInt backProcessXPortion = 
                std::min( _xChunkSize, _control.nx-(_myXOffset+_xChunkSize) );
            colIndices[entry] = backProcessOffset + yLocal*backProcessXPortion +
                zLocal*backProcessXPortion*_myYPortion;
        }
        row[entry] = xTermR;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Right connection to (x,y+1,z)
    if( y != _control.ny )
    {
        if( yLocal != _myYPortion )
        {
            colIndices[entry] = rowIdx + _myXPortion;
        }
        else
        {
            const PetscInt rightProcessYPortion =
                std::min( _yChunkSize, _control.ny-(_myYOffset+_yChunkSize) );
            const PetscInt rightProcessOffset = 
                myOffset + _control.nx*rightProcessYPortion*zSize;
            colIndices[entry] = rightProcessOffset + xLocal + 
                zLocal*_xChunkSize*rightProcessYPortion;
        }
        row[entry] = yTermR;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Bottom connection to (x,y,z+1)
    if( zLocal != zSize-1 )
    {
        colIndices[entry] = rowIdx + _myXPortion*_myYPortion;
        row[entry] = zTermR;
    }
    else
    {
        colIndices[entry] = -1;
    }
}

PetscInt
psp::FiniteDiffSweepingPC::GetRowSize() const
{
    // TODO: Support more than just the 7-point stencil
    return 7;
}

void
psp::FiniteDiffSweepingPC::FormRow
( PetscReal imagShift,
  PetscInt x, PetscInt y, PetscInt z, 
  PetscInt zOffset, PetscInt zSize, PetscReal pmlSize,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt pillarSize = _xChunkSize*_yChunkSize*zSize;

    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*zSize + 
        _myProcessCol*pillarSize;
    const PetscInt xLocal = x - _myXOffset;
    const PetscInt yLocal = y - _myYOffset;
    const PetscInt zLocal = z - zOffset;
    const PetscInt rowIdx = myOffset + 
        xLocal + yLocal*_myXPortion + zLocal*_myXPortion*_myYPortion;

    // Evaluate all of our inverse s functions
    const PetscScalar s1InvL = s1Inv(x-1);
    const PetscScalar s1InvM = s1Inv(x);
    const PetscScalar s1InvR = s1Inv(x+1);
    const PetscScalar s2InvL = s2Inv(y-1);
    const PetscScalar s2InvM = s2Inv(y);
    const PetscScalar s2InvR = s2Inv(y+1);
    const PetscScalar s3InvL = s3InvArtificial(z-1,zOffset,pmlSize);
    const PetscScalar s3InvM = s3InvArtificial(z,zOffset,pmlSize);
    const PetscScalar s3InvR = s3InvArtificial(z+1,zOffset,pmlSize);
    // Compute all of the x-shifted terms
    const PetscScalar xTempL = s2InvM*s3InvM/s1InvL;
    const PetscScalar xTempM = s2InvM*s3InvM/s1InvM;
    const PetscScalar xTempR = s2InvM*s3InvM/s1InvR;
    const PetscScalar xTermL = (xTempL+xTempM)/(2*_hx*_hx);
    const PetscScalar xTermR = (xTempR+xTempM)/(2*_hx*_hx);
    // Compute all of the y-shifted terms
    const PetscScalar yTempL = s1InvM*s3InvM/s2InvL;
    const PetscScalar yTempM = s1InvM*s3InvM/s2InvM;
    const PetscScalar yTempR = s1InvM*s3InvM/s2InvR;
    const PetscScalar yTermL = (yTempL+yTempM)/(2*_hy*_hy);
    const PetscScalar yTermR = (yTempR+yTempM)/(2*_hy*_hy);
    // Compute all of the z-shifted terms
    const PetscScalar zTempL = s1InvM*s2InvM/s3InvL;
    const PetscScalar zTempM = s1InvM*s2InvM/s3InvM;
    const PetscScalar zTempR = s1InvM*s2InvM/s3InvR;
    const PetscScalar zTermL = (zTempL+zTempM)/(2*_hz*_hz);
    const PetscScalar zTermR = (zTempR+zTempM)/(2*_hz*_hz);
    // Compute the center term
    const PetscScalar shiftedOmega =
        std::complex<PetscReal>(_control.omega,imagShift);
    const PetscScalar alpha =
        _localSlownessData[xLocal+_myXPortion*yLocal+_myXPortion*_myYPortion*z];
    const PetscScalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR)+
        (shiftedOmega*alpha)*(shiftedOmega*alpha)/(xTempM*yTempM*zTempM);

    // Fill in value and local index for the diagonal entry in this panel + PML
    PetscInt entry = 0;
    colIndices[entry] = rowIdx;
    row[entry] = centerTerm;
    ++entry;

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        if( xLocal != 0 )
        {
            colIndices[entry] = rowIdx - 1;
        }
        else
        {
            const PetscInt frontProcessOffset = myOffset - pillarSize;
            colIndices[entry] = frontProcessOffset + (_xChunkSize-1) + 
                yLocal*_xChunkSize + zLocal*_xChunkSize*_yChunkSize;
        }
        row[entry] = xTermL;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Back connection to (x+1,y,z)
    if( x != _control.nx-1 )
    {
        if( xLocal != _myXPortion-1 )
        {
            colIndices[entry] = rowIdx+1;
        }
        else
        {
            const PetscInt backProcessOffset = myOffset + pillarSize;
            const PetscInt backProcessXPortion = 
                std::min( _xChunkSize, _control.nx-(_myXOffset+_xChunkSize) );
            colIndices[entry] = backProcessOffset + yLocal*backProcessXPortion +
                zLocal*backProcessXPortion*_myYPortion;
        }
        row[entry] = xTermR;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        if( yLocal != 0 )
        {
            colIndices[entry] = rowIdx - _myXPortion;
        }
        else
        {
            const PetscInt leftProcessOffset = 
                myOffset - _control.nx*_yChunkSize*zSize;
            colIndices[entry] = leftProcessOffset + xLocal + 
                (_yChunkSize-1)*_xChunkSize + zLocal*_xChunkSize*_yChunkSize;
        }
        row[entry] = yTermL;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Right connection to (x,y+1,z)
    if( y != _control.ny )
    {
        if( yLocal != _myYPortion )
        {
            colIndices[entry] = rowIdx + _myXPortion;
        }
        else
        {
            const PetscInt rightProcessYPortion =
                std::min( _yChunkSize, _control.ny-(_myYOffset+_yChunkSize) );
            const PetscInt rightProcessOffset = 
                myOffset + _control.nx*rightProcessYPortion*zSize;
            colIndices[entry] = rightProcessOffset + xLocal + 
                zLocal*_xChunkSize*rightProcessYPortion;
        }
        row[entry] = yTermR;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Top connection to (x,y,z-1)
    if( zLocal != 0 )
    {
        colIndices[entry] = rowIdx - _myXPortion*_myYPortion;
        row[entry] = zTermL;
    }
    else
    {
        colIndices[entry] = -1;
    }
    ++entry;

    // Bottom connection to (x,y,z+1)
    if( zLocal != zSize-1 )
    {
        colIndices[entry] = rowIdx + _myXPortion*_myYPortion;
        row[entry] = zTermR;
    }
    else
    {
        colIndices[entry] = -1;
    }
}

PetscInt
psp::FiniteDiffSweepingPC::GetPanelConnectionSize() const
{
    // TODO: Support more than just the 7-point stencil
    return 1;
}

void 
psp::FiniteDiffSweepingPC::FormPanelConnections
( PetscInt x, PetscInt y, PetscInt z, 
  PetscInt zSize, PetscInt zSizeNext, PetscInt rowIndex,
  std::vector<PetscScalar>& nonzeros, std::vector<PetscInt>& colIndices ) const
{
    // Compute the right z term
    const PetscScalar s1InvM = s1Inv(x);
    const PetscScalar s2InvM = s2Inv(y);
    const PetscScalar s3InvM = s3Inv(z);
    const PetscScalar s3InvR = s3Inv(z+1);
    const PetscScalar zTempM = s1InvM*s2InvM/s3InvM;
    const PetscScalar zTempR = s1InvM*s2InvM/s3InvR;
    const PetscScalar zTermR = (zTempR+zTempM)/(2*_hz*_hz);

    colIndices[0] = rowIndex + _myXPortion*_myYPortion;
    nonzeros[0] = zTermR;
}