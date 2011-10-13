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
#include <fstream>
#include <sstream>


#define PETSCMAT_DLL
#include "../src/mat/impls/aij/mpi/mpiaij.h" /*I  "petscmat.h"  I*/
#include "../src/mat/impls/sbaij/mpi/mpisbaij.h"
extern "C" {
#include "zmumps_c.h"
}
// This is defined so that we can reach into the PETSc MUMPS interface and 
// manually specify the reordering information
typedef struct {
#if defined(PETSC_USE_COMPLEX)
  ZMUMPS_STRUC_C id;
#else
  DMUMPS_STRUC_C id;
#endif 
  MatStructure   matstruc;
  PetscMPIInt    myid,size;
  PetscInt       *irn,*jcn,nz,sym,nSolve;
  PetscScalar    *val;
  MPI_Comm       comm_mumps;
  VecScatter     scat_rhs, scat_sol;
  PetscBool      isAIJ,CleanUpMUMPS;
  Vec            b_seq,x_seq;
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*ConvertToTriples)(Mat, int, MatReuse, int*, int**, int**, PetscScalar**);
} Mat_MUMPS;

//----------------------------------------------------------------------------//
// Private functions                                                          //
//----------------------------------------------------------------------------//

PetscScalar
psp::FiniteDiffSweepingPC::s1Inv( PetscInt x ) const
{
    if( (x+1)<_bx && _control.frontBC==PML )
    {
        const PetscReal delta = _bx-(x+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cx*delta*delta/(_bx*_bx*_bx*_hx*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( x>(_control.nx-_bx) && _control.backBC==PML )
    {
        const PetscReal delta = x-(_control.nx-_bx);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cx*delta*delta/(_bx*_bx*_bx*_hx*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

PetscScalar
psp::FiniteDiffSweepingPC::s2Inv( PetscInt y ) const
{
    if( (y+1)<_by && _control.leftBC==PML )
    {
        const PetscReal delta = _by-(y+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cy*delta*delta/(_by*_by*_by*_hy*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( y >(_control.ny-_by) && _control.rightBC==PML )
    {
        const PetscReal delta = y-(_control.ny-_by);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cy*delta*delta/(_by*_by*_by*_hy*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

PetscScalar
psp::FiniteDiffSweepingPC::s3Inv( PetscInt z ) const
{
    if( (z+1)<_bz )
    {
        const PetscReal delta = _bz-(z+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cz*delta*delta/(_bz*_bz*_bz*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( z>(_control.nz-_bz) && _control.bottomBC==PML )
    {
        const PetscReal delta = z-(_control.nz-_bz);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cz*delta*delta/(_bz*_bz*_bz*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

PetscScalar
psp::FiniteDiffSweepingPC::s3InvArtificial
( PetscInt z, PetscInt zOffset, PetscReal sizeOfPml ) const
{
    if( (z+1)<zOffset+sizeOfPml )
    {
        const PetscReal delta = zOffset+sizeOfPml-(z+1);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cz*delta*delta/
            (sizeOfPml*sizeOfPml*sizeOfPml*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else if( z>(_control.nz-_bz) && _control.bottomBC==PML )
    {
        const PetscReal delta = z-(_control.nz-_bz);
        const PetscReal realPart = 1;
        const PetscReal imagPart =
            _control.Cz*delta*delta/(_bz*_bz*_bz*_hz*_control.omega);
        return std::complex<PetscReal>(realPart,imagPart);
    }
    else
        return 1;
}

PetscInt
psp::FiniteDiffSweepingPC::GetSymmetricRowSize() const
{
    // TODO: Support more than just the 7-point stencil
    return 4;
}

PetscInt
psp::FiniteDiffSweepingPC::FormSymmetricRow
( PetscReal imagShift, 
  PetscInt x, PetscInt y, PetscInt z, 
  PetscInt zOffset, PetscInt zSize, PetscReal pmlSize,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*zSize + 
        _myProcessCol*_xChunkSize*_myYPortion*zSize;
    const PetscInt xLocal = x - _myXOffset;
    const PetscInt yLocal = y - _myYOffset;
    const PetscInt zLocal = z - zOffset;
    const PetscInt rowIdx = myOffset + 
        xLocal + _myXPortion*yLocal + _myXPortion*_myYPortion*zLocal;

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
    const PetscScalar alpha = 
        _localSlownessData[xLocal+_myXPortion*yLocal+_myXPortion*_myYPortion*z];
    // Avoid the formula shown in the slides in favor of what's in the serial
    // code, i.e., use (w^2*a^2 + i s) instead of (w + i s)^2*a^2, where 
    // s is the imaginary shift.
    /*
    const PetscScalar shiftedOmega = 
        std::complex<PetscReal>(_control.omega,imagShift);
    const PetscScalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR)+
        (shiftedOmega*alpha)*(shiftedOmega*alpha)*s1InvM*s2InvM*s3InvM;
    */
    const PetscScalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR)+
        (_control.omega*alpha)*(_control.omega*alpha)*s1InvM*s2InvM*s3InvM+
        std::complex<PetscReal>(0,imagShift);

    // Fill in value and local index for the diagonal entry in this panel + PML
    PetscInt entry = 0;
    colIndices[entry] = rowIdx;
    row[entry] = centerTerm;
    ++entry;

    // Back connection to (x+1,y,z)
    if( x != _control.nx-1 )
    {
        colIndices[entry] = MapNaturalToParallel( x+1, y, zLocal, zSize );
        row[entry] = xTermR;
        ++entry;
    }

    // Right connection to (x,y+1,z)
    if( y != _control.ny-1 )
    {
        colIndices[entry] = MapNaturalToParallel( x, y+1, zLocal, zSize );
        row[entry] = yTermR;
        ++entry;
    }

    // Bottom connection to (x,y,z+1)
    if( zLocal != zSize-1 )
    {
        // Short-circuit the MapNaturalToParallel routine
        colIndices[entry] = rowIdx + _myXPortion*_myYPortion;
        row[entry] = zTermR;
        ++entry;
    }

    return entry;
}

PetscInt
psp::FiniteDiffSweepingPC::GetRowSize() const
{
    // TODO: Support more than just the 7-point stencil
    return 7;
}

PetscInt
psp::FiniteDiffSweepingPC::FormRow
( PetscReal imagShift,
  PetscInt x, PetscInt y, PetscInt z, 
  PetscInt zOffset, PetscInt zSize, PetscReal pmlSize,
  std::vector<PetscScalar>& row, std::vector<PetscInt>& colIndices ) const
{
    const PetscInt myOffset = 
        _myProcessRow*_control.nx*_yChunkSize*zSize + 
        _myProcessCol*_xChunkSize*_myYPortion*zSize;
    const PetscInt xLocal = x - _myXOffset;
    const PetscInt yLocal = y - _myYOffset;
    const PetscInt zLocal = z - zOffset;
    const PetscInt rowIdx = myOffset + 
        xLocal + _myXPortion*yLocal + _myXPortion*_myYPortion*zLocal;

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
    const PetscScalar alpha = 
        _localSlownessData[xLocal+_myXPortion*yLocal+_myXPortion*_myYPortion*z];
    // Avoid the formula shown in the slides in favor of what's in the serial
    // code, i.e., use (w^2*a^2 + i s) instead of (w + i s)^2*a^2, where 
    // s is the imaginary shift.
    /*
    const PetscScalar shiftedOmega = 
        std::complex<PetscReal>(_control.omega,imagShift);
    const PetscScalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR)+
        (shiftedOmega*alpha)*(shiftedOmega*alpha)*s1InvM*s2InvM*s3InvM;
    */
    const PetscScalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR)+
        (_control.omega*alpha)*(_control.omega*alpha)*s1InvM*s2InvM*s3InvM+
        std::complex<PetscReal>(0,imagShift);

    // Fill in value and local index for the diagonal entry in this panel + PML
    PetscInt entry = 0;
    colIndices[entry] = rowIdx;
    row[entry] = centerTerm;
    ++entry;

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices[entry] = MapNaturalToParallel( x-1, y, zLocal, zSize );
        row[entry] = xTermL;
        ++entry;
    }

    // Back connection to (x+1,y,z)
    if( x != _control.nx-1 )
    {
        colIndices[entry] = MapNaturalToParallel( x+1, y, zLocal, zSize );
        row[entry] = xTermR;
        ++entry;
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices[entry] = MapNaturalToParallel( x, y-1, zLocal, zSize );
        row[entry] = yTermL;
        ++entry;
    }

    // Right connection to (x,y+1,z)
    if( y != _control.ny-1 )
    {
        colIndices[entry] = MapNaturalToParallel( x, y+1, zLocal, zSize );
        row[entry] = yTermR;
        ++entry;
    }

    // Top connection to (x,y,z-1)
    if( zLocal != 0 )
    {
        // Short-circuit the MapNaturalToParallel routine
        colIndices[entry] = rowIdx - _myXPortion*_myYPortion;
        row[entry] = zTermL;
        ++entry;
    }

    // Bottom connection to (x,y,z+1)
    if( zLocal != zSize-1 )
    {
        // Short-circuit the MapNaturalToParallel routine
        colIndices[entry] = rowIdx + _myXPortion*_myYPortion;
        row[entry] = zTermR;
        ++entry;
    }

    return entry;
}

PetscInt
psp::FiniteDiffSweepingPC::GetPanelConnectionSize() const
{
    // TODO: Support more than just the 7-point stencil
    return 1;
}

PetscInt 
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

    // This should eventually be moved outside of this loop since it is 
    // almost as expensive as the floating point computation and only needs
    // to be performed once per off-diagonal block.
    const PetscInt myColOffset = 
        _myProcessRow*_control.nx*_yChunkSize*zSizeNext + 
        _myProcessCol*_xChunkSize*_myYPortion*zSizeNext;

    const PetscInt xLocal = x - _myXOffset;
    const PetscInt yLocal = y - _myYOffset;

    // The 7-point stencil will only touch the first xy plane of the next 
    // panel (so that zLocal=0)
    PetscInt entry = 0;
    // Short-circuit the MapNaturalToParallel routine
    colIndices[entry] = myColOffset + xLocal + yLocal*_myXPortion;
    nonzeros[entry] = zTermR;
    ++entry;

    return entry;
}

void
psp::FiniteDiffSweepingPC::RecursiveReordering
( PetscInt xOffset, PetscInt xSize, 
  PetscInt yOffset, PetscInt ySize,
  PetscInt zOffset, PetscInt zSize, PetscInt zSizePanel,
  PetscInt minSize, PetscInt* reordering ) const
{
    if( xSize <= minSize && ySize <= minSize && zSize <= minSize )
    {
        // Write the leaf
        for( PetscInt x=xOffset; x<xOffset+xSize; ++x )
            for( PetscInt y=yOffset; y<yOffset+ySize; ++y )
                for( PetscInt z=zOffset; z<zOffset+zSize; ++z )
                    reordering[(x-xOffset)*ySize*zSize+
                               (y-yOffset)*zSize+
                               (z-zOffset)] = 
                        MapNaturalToParallel(x,y,z,zSizePanel)+1;
    }
    else if( xSize >= ySize && xSize >= zSize )
    {
        // Cut the x dimension and write the separator
        const PetscInt xLeftSize = (xSize-1) / 2;
        const PetscInt separatorSize = ySize*zSize;
        PetscInt* separatorSection = 
            &reordering[xSize*ySize*zSize-separatorSize];
        for( PetscInt y=yOffset; y<yOffset+ySize; ++y )
            for( PetscInt z=zOffset; z<zOffset+zSize; ++z )
                separatorSection[(y-yOffset)*zSize+(z-zOffset)] = 
                    MapNaturalToParallel(xOffset+xLeftSize,y,z,zSizePanel)+1;
        // Recurse on the left side of the x cut
        RecursiveReordering
        ( xOffset, xLeftSize, yOffset, ySize, zOffset, zSize, zSizePanel,
          minSize, reordering );
        // Recurse on the right side of the x cut
        RecursiveReordering
        ( xOffset+(xLeftSize+1), xSize-(xLeftSize+1), yOffset, ySize, 
          zOffset, zSize, zSizePanel, minSize, 
          &reordering[xLeftSize*ySize*zSize] );
    }
    else if( ySize >= zSize )
    {
        // Cut the y dimension and write the separator
        const PetscInt yLeftSize = (ySize-1) / 2;
        const PetscInt separatorSize = xSize*zSize;
        PetscInt* separatorSection = 
            &reordering[xSize*ySize*zSize-separatorSize];
        for( PetscInt x=xOffset; x<xOffset+xSize; ++x )
            for( PetscInt z=zOffset; z<zOffset+zSize; ++z )
                separatorSection[(x-xOffset)*zSize+(z-zOffset)] = 
                    MapNaturalToParallel(x,yOffset+yLeftSize,z,zSizePanel)+1;
        // Recurse on the left side of the y cut
        RecursiveReordering
        ( xOffset, xSize, yOffset, yLeftSize, zOffset, zSize, zSizePanel,
          minSize, reordering );
        // Recurse on the right side of the y cut
        RecursiveReordering
        ( xOffset, xSize, yOffset+(yLeftSize+1), ySize-(yLeftSize+1), 
          zOffset, zSize, zSizePanel, minSize, 
          &reordering[xSize*yLeftSize*zSize] );
    }
    else // the z dimension is the largest and is above the threshold
    {
        // Cut the z dimension and write the separator
        const PetscInt zLeftSize = (zSize-1) / 2;
        const PetscInt separatorSize = xSize*ySize;
        PetscInt* separatorSection = 
            &reordering[xSize*ySize*zSize-separatorSize];
        for( PetscInt x=xOffset; x<xOffset+xSize; ++x )
            for( PetscInt y=yOffset; y<yOffset+ySize; ++y )
                separatorSection[(x-xOffset)*ySize+(y-yOffset)] = 
                    MapNaturalToParallel(x,y,zOffset+zLeftSize,zSizePanel)+1;
        // Recurse on the left side of the y cut
        RecursiveReordering
        ( xOffset, xSize, yOffset, ySize, zOffset, zLeftSize, zSizePanel,
          minSize, reordering );
        // Recurse on the right side of the y cut
        RecursiveReordering
        ( xOffset, xSize, yOffset, ySize, 
          zOffset+(zLeftSize+1), zSize-(zLeftSize+1), zSizePanel, minSize,
          &reordering[xSize*ySize*zLeftSize] );
    }
}

//----------------------------------------------------------------------------//
// Public functions                                                           //
//----------------------------------------------------------------------------//

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
    _xLastSize = _xChunkSize + (_control.nx % _numProcessCols);
    _yLastSize = _yChunkSize + (_control.ny % _numProcessRows);
    _myXOffset = _myProcessCol*_xChunkSize;
    _myYOffset = _myProcessRow*_yChunkSize;
    _myXPortion = 
      ( _myProcessCol==_numProcessCols-1 ? _xLastSize : _xChunkSize );
    _myYPortion = 
      ( _myProcessRow==_numProcessRows-1 ? _yLastSize : _yChunkSize );
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

PetscInt
psp::FiniteDiffSweepingPC::GetMyXOffset() const
{
    return _myXOffset;
}

PetscInt
psp::FiniteDiffSweepingPC::GetMyYOffset() const
{
    return _myYOffset;
}

PetscInt
psp::FiniteDiffSweepingPC::GetMyXPortion() const
{
    return _myXPortion;
}

PetscInt
psp::FiniteDiffSweepingPC::GetMyYPortion() const
{
    return _myYPortion;
}

PetscReal
psp::FiniteDiffSweepingPC::GetXSpacing() const
{ 
    return _hx;
}

PetscReal
psp::FiniteDiffSweepingPC::GetYSpacing() const
{
    return _hy;
}

PetscReal
psp::FiniteDiffSweepingPC::GetZSpacing() const
{
    return _hz;
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
    const double forwardStartTime = MPI_Wtime();
    if( _rank == 0 )
    {
        std::cout << "\n  Forming forward operator...";
        std::cout.flush();
    }
    {
        const PetscInt localSize = GetLocalSize();
        const PetscInt size = _control.nx*_control.ny*_control.nz;

        MatCreate( _comm, &A );
        MatSetSizes( A, localSize, localSize, size, size );
        MatSetType( A, MATSBAIJ );
        MatSetBlockSize( A, 1 );
        // TODO: Generalize this step for more stencils.
        MatMPISBAIJSetPreallocation( A, 1, 4, PETSC_NULL, 3, PETSC_NULL );

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
            const PetscInt validEntries = FormSymmetricRow
            ( 0.0, x, y, z, 0, _control.nz, _bz, nonzeros, colIndices ); 

            // Put this row into the distributed matrix
            MatSetValues
            ( A, 1, &i, validEntries, &colIndices[0], &nonzeros[0],
              INSERT_VALUES );
        }
        MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
#ifdef VIEW_MATRICES
        // Write the forward operator to file in ASCII
        PetscViewer viewer;
        PetscViewerCreate( _comm, &viewer );
        PetscViewerSetType( viewer, PETSCVIEWERASCII );
        PetscViewerSetFormat( viewer, PETSC_VIEWER_DEFAULT );
        PetscViewerFileSetName( viewer, "fullOperator.dat" );
        MatView( A, viewer );

#ifdef BUILT_PETSC_WITH_X11
        // Draw a picture of the sparsity structure
        MatView( A, PETSC_VIEWER_DRAW_WORLD );
#endif // BUILT_PETSC_WITH_X11
#endif // VIEW_MATRICES
    }
    if( _rank == 0 )
    {
        const double forwardStopTime = MPI_Wtime();
        std::cout << forwardStopTime-forwardStartTime << " secs." << std::endl;
    }

    // Determine the z sizes and offsets of the padded and unpadded panels
    //
    // TODO: Consider allowing the artificial panel PML to be of a different 
    //       size than the outer fixed PML.
    const PetscInt nzMinusTopPml = _control.nz - (_bzPadded-1);
    const PetscInt numPanels = 
      ( nzMinusTopPml>0 ? (nzMinusTopPml-1)/_control.planesPerPanel+1 : 1 );
    _zSizesOfPanels.resize( numPanels );
    _zOffsetsOfPanels.resize( numPanels );
    _zPmlSizesOfPanels.resize( numPanels );
    _zSizesOfPaddedPanels.resize( numPanels );
    _zOffsetsOfPaddedPanels.resize( numPanels );
    _zPmlSizesOfPaddedPanels.resize( numPanels );
    _paddedFactors.resize( numPanels );
    for( PetscInt m=0; m<numPanels; ++m )
    {
        if( m == 0 )
        {
            const PetscInt zOffset = 0;
            const PetscInt zSize = 
                std::min(_control.nz,(_bzPadded-1)+_control.planesPerPanel);
            const PetscReal pmlSize = _bz;

            _zSizesOfPanels[m] = zSize;
            _zOffsetsOfPanels[m] = zOffset;
            _zPmlSizesOfPanels[m] = pmlSize;
            _zSizesOfPaddedPanels[m] = zSize;
            _zOffsetsOfPaddedPanels[m] = zOffset;
            _zPmlSizesOfPaddedPanels[m] = pmlSize;
        }
        else if( m == numPanels-1 )
        {
            const PetscInt zOffset = m*_control.planesPerPanel;
            const PetscInt zSize = _control.nz - zOffset;
            const PetscInt pmlSize = _bzPadded;

            _zSizesOfPanels[m] = zSize - (pmlSize-1);
            _zOffsetsOfPanels[m] = zOffset + (pmlSize-1);
            _zPmlSizesOfPanels[m] = 0;
            _zSizesOfPaddedPanels[m] = zSize;
            _zOffsetsOfPaddedPanels[m] = zOffset;
            _zPmlSizesOfPaddedPanels[m] = pmlSize;
        }
        else
        {
            const PetscInt zOffset = m*_control.planesPerPanel;
            const PetscInt zSize = (_bzPadded-1) + _control.planesPerPanel;
            const PetscInt pmlSize = _bzPadded;

            _zSizesOfPanels[m] = _control.planesPerPanel;
            _zOffsetsOfPanels[m] = zOffset + (pmlSize-1);
            _zPmlSizesOfPanels[m] = 0;
            _zSizesOfPaddedPanels[m] = zSize;
            _zOffsetsOfPaddedPanels[m] = zOffset;
            _zPmlSizesOfPaddedPanels[m] = pmlSize;
        }
    }

    //----------------------------------------------------------------//
    // Form each of the PML-padded diagonal blocks and then factor it //
    //----------------------------------------------------------------//
    if( _solver == MUMPS_SYMMETRIC )
    {
        // Our solver supports symmetry, so only fill the lower triangles
        for( PetscInt m=0; m<numPanels; ++m )
        {
            Mat D;
            MatCreate( _comm, &D );

            const PetscInt zSize = _zSizesOfPaddedPanels[m];
            const PetscInt zOffset = _zOffsetsOfPaddedPanels[m];
            const PetscReal pmlSize = _zPmlSizesOfPaddedPanels[m];

            const PetscInt localSize = _myXPortion*_myYPortion*zSize;
            const PetscInt size = _control.nx*_control.ny*zSize;
            MatSetSizes( D, localSize, localSize, size, size );

            // SBAIJ is required for symmetry support
            MatSetType( D, MATSBAIJ ); 
            MatSetBlockSize( D, 1 );

            // Preallocate memory. This should be an upper bound for 7-point.
            // TODO: Generalize this for more stencils.
            MatMPISBAIJSetPreallocation( D, 1, 4, PETSC_NULL, 3, PETSC_NULL );

            // Fill our portion of the distributed matrix.
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
                const PetscInt validEntries = FormSymmetricRow
                ( _control.imagShift, x, y, z, 
                  zOffset, zSize, pmlSize, nonzeros, colIndices );

                // Store the values into the distributed matrix
                MatSetValues
                ( D, 1, &i, validEntries, &colIndices[0], &nonzeros[0],
                  INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );
#ifdef VIEW_MATRICES
            // Write the approximate diagonal block to file
            PetscViewer viewer;
            PetscViewerCreate( _comm, &viewer );
            PetscViewerSetType( viewer, PETSCVIEWERASCII );
            PetscViewerSetFormat( viewer, PETSC_VIEWER_DEFAULT );
            std::ostringstream name;
            name << "diagBlock-" << m << ".dat"; 
            PetscViewerFileSetName( viewer, name.str().c_str() );
            MatView( D, viewer );

#ifdef BUILT_PETSC_WITH_X11
            // Draw a picture of the sparsity structure
            MatView( D, PETSC_VIEWER_DRAW_WORLD );
#endif // BUILT_PETSC_WITH_X11
#endif // VIEW_MATRICES

            // Factor the matrix
            Mat& F = _paddedFactors[m];
            MatGetFactor( D, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &F );
            MatFactorInfo cholInfo;
            cholInfo.fill = 10.0; // TODO: Tweak/expose this
            cholInfo.dtcol = 1.e-6; // irrelevant for us?
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            // The manual ordering does not seem to work very well...
            //Mat_MUMPS* chol = (Mat_MUMPS*)F->spptr;
            //chol->id.icntl[6] = 1;
            //chol->id.icntl[27] = 1;
            //std::vector<PetscInt> reordering;
            //if( _rank == 0 )
            //{
            //    // For now, hardcode the minimum leaf dimension
            //    const PetscInt minSize = 1;
            //
            //    reordering.resize( _control.nx*_control.ny*zSize );
            //    RecursiveReordering
            //    ( 0, _control.nx, 0, _control.ny, 0, zSize, zSize, minSize, 
            //      &reordering[0] );
            //    chol->id.perm_in = &reordering[0];
            // }
            const double factorStartTime = MPI_Wtime();
            if( _rank == 0 )
            {
                std::cout << "  Factoring diagonal block " << m << "...";
                std::cout.flush();
            }
            MatCholeskyFactorSymbolic( F, D, perm, &cholInfo );
            //if( _rank == 0 )
            //    reordering.clear();
            MatCholeskyFactorNumeric( F, D, &cholInfo );
            MatDestroy( D );
            if( _rank == 0 )
            {
                const double factorStopTime = MPI_Wtime();
                std::cout << factorStopTime-factorStartTime << " seconds." 
                          << std::endl;
            }
        }
    }
    else // _solver in { ILU, ILUDT, MUMPS, SUPERLU_DIST }
    {
        // Our solver does not support symmetry, fill the entire matrices
        for( PetscInt m=0; m<numPanels; ++m )
        {
            Mat D;
            MatCreate( _comm, &D );

            const PetscInt zSize = _zSizesOfPaddedPanels[m];
            const PetscInt zOffset = _zOffsetsOfPaddedPanels[m];
            const PetscReal pmlSize = _zPmlSizesOfPaddedPanels[m];

            const PetscInt localSize = _myXPortion*_myYPortion*zSize;
            const PetscInt size = _control.nx*_control.ny*zSize;
            MatSetSizes( D, localSize, localSize, size, size );

            MatSetType( D, MATAIJ );

            // Preallocate memory
            // TODO: Generalize this for more general stencils.
            MatMPIAIJSetPreallocation( D, 7, PETSC_NULL, 3, PETSC_NULL );

            // Fill our portion of the diagonal block
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
                const PetscInt validEntries = FormRow
                ( _control.imagShift, x, y, z, 
                  zOffset, zSize, pmlSize, nonzeros, colIndices );

                // Store the row in the distributed matrix
                MatSetValues
                ( D, 1, &i, validEntries, &colIndices[0], &nonzeros[0],
                  INSERT_VALUES );
            }
            MatAssemblyBegin( D, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( D, MAT_FINAL_ASSEMBLY );
#ifdef VIEW_MATRICES
            // Write the approximate diagonal block to file
            PetscViewer viewer;
            PetscViewerCreate( _comm, &viewer );
            PetscViewerSetType( viewer, PETSCVIEWERASCII );
            PetscViewerSetFormat( viewer, PETSC_VIEWER_DEFAULT );
            std::ostringstream name;
            name << "diagBlock-" << m << ".dat"; 
            PetscViewerFileSetName( viewer, name.str().c_str() );
            MatView( D, viewer );

#ifdef BUILT_PETSC_WITH_X11
            // Draw a picture of the sparsity structure
            MatView( D, PETSC_VIEWER_DRAW_WORLD );
#endif // BUILT_PETSC_WITH_X11
#endif // VIEW_MATRICES

            // Factor the matrix
            Mat& F = _paddedFactors[m];
            MatFactorInfo luInfo;
            if( _solver == ILU )
            {
                std::cout << "Using PETSC ILU...";
                std::cout.flush();
                MatGetFactor( D, MATSOLVERPETSC, MAT_FACTOR_ILU, &F );
                luInfo.fill = 5.0; // TODO: Tweak/expose this
                luInfo.dtcol = 0.1; // irrelevant for us?
            }
            else if( _solver == ILUDT )
            {
                MatGetFactor( D, MATSOLVERPETSC, MAT_FACTOR_ILUDT, &F );
                luInfo.fill = 5.0; // TODO: Tweak/expose this
                luInfo.dtcol = 0.1; // irrelevant for us?
            }
            else if( _solver == MUMPS )
            {
                MatGetFactor( D, MATSOLVERMUMPS, MAT_FACTOR_LU, &F );
                luInfo.fill = 10.0; // TODO: Tweak/expose this
                luInfo.dtcol = 0.1; // irrelevant for us?
            }
            else
            {
                MatGetFactor( D, MATSOLVERSUPERLU_DIST, MAT_FACTOR_LU, &F );
                luInfo.fill = 10.0; // TODO: Tweak/expose this
                luInfo.dtcol = 0.1; // irrelevant for us?
            }
            IS perm;
            ISCreateStride( _comm, size, 0, 1, &perm );
            // The manual ordering does not seem to work very well...
            //Mat_MUMPS* lu = (Mat_MUMPS*)F->spptr;
            //lu->id.icntl[6] = 1;
            //lu->id.icntl[27] = 1;
            //std::vector<PetscInt> reordering;
            //if( _rank == 0 )
            //{
            //    // For now, hardcode the minimum leaf dimension
            //    const PetscInt minSize = 1;
            //
            //    reordering.resize( _control.nx*_control.ny*zSize );
            //    RecursiveReordering
            //    ( 0, _control.nx, 0, _control.ny, 0, zSize, zSize, minSize, 
            //      &reordering[0] );
            //    lu->id.perm_in = &reordering[0];
            //}
            const double factorStartTime = MPI_Wtime();
            if( _rank == 0 )
            {
                std::cout << "  Factoring diagonal block " << m << "...";
                std::cout.flush();
            }
            MatLUFactorSymbolic( F, D, perm, perm, &luInfo );
            //if( _rank == 0 )
            //    reordering.clear();
            MatLUFactorNumeric( F, D, &luInfo );
            MatDestroy( D );
            if( _rank == 0 )
            {
                const double factorStopTime = MPI_Wtime();
                std::cout << factorStopTime-factorStartTime << " seconds." 
                          << std::endl;
            }
        }
    }

    //---------------------------------------//
    // Form the unpadded off-diagonal blocks //
    //---------------------------------------//
    if( _rank == 0 )
    {
        std::cout << "  Forming off-diagonal blocks...";
        std::cout.flush();
    }
    _offDiagBlocks.resize( numPanels-1 );
    for( PetscInt m=0; m<numPanels-1; ++m )
    {
        Mat& B = _offDiagBlocks[m];
        MatCreate( _comm, &B );

        const PetscInt zSize = _zSizesOfPanels[m];
        const PetscInt zOffset = _zOffsetsOfPanels[m];
        const PetscInt zSizeNext = _zSizesOfPanels[m+1];

        const PetscInt matrixHeight = _control.nx*_control.ny*zSize;
        const PetscInt matrixWidth = _control.nx*_control.ny*zSizeNext;
        const PetscInt localMatrixHeight = _myXPortion*_myYPortion*zSize;
        const PetscInt localMatrixWidth = _myXPortion*_myYPortion*zSizeNext;
        MatSetSizes
        ( B, localMatrixHeight, localMatrixWidth, matrixHeight, matrixWidth );

        MatSetType( B, MATAIJ );

        // Preallocate memory (go ahead and use one diagonal + one off-diagonal)
        // even though this is a gross overestimate (roughly factor of 10).
        //
        // TODO: Generalize beyond 7-point stencil
        MatMPIAIJSetPreallocation( B, 1, PETSC_NULL, 1, PETSC_NULL );

        // Fill the connections between our nodes in the last xy plane of 
        // unpadded panel 'm' and the first xy plane of unpadded panel 
        // 'm+1'.
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
            const PetscInt validEntries = FormPanelConnections
            ( x, y, z, zSize, zSizeNext, i, nonzeros, colIndices );

            // Store the row in the distributed matrix
            MatSetValues
            ( B, 1, &i, validEntries, &colIndices[0], &nonzeros[0], 
              INSERT_VALUES );
        }
        MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY );
#ifdef VIEW_MATRICES
        // Write the approximate diagonal block to file
        PetscViewer viewer;
        PetscViewerCreate( _comm, &viewer );
        PetscViewerSetType( viewer, PETSCVIEWERASCII );
        PetscViewerSetFormat( viewer, PETSC_VIEWER_DEFAULT );
        std::ostringstream name;
        name << "offDiagBlock-" << m << ".dat"; 
        PetscViewerFileSetName( viewer, name.str().c_str() );
        MatView( B, viewer );

#ifdef BUILT_PETSC_WITH_X11
        // Draw a picture of the sparsity structure
        MatView( B, PETSC_VIEWER_DRAW_WORLD );
#endif // BUILT_PETSC_WITH_X11
#endif // VIEW_MATRICES
    }
    if( _rank == 0 )
        std::cout << "done." << std::endl;
    
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

    _zSizesOfPanels.clear();
    _zOffsetsOfPanels.clear();
    _zPmlSizesOfPanels.clear();
    _zSizesOfPaddedPanels.clear();
    _zOffsetsOfPaddedPanels.clear();
    _zPmlSizesOfPaddedPanels.clear();

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
        // its right vector solution should match u_{m+1}. Unfortunately, 
        // we cannot assume that there exists an off-diagonal block for this 
        // diagonal block, so directly create the vector.
        //
        // MatGetVecs( _offDiagBlocks[m], PETSC_NULL, &uPanels[m] );
        const PetscInt localSize = _myXPortion*_myYPortion*_zSizesOfPanels[m];
        VecCreate( _comm, &uPanels[m] );
        VecSetSizes( uPanels[m], localSize, PETSC_DECIDE );
        VecSetBlockSize( uPanels[m], 1 );
        VecSetType( uPanels[m], VECMPI );

        // Get a pointer to this panel solution so that we may memcpy the
        // appropriate chunk of f into it
        PetscScalar* uPanelLocalData;
        VecGetArray( uPanels[m], &uPanelLocalData );

        // Copy this panel's RHS, "f_m", into u[m]
        const PetscInt zOffset = _zOffsetsOfPanels[m];
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
            ( uPanelLocalData, paddedPanelSolLocalData,
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }
        else
        {
#ifndef RELEASE
            if( _zPmlSizesOfPanels[m] != 0 )
                throw std::logic_error("Nonsensical PML size.");
#endif
            const PetscInt zDimsAdded = _zPmlSizesOfPaddedPanels[m]-1;
            const PetscInt padding = _myXPortion*_myYPortion*zDimsAdded;
#ifndef RELEASE
            const PetscInt localRHSSize = VecGetLocalSize( paddedPanelRHS );
            if( padding + uPanelLocalSize != localRHSSize )
            {
                std::ostringstream s;
                s << _rank << ": m=" << m << ", padding=" << padding 
                  << ", uPanelLocalSize=" << uPanelLocalSize 
                  << ", localRHSSize=" << localRHSSize;
                throw std::logic_error( s.str().c_str() );
            }
#endif

            // Copy our panel's data into the back of the zero-padded buffer
            VecSet( paddedPanelRHS, 0.0 );
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( &paddedPanelRHSLocalData[padding], uPanelLocalData,
              uPanelLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the relevant portion of the solution and store in u_m
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( uPanelLocalData, &paddedPanelSolLocalData[padding],
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
        MatGetVecs( _offDiagBlocks[m], PETSC_NULL, &work );
        MatMult( _offDiagBlocks[m], uPanels[m+1], work );

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
            ( workLocalData, paddedPanelSolLocalData,
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelSol, &paddedPanelSolLocalData );
        }
        else
        {
            const PetscInt zDimsAdded = _zPmlSizesOfPaddedPanels[m]-1;
            const PetscInt padding = _myXPortion*_myYPortion*zDimsAdded;

            // Copy our panel's data into the back of the zero-padded buffer
            VecSet( paddedPanelRHS, 0.0 );
            PetscScalar* paddedPanelRHSLocalData;
            VecGetArray( paddedPanelRHS, &paddedPanelRHSLocalData );
            std::memcpy
            ( &paddedPanelRHSLocalData[padding], workLocalData,
              workLocalSize*sizeof(PetscScalar) );
            VecRestoreArray( paddedPanelRHS, &paddedPanelRHSLocalData );

            // Solve using our sparse-direct factorization
            MatSolve( _paddedFactors[m], paddedPanelRHS, paddedPanelSol );

            // Extract the relevant portion of the solution and store in u_m
            PetscScalar* paddedPanelSolLocalData;
            VecGetArray( paddedPanelSol, &paddedPanelSolLocalData );
            std::memcpy
            ( workLocalData, &paddedPanelSolLocalData[padding],
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
        const PetscInt zOffset = _zOffsetsOfPanels[m];
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

void
psp::FiniteDiffSweepingPC::WriteParallelVtkFile
( Vec& v, const char* basename ) const
{
    // Have the root process right out the parallel XML description of the data
    if( _rank == 0 )
    {
        std::ofstream realFile, imagFile;
        std::ostringstream os;
        os << basename << "_real.pvti";
        realFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_imag.pvti";
        imagFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
           << " <PImageData WholeExtent=\""
           << "0 " << _control.nx << " 0 " << _control.ny 
           << " 0 " << _control.nz 
           << "\" Origin=\"0 0 0\" Spacing=\"" 
           << _hx << " " << _hy << " " << _hz
           << "\" GhostLevel=\"0\">\n"
           << "  <PCellData Scalars=\"cell_scalars\">\n"
           << "   <PDataArray type=\"Float32\" Name=\"cell_scalars\"/>\n"
           << "  </PCellData>\n";
        for( PetscInt i=0; i<_numProcessRows; ++i )
        {
            const PetscInt yPortion = GetProcessRowYPortion( i );
            const PetscInt yOffset = i*_yChunkSize;
            for( PetscInt j=0; j<_numProcessCols; ++j )
            {
                const PetscInt xPortion = GetProcessColXPortion( j );
                const PetscInt xOffset = j*_xChunkSize;

                os << "  <Piece Extent=\"" 
                   << xOffset << " " << xOffset+xPortion << " "
                   << yOffset << " " << yOffset+yPortion << " "
                   << "0 " << _control.nz << "\" ";
                realFile << os.str();
                imagFile << os.str();
                os.clear(); os.str("");
                realFile << "Source=\"" << basename << "_real_"
                         << i << "_" << j << ".vti\"/>\n";
                imagFile << "Source=\"" << basename << "_imag_"
                         << i << "_" << j << ".vti\"/>\n";
            }
        }
        os << " </PImageData>\n"
           << "</VTKFile>" << std::endl;
        realFile << os.str();
        imagFile << os.str();
        realFile.close();
        imagFile.close();
    }

    // Have each process write out their local data
    std::ofstream realFile, imagFile;
    std::ostringstream os;
    os << basename << "_real_" << _myProcessRow << "_" << _myProcessCol 
       << ".vti";
    realFile.open( os.str().c_str() );
    os.clear(); os.str("");
    os << basename << "_imag_" << _myProcessRow << "_" << _myProcessCol
       << ".vti";
    imagFile.open( os.str().c_str() );
    os.clear(); os.str("");
    os << "<?xml version=\"1.0\"?>\n"
       << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
       << " <ImageData WholeExtent=\""
       << "0 " << _control.nx << " 0 " << _control.ny << " 0 " << _control.nz
       << "\" Origin=\"0 0 0\" Spacing=\"" << _hx << " " << _hy << " " << _hz 
       << "\">\n"
       << "  <Piece Extent=\"" 
       << _myXOffset << " " << _myXOffset + _myXPortion << " "
       << _myYOffset << " " << _myYOffset + _myYPortion << " "
       << "0 " << _control.nz << "\">\n"
       << "   <CellData Scalars=\"cell_scalars\">\n"
       << "    <DataArray type=\"Float32\" Name=\"cell_scalars\""
       << " format=\"ascii\">\n";
    realFile << os.str();
    imagFile << os.str();
    os.clear(); os.str("");
    const PetscInt myPanelSize = _myXPortion*_myYPortion;
    PetscScalar* vLocalData;
    VecGetArray( v, &vLocalData );
    for( PetscInt z=0; z<_control.nz; ++z )
    {
        for( PetscInt k=0; k<_myXPortion*_myYPortion; ++k )
        {
            std::complex<PetscReal> u = vLocalData[k+z*myPanelSize];
            realFile << (float)std::real(u) << " "; 
            imagFile << (float)std::imag(u) << " ";
        }
        realFile << "\n";
        imagFile << "\n";
    }
    VecRestoreArray( v, &vLocalData );
    os << "    </DataArray>\n"
       << "   </CellData>\n"
       << "  </Piece>\n"
       << " </ImageData>\n"
       << "</VTKFile>" << std::endl;
    realFile << os.str();
    imagFile << os.str();
    realFile.close();
    imagFile.close();
}
