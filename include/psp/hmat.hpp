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
#ifndef PSP_HMAT_HPP
#define PSP_HMAT_HPP 1

#include "psp/blas.hpp"
#include "psp/lapack.hpp"
#include "psp/hmat/factor_matrix.hpp"
#include "psp/hmat/sparse_matrix.hpp"

namespace psp {
namespace hmat {

// Ensure that the matrix A has a rank of at most 'maxRank'
void Compress( int maxRank, FactorMatrix& A );

// Generalized add of two factor matrices, C := alpha A + beta B
void MatAdd
( PetscScalar alpha, const FactorMatrix& A,
  PetscScalar beta,  const FactorMatrix& B,
                           FactorMatrix& C );

// Generalized add of two factor matrices, C := alpha A + beta B, 
// where C is then forced to be of rank at most 'maxRank'
void MatAddRounded
( int maxRank,
  PetscScalar alpha, const FactorMatrix& A,
  PetscScalar beta,  const FactorMatrix& B,
                           FactorMatrix& C );

// C := alpha A B
void MatMult
( PetscScalar alpha, const FactorMatrix& A, const FactorMatrix& B,
                           FactorMatrix& C );

// y := alpha A x + beta y
void MatVec
( PetscScalar alpha, const FactorMatrix& A, const std::vector<PetscScalar>& x,
  PetscScalar beta,                               std::vector<PetscScalar>& y );

// y := alpha A x
void MatVec
( PetscScalar alpha, const FactorMatrix& A, const std::vector<PetscScalar>& x,
                                                  std::vector<PetscScalar>& y );
 
} // namespace hmat
} // namespace psp

#endif // PSP_HMAT_HPP
