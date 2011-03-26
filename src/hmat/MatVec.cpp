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
#include <cstring>

// y := alpha A x + beta y
void psp::hmat::MatVec
( PetscScalar alpha, const FactorMatrix& A, const std::vector<PetscScalar>& x,
  PetscScalar beta,                               std::vector<PetscScalar>& y )
{
    const int r = A.r;
    std::vector<PetscScalar> t(r);

    // Form t := alpha (A.V)^H x
    blas::Gemv( 'C', A.n, A.r, alpha, &A.V[0], A.n, &x[0], 1, 0, &t[0], 1 );

    // Form y := (A.U) t + beta y
    blas::Gemv( 'N', A.m, A.r, 1, &A.U[0], A.m, &t[0], 1, beta, &y[0], 1 );
}

// y := alpha A x
void psp::hmat::MatVec
( PetscScalar alpha, const FactorMatrix& A, const std::vector<PetscScalar>& x,
                                                  std::vector<PetscScalar>& y )
{
    const int r = A.r;
    std::vector<PetscScalar> t(r);

    // Form t := alpha (A.V)^H x
    blas::Gemv( 'C', A.n, A.r, alpha, &A.V[0], A.n, &x[0], 1, 0, &t[0], 1 );

    // Form y := (A.U) t
    y.resize( x.size() );
    blas::Gemv( 'N', A.m, A.r, 1, &A.U[0], A.m, &t[0], 1, 0, &y[0], 1 );
}

