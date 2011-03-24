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

// C := alpha A B
void psp::hmat::FactorFactorMatMult
( const PetscScalar alpha, 
  const FactorMatrix& A, 
  const FactorMatrix& B, 
        FactorMatrix& C )
{
    C.m = A.m;
    C.n = B.n;
    if( A.r < B.r )
    {
        // C := A.U (alpha (A.V^T B.U) B.V^T)
        C.r = A.r;

        // C.U := A.U
        C.U.resize( C.m*C.r );
        std::memcpy( &C.U[0], &A.U[0], C.m*C.r*sizeof(PetscScalar) );

        // C.V := alpha B.V (B.U^T A.V)
        //
        // We need a temporary buffer of size B.r*A.r; for now, we will allocate
        // it on the fly, but eventually we may want to have a permanent buffer
        // of a fixed size.
        C.V.resize( C.n*C.r );
        std::vector<PetscScalar> W( B.r*A.r );
        // W := B.U^T A.V
        Gemm
        ( 'T', 'N', B.r, A.r, B.m, 
          1, &B.U[0], B.m, &A.V[0], A.n, 0, &W[0], B.r );
        // C.V := alpha B.V W = alpha B.V (B.U^T A.V)
        Gemm
        ( 'N', 'N', C.n, C.r, B.r, 
          alpha, &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
    }
    else
    {
        // C := (alpha A.U (A.V^T B.U)) B.V^T
        C.r = B.r;

        // C.U := alpha A.U (A.V^T B.U)
        C.U.resize( C.m*C.r );
        std::vector<PetscScalar> W( A.r*B.r );
        // W := A.V^T B.U
        Gemm
        ( 'T', 'N', A.r, B.r, A.n,
          1, &A.V[0], A.n, &B.U[0], B.m, 0, &W[0], A.r );
        // C.U := alpha A.U W = alpha A.U (A.V^T B.U)
        Gemm
        ( 'N', 'N', C.m, C.r, A.r,
          alpha, &A.U[0], A.m, &W[0], A.r, 0, &C.U[0], C.m );

        // C.V := B.V
        C.V.resize( C.n*C.r );
        std::memcpy( &C.V[0], &B.V[0], C.n*C.r*sizeof(PetscScalar) );
    }
}

