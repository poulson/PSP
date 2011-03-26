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

// C := alpha A + beta B
void psp::hmat::MatAdd
( PetscScalar alpha, const FactorMatrix& A, 
  PetscScalar beta,  const FactorMatrix& B, 
                           FactorMatrix& C )
{
    C.m = A.m;
    C.n = A.n;
    C.r = A.r + B.r;

    // C.U := [(alpha A.U), (beta B.U)]
    C.U.resize( C.m*C.r );
    // Copy in (alpha A.U)
    {
        const int r = A.r;
        const int m = A.m;
        PetscScalar* RESTRICT CU_A = &C.U[0];
        const PetscScalar* RESTRICT AU = &A.U[0]; 
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                CU_A[i+j*m] = alpha*AU[i+j*m];
    }
    // Copy in (beta B.U)
    {
        const int r = B.r;
        const int m = A.m;
        PetscScalar* RESTRICT CU_B = &C.U[C.m*A.r];
        const PetscScalar* RESTRICT BU = &B.U[0];
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                CU_B[i+j*m] = beta*BU[i+j*m];
    }

    // C.V := [A.V B.V]
    C.V.resize( C.n*C.r );
    std::memcpy( &C.V[0], &A.V[0], C.n*A.r*sizeof(PetscScalar) );
    std::memcpy( &C.V[C.n*A.r], &B.V[0], C.n*B.r*sizeof(PetscScalar) );
}

