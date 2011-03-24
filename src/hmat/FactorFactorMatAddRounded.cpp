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

// C :~= alpha A + beta B
//
// We will make use of a pivoted QLP factorization [Stewart, 1999] in order to
// get an accurate approximation to the truncated singular value decomposition 
// in O(mnk) work for an m x n matrix and k singular vectors.
//
// See Huckaby and Chan's 2004 paper:
// "Stewart's pivoted QLP decomposition for low-rank matrices".
//
// TODO: 
// Carefully consider whether or not we can avoid explicitly expanding our
// low-rank form before performing the QLP decomposition. My suspicion is no, so
// perhaps we should instead aim for avoiding the buffer allocation cost by 
// leaving one around.
void psp::hmat::FactorFactorRoundedMatAdd
( const PetscInt forcedRank,
  const PetscScalar alpha, 
  const FactorMatrix& A, 
  const PetscScalar beta,
  const FactorMatrix& B, 
        FactorMatrix& C )
{
    C.m = A.m;
    C.n = A.n;
    C.r = forcedRank;

    // Expand the factors into a buffer:
    //
    // explicitC := alpha (A.U A.V^T) + beta (B.U B.V^T)
    std::vector<PetscScalar> explicitC( C.m*C.n );
    Gemm
    ( 'N', 'T', C.m, C.n, A.r,
      alpha, &A.U[0], C.m, &A.V[0], C.n, 0, &explicitC[0], C.m );
    Gemm
    ( 'N', 'T', C.m, C.n, B.r,
      beta, &B.U[0], C.m, &B.V[0], C.n, 1, &explicitC[0], C.m );

    // TODO: 
    // Perform the rank-(forcedRank) QLP decomposition on explicitC
    //   (explicitC) = Q L P^T Pi,
    // where Pi is the permutation matrix associated with the pivoted QR step.
    // We would then set:
    //   C.U := Q L,    [Trmv]
    //   C.V := Pi^T P  [memcpy's]
}

