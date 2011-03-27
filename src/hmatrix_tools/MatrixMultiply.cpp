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

// C := alpha A B
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B, 
                      FactorMatrix<Scalar>& C )
{
    C.m = A.m;
    C.n = B.n;
    if( A.r < B.r )
    {
        // C := A.U (alpha (A.V^H B.U) B.V^H)
        C.r = A.r;

        // C.U := A.U
        C.U.resize( C.m*C.r );
        std::memcpy( &C.U[0], &A.U[0], C.m*C.r*sizeof(Scalar) );

        // C.V := alpha B.V (B.U^H A.V)
        //
        // We need a temporary buffer of size B.r*A.r; for now, we will allocate
        // it on the fly, but eventually we may want to have a permanent buffer
        // of a fixed size.
        C.V.resize( C.n*C.r );
        std::vector<Scalar> W( B.r*A.r );
        // W := B.U^H A.V
        blas::Gemm
        ( 'C', 'N', B.r, A.r, B.m, 
          1, &B.U[0], B.m, &A.V[0], A.n, 0, &W[0], B.r );
        // C.V := alpha B.V W = alpha B.V (B.U^H A.V)
        blas::Gemm
        ( 'N', 'N', C.n, C.r, B.r, 
          alpha, &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
    }
    else
    {
        // C := (alpha A.U (A.V^H B.U)) B.V^H
        C.r = B.r;

        // C.U := alpha A.U (A.V^H B.U)
        C.U.resize( C.m*C.r );
        std::vector<Scalar> W( A.r*B.r );
        // W := A.V^H B.U
        blas::Gemm
        ( 'C', 'N', A.r, B.r, A.n,
          1, &A.V[0], A.n, &B.U[0], B.m, 0, &W[0], A.r );
        // C.U := alpha A.U W = alpha A.U (A.V^H B.U)
        blas::Gemm
        ( 'N', 'N', C.m, C.r, A.r,
          alpha, &A.U[0], A.m, &W[0], A.r, 0, &C.U[0], C.m );

        // C.V := B.V
        C.V.resize( C.n*C.r );
        std::memcpy( &C.V[0], &B.V[0], C.n*C.r*sizeof(Scalar) );
    }
}

template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float>& A,
               const FactorMatrix<float>& B,
                     FactorMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double>& A,
                const FactorMatrix<double>& B,
                      FactorMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
                             const FactorMatrix< std::complex<float> >& B,
                                   FactorMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
                              const FactorMatrix< std::complex<double> >& B,
                                    FactorMatrix< std::complex<double> >& C );
