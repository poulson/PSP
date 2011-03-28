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

// TODO: Decide whether or not to add in tranposition options. It would seem
//       that they are not needed for our current problem.

// Dense C := alpha A B
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.symmetric && B.symmetric )
        throw std::logic_error("Product of symmetric matrices not supported.");
#endif
    C.m = A.m;
    C.n = B.n;
    C.ldim = C.m;
    C.buffer.resize( C.ldim*C.n );
    C.symmetric = false;

    if( A.symmetric )
    {
        blas::Symm
        ( 'L', 'L', C.m, C.n,
          alpha, &A.buffer[0], A.ldim, &B.buffer[0], B.ldim, 
          0, &C.buffer[0], C.ldim );
    }
    else if( B.symmetric )
    {
        blas::Symm
        ( 'R', 'L', C.m, C.n,
          alpha, &B.buffer[0], B.ldim, &A.buffer[0], A.ldim,
          0, &C.buffer[0], C.ldim );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.m, C.n, A.n,
          alpha, &A.buffer[0], A.ldim, &B.buffer[0], B.ldim,
          0, &C.buffer[0], C.ldim );
    }
}

// Low-rank C := alpha A B
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B, 
                      FactorMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
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

// Form a factor matrix from a dense matrix times a factor matrix
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B, 
                      FactorMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.symmetric && A.m != A.n )
        throw std::logic_error("Nonsquare matrices cannot be symmetric.");
#endif
    C.m = A.m;
    C.n = B.n;
    C.r = B.r;
    C.U.resize( C.m*C.r );
    C.V.resize( C.n*C.r );

    // Form C.U := A B.U
    if( A.symmetric )
    {
        blas::Symm
        ( 'L', 'L', C.m, C.r, 
          alpha, &A.buffer[0], A.ldim, &B.U[0], B.m, 0, &C.U[0], C.m );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.m, C.r, A.n,
          alpha, &A.buffer[0], A.ldim, &B.U[0], B.m, 0, &C.U[0], C.m );
    }

    // Form C.V := B.V
    std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
}

// Form a factor matrix from a factor matrix times a dense matrix
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      FactorMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.symmetric && B.m != B.n )
        throw std::logic_error("Nonsquare matrices cannot be symmetric.");
#endif
    C.m = A.m;
    C.n = B.n;
    C.r = A.r;
    C.U.resize( C.m*C.r );
    C.V.resize( C.n*C.r );

    // Form C.U := A.U
    std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );

    // Form C.V := B' V
    if( B.symmetric )
    {
        // Since B is symmetric, B' V = conj(B) V = conj(B conj(V))
        const int size = A.n*A.r;
        std::vector<Scalar> W( size );
        const Scalar* RESTRICT AV = &A.V[0];
        for( int i=0; i<size; ++i )
            W[i] = Conj( AV[i] );

        blas::Symm
        ( 'L', 'L', C.n, C.r, 
          alpha, &B.buffer[0], B.ldim, &W[0], A.n, 0, &C.V[0], C.n );

        Scalar* CV = &C.V[0];
        for( int i=0; i<size; ++i )
            C.V[i] = Conj( C.V[i] );
    }
    else
    {
        blas::Gemm
        ( 'C', 'N', C.n, C.r, B.m,
          alpha, &B.buffer[0], B.ldim, &A.V[0], A.n, 0, &C.V[0], C.n );
    }
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.symmetric && A.m != A.n )
        throw std::logic_error("Nonsquare matrices cannot be symmetric.");
#endif
    C.m = A.m;
    C.n = B.n;
    C.ldim = C.m;
    C.buffer.resize( C.ldim*C.n );
    C.symmetric = false;

    // W := A B.U
    std::vector<Scalar> W( A.m*B.r );
    if( A.symmetric )
    {
        blas::Symm
        ( 'L', 'L', C.m, B.r,
          1, &A.buffer[0], A.ldim, &B.U[0], B.m, 0, &W[0], A.m );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.m, B.r, A.n,
          1, &A.buffer[0], A.ldim, &B.U[0], B.m, 0, &W[0], A.m );
    }
    // C := alpha W B.V' = alpha (A B.U) B.V'
    blas::Gemm
    ( 'N', 'C', C.m, C.n, B.r,
      alpha, &W[0], A.m, &B.V[0], B.n, 0, &C.buffer[0], C.ldim );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.symmetric && B.m != B.n )
        throw std::logic_error("Nonsquare matrices cannot be symmetric.");
#endif
    C.m = A.m;
    C.n = B.n;
    C.ldim = C.m;
    C.buffer.resize( C.ldim*C.n );
    C.symmetric = false;

    if( B.symmetric )
    {
        // Form AVConj := conj(A.V)
        const int size = A.n*A.r;
        const Scalar* RESTRICT AV = &A.V[0];
        std::vector<Scalar> AVConj( size );
        for( int i=0; i<size; ++i )
            AVConj[i] = Conj( AV[i] );
        
        // W := B conj(A.V) = conj(B' A.V) = conj( (A.V' B)' ) = (A.V' B)^T
        std::vector<Scalar> W( B.m*A.r );
        blas::Symm
        ( 'L', 'L', B.m, A.r,
          1, &B.buffer[0], B.ldim, &AVConj[0], A.n, 0, &W[0], B.m );
        // C := alpha A.U W^T =  alpha A.U (A.V' B)
        blas::Gemm
        ( 'N', 'T', C.m, C.n, A.r,
          alpha, &A.U[0], A.m, &W[0], B.m, 0, &C.buffer[0], C.ldim );
    }
    else
    {
        std::vector<Scalar> W( A.r*B.n );
        // W := A.V' B
        blas::Gemm
        ( 'C', 'N', A.r, B.n, A.n,
          1, &A.V[0], A.n, &B.buffer[0], B.ldim, 0, &W[0], A.r );
        // C := alpha A.U W = alpha A.U (A.V' B)
        blas::Gemm
        ( 'N', 'N', C.m, C.n, A.r,
          alpha, &A.U[0], A.m, &W[0], A.r, 0, &C.buffer[0], C.ldim );
    }
}

// Dense C := alpha A B
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A B
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

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a factor matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float>& B, 
                     FactorMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double>& B,
                      FactorMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix< std::complex<float> >& B,
                                   FactorMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix< std::complex<double> >& B,
                                    FactorMatrix< std::complex<double> >& C );

// Form a factor matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix< std::complex<double> >& C );

