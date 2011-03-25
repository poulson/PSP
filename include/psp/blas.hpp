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
#ifndef PSP_BLAS_HPP
#define PSP_BLAS_HPP 1

#include "psp/config.h"
#include <complex>
#include <sstream>
#include <stdexcept>

#if defined(BLAS_POST)
#define BLAS(name) name ## _
#else
#define BLAS(name) name
#endif

//----------------------------------------------------------------------------//
// General matrix-vector multiplication                                       //
//----------------------------------------------------------------------------//

void psp::blas::Gemv
( char trans, int m, int n, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                             const std::complex<float>* x, int incx, 
  std::complex<float> beta,        std::complex<float>* y, int incy )
{
    BLAS(cgemv)
    ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

void psp::blas::Gemv
( char trans, int m, int n, 
  std::complex<double> alpha, const std::complex<double>* A, int lda, 
                              const std::complex<double>* x, int incx, 
  std::complex<double> beta,        std::complex<double>* y, int incy )
{
    BLAS(zgemv)
    ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

//----------------------------------------------------------------------------//
// Symmetric matrix-vector multiplication                                     //
//----------------------------------------------------------------------------//

void psp::blas::Symv
( char uplo, int n, 
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                             const std::complex<float>* x, int incx, 
  std::complex<float> beta,        std::complex<float>* y, int incy )
{
    BLAS(csymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

void psp::blas::Symv
( char uplo, int n, 
  std::complex<double> alpha, const std::complex<double>* A, int lda,
                              const std::complex<double>* x, int incx, 
  std::complex<double> beta,        std::complex<double>* y, int incy )
{
    BLAS(zsymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

//----------------------------------------------------------------------------//
// General matrix-matrix multiplication                                       //
//----------------------------------------------------------------------------//

void psp::blas::Gemm
( char transa, char transb, int m, int n, int k, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                             const std::complex<float>* B, int ldb,
  std::complex<float> beta,        std::complex<float>* C, int ldc )
{
    BLAS(cgemm)
    ( &transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void psp::blas::Gemm
( char transa, char transb, int m, int n, int k, 
  std::complex<double> alpha, const std::complex<double>* A, int lda, 
                              const std::complex<double>* B, int ldb,
  std::complex<double> beta,        std::complex<double>* C, int ldc )
{
    BLAS(zgemm)
    ( &transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

//----------------------------------------------------------------------------//
// Symmetric matrix-matrix multiplication                                     //
//----------------------------------------------------------------------------//

void psp::blas::Symm
( char side, char uplo, int m, int n, std::complex<float> alpha, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                             const std::complex<float>* B, int ldb,
  std::complex<float> beta,        std::complex<float>* C, int ldc )
{
    BLAS(csymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void psp::blas::Symm
( char side, char uplo, int m, int n, std::complex<double> alpha, 
  std::complex<double> alpha, const std::complex<double>* A, int lda, 
                              const std::complex<double>* B, int ldb,
  std::complex<double> beta,        std::complex<double>* C, int ldc )
{
    BLAS(zsymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

//----------------------------------------------------------------------------//
// Triangular matrix-matrix multiplication                                    //
//----------------------------------------------------------------------------//

void psp::blas::Trmm
( char side, char uplo, char transa, char diag, int m, int n, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                                   std::complex<float>* B, int ldb )
{
    BLAS(ctrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

void psp::blas::Trmm
( char side, char uplo, char transa, char diag, int m, int n, 
  std::complex<double> alpha, const std::complex<double>* A, int lda, 
                                    std::complex<double>* B, int ldb )
{
    BLAS(ztrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

#endif // PSP_BLAS_HPP
