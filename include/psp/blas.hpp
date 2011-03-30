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

// Declarations for an external BLAS library
extern "C" {

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

void BLAS(sscal)
( const int* n, const float* alpha, float* x, const int* incx );

void BLAS(dscal)
( const int* n, const double* alpha, double* x, const int* incx );

void BLAS(cscal)
( const int* n, const scomplex* alpha, scomplex* x, const int* incx );

void BLAS(zscal)
( const int* n, const dcomplex* alpha, dcomplex* x, const int* incx );

void BLAS(sgemv)
( const char* trans, const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* x, const int* incx,
  const float* beta,        float* y, const int* incy );

void BLAS(dgemv)
( const char* trans, const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* x, const int* incx,
  const double* beta,        double* y, const int* incy );

void BLAS(cgemv)
( const char* trans, const int* m, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* x, const int* incx,
  const scomplex* beta,        scomplex* y, const int* incy );

void BLAS(zgemv)
( const char* trans, const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* x, const int* incx,
  const dcomplex* beta,        dcomplex* y, const int* incy );

void BLAS(ssymv)
( const char* uplo, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* x, const int* incx,
  const float* beta,        float* y, const int* incy );

void BLAS(dsymv)
( const char* uplo, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* x, const int* incx,
  const double* beta,        double* y, const int* incy );

void BLAS(csymv)
( const char* uplo, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* x, const int* incx,
  const scomplex* beta,        scomplex* y, const int* incy );

void BLAS(zsymv)
( const char* uplo, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* x, const int* incx,
  const dcomplex* beta,        dcomplex* y, const int* incy );

void BLAS(sgemm)
( const char* transa, const char* transb, 
  const int* m, const int* n, const int* k,
  const float* alpha, const float* A, const int* lda,
                      const float* B, const int* ldb,
  const float* beta,        float* C, const int* ldc );

void BLAS(dgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* A, const int* lda,
                       const double* B, const int* ldb,
  const double* beta,        double* C, const int* ldc );

void BLAS(cgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* B, const int* ldb,
  const scomplex* beta,        scomplex* C, const int* ldc );

void BLAS(zgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* B, const int* ldb,
  const dcomplex* beta,        dcomplex* C, const int* ldc );

void BLAS(ssymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* B, const int* ldb,
  const float* beta,        float* C, const int* ldc );

void BLAS(dsymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* B, const int* ldb,
  const double* beta,        double* C, const int* ldc );

void BLAS(csymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* B, const int* ldb,
  const scomplex* beta,        scomplex* C, const int* ldc );

void BLAS(zsymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* B, const int* ldb,
  const dcomplex* beta,        dcomplex* C, const int* ldc );

void BLAS(strmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                            float* B, const int* ldb );

void BLAS(dtrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                             double* B, const int* ldb );

void BLAS(ctrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                               scomplex* B, const int* ldb );

void BLAS(ztrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                               dcomplex* B, const int* ldb );

} // extern "C"

namespace psp {
namespace blas {

//----------------------------------------------------------------------------//
// Scale a vector                                                             //
//----------------------------------------------------------------------------//

inline void Scal
( int n, float alpha, float* x, int incx )
{
    BLAS(sscal)( &n, &alpha, x, &incx );
}

inline void Scal
( int n, double alpha, double* x, int incx )
{
    BLAS(dscal)( &n, &alpha, x, &incx );
}

inline void Scal
( int n, std::complex<float> alpha, std::complex<float>* x, int incx )
{
    BLAS(cscal)( &n, &alpha, x, &incx );
}

inline void Scal
( int n, std::complex<double> alpha, std::complex<double>* x, int incx )
{
    BLAS(zscal)( &n, &alpha, x, &incx );
}

//----------------------------------------------------------------------------//
// General matrix-vector multiplication                                       //
//----------------------------------------------------------------------------//

inline void Gemv
( char trans, int m, int n, 
  float alpha, const float* A, int lda, 
               const float* x, int incx, 
  float beta,        float* y, int incy )
{
    BLAS(sgemv)
    ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Gemv
( char trans, int m, int n, 
  double alpha, const double* A, int lda, 
                const double* x, int incx, 
  double beta,        double* y, int incy )
{
    BLAS(dgemv)
    ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Gemv
( char trans, int m, int n, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                             const std::complex<float>* x, int incx, 
  std::complex<float> beta,        std::complex<float>* y, int incy )
{
    BLAS(cgemv)
    ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Gemv
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

inline void Symv
( char uplo, int n, 
  float alpha, const float* A, int lda,
               const float* x, int incx, 
  float beta,        float* y, int incy )
{
    BLAS(ssymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Symv
( char uplo, int n, 
  double alpha, const double* A, int lda,
                const double* x, int incx, 
  double beta,        double* y, int incy )
{
    BLAS(dsymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Symv
( char uplo, int n, 
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                             const std::complex<float>* x, int incx, 
  std::complex<float> beta,        std::complex<float>* y, int incy )
{
    BLAS(csymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Symv
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

inline void Gemm
( char transa, char transb, int m, int n, int k, 
  float alpha, const float* A, int lda, 
               const float* B, int ldb,
  float beta,        float* C, int ldc )
{
    BLAS(sgemm)
    ( &transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Gemm
( char transa, char transb, int m, int n, int k, 
  double alpha, const double* A, int lda, 
                const double* B, int ldb,
  double beta,        double* C, int ldc )
{
    BLAS(dgemm)
    ( &transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Gemm
( char transa, char transb, int m, int n, int k, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                             const std::complex<float>* B, int ldb,
  std::complex<float> beta,        std::complex<float>* C, int ldc )
{
    BLAS(cgemm)
    ( &transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Gemm
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

inline void Symm
( char side, char uplo, int m, int n, 
  float alpha, const float* A, int lda, 
               const float* B, int ldb,
  float beta,        float* C, int ldc )
{
    BLAS(ssymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Symm
( char side, char uplo, int m, int n, 
  double alpha, const double* A, int lda, 
                const double* B, int ldb,
  double beta,        double* C, int ldc )
{
    BLAS(dsymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Symm
( char side, char uplo, int m, int n,
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                             const std::complex<float>* B, int ldb,
  std::complex<float> beta,        std::complex<float>* C, int ldc )
{
    BLAS(csymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Symm
( char side, char uplo, int m, int n, 
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

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n, 
  float alpha, const float* A, int lda, 
                     float* B, int ldb )
{
    BLAS(strmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n, 
  double alpha, const double* A, int lda, 
                      double* B, int ldb )
{
    BLAS(dtrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n, 
  std::complex<float> alpha, const std::complex<float>* A, int lda, 
                                   std::complex<float>* B, int ldb )
{
    BLAS(ctrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n, 
  std::complex<double> alpha, const std::complex<double>* A, int lda, 
                                    std::complex<double>* B, int ldb )
{
    BLAS(ztrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

} // namespace blas
} // namespace psp

#endif // PSP_BLAS_HPP
