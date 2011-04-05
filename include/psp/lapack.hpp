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
#ifndef PSP_LAPACK_HPP
#define PSP_LAPACK_HPP 1

#include "psp/config.h"
#include <complex>
#include <cstring>
#include <sstream>
#include <stdexcept>

#if defined(LAPACK_POST)
#define LAPACK(name) name ## _
#else
#define LAPACK(name) name
#endif

extern "C" {

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

void LAPACK(sgeqrf)
( const int* m, const int* n,
  float* A, const int* lda,
  float* tau,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgeqrf)
( const int* m, const int* n,
  double* A, const int* lda,
  double* tau,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgeqrf)
( const int* m, const int* n,
  scomplex* A, const int* lda,
  scomplex* tau,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zgeqrf)
( const int* m, const int* n,
  dcomplex* A, const int* lda,
  dcomplex* tau,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(sgeqp3)
( const int* m, const int* n,
  float* A, const int* lda,
  int* jpvt,
  float* tau,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgeqp3)
( const int* m, const int* n,
  double* A, const int* lda,
  int* jpvt,
  double* tau,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgeqp3)
( const int* m, const int* n,
  scomplex* A, const int* lda,
  int* jpvt,
  scomplex* tau,
  scomplex* work, const int* lwork,
  float* rwork,
  int* info );

void LAPACK(zgeqp3)
( const int* m, const int* n,
  dcomplex* A, const int* lda,
  int* jpvt,
  dcomplex* tau,
  dcomplex* work, const int* lwork,
  double* rwork,
  int* info );

void LAPACK(sormqr)
( const char* side, const char* trans, 
  const int* m, const int* n, const int* k,
  const float* A, const int* lda,
  const float* tau,
  float* C, const int* ldc,
  float* work, const int* lwork,
  int* info );

void LAPACK(dormqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const double* A, const int* lda,
  const double* tau,
  double* C, const int* ldc,
  double* work, const int* lwork,
  int* info );

void LAPACK(cunmqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const scomplex* A, const int* lda,
  const scomplex* tau,
  scomplex* C, const int* ldc,
  scomplex* work, const int* lwork, 
  int* info );

void LAPACK(zunmqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const dcomplex* A, const int* lda,
  const dcomplex* tau,
  dcomplex* C, const int* ldc,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(sorgqr)
( const int* m, const int* n, const int* k,
        float* A, const int* lda,
  const float* tau,
        float* work, const int* lwork,
  int* info );

void LAPACK(dorgqr)
( const int* m, const int* n, const int* k,
        double* A, const int* lda,
  const double* tau,
        double* work, const int* lwork,
  int* info );

void LAPACK(cungqr)
( const int* m, const int* n, const int* k,
        scomplex* A, const int* lda,
  const scomplex* tau,
        scomplex* work, const int* lwork, 
  int* info );

void LAPACK(zungqr)
( const int* m, const int* n, const int* k,
        dcomplex* A, const int* lda,
  const dcomplex* tau,
        dcomplex* work, const int* lwork,
  int* info );

void LAPACK(sgesvd)
( const char* jobu, const char* jobvh, 
  const int* m, const int* n,
  float* A, const int* lda,
  float* s,
  float* U, const int* ldu,
  float* VH, const int* ldvh,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  double* A, const int* lda,
  double* s,
  double* U, const int* ldu,
  double* VH, const int* ldvh,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  scomplex* A, const int* lda,
  float* s,
  scomplex* U, const int* ldu,
  scomplex* VH, const int* ldvh,
  scomplex* work, const int* lwork,
  float* rwork,
  int* info );

void LAPACK(zgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  dcomplex* A, const int* lda,
  double* s,
  dcomplex* U, const int* ldu,
  dcomplex* VH, const int* ldvh,
  dcomplex* work, const int* lwork,
  double* rwork,
  int* info );

float LAPACK(slamch)( const char* cmach );
double LAPACK(dlamch)( const char* cmach );

void LAPACK(sgetrf)
( const int* m, const int* n, 
  float* A, const int* lda, 
  int* ipiv, 
  int* info );

void LAPACK(dgetrf)
( const int* m, const int* n, 
  double* A, const int* lda, 
  int* ipiv, 
  int* info );

void LAPACK(cgetrf)
( const int* m, const int* n, 
  scomplex* A, const int* lda, 
  int* ipiv, 
  int* info );

void LAPACK(zgetrf)
( const int* m, const int* n,
  dcomplex* A, const int* lda, 
  int* ipiv, 
  int* info );

void LAPACK(sgetri)
( const int* n, 
  float* A, const int* lda, 
  const int* ipiv, 
  float* work, const int* lwork, 
  int* info );

void LAPACK(dgetri)
( const int* n,
  double* A, const int* lda,
  const int* ipiv,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgetri)
( const int* n,
  scomplex* A, const int* lda,
  const int* ipiv,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zgetri)
( const int* n,
  dcomplex* A, const int* lda,
  const int* ipiv,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(ssytrf)
( const char* uplo, const int* n,
  float* A, const int* lda,
  int* ipiv,
  float* work, const int* lwork,
  int* info );

void LAPACK(dsytrf)
( const char* uplo, const int* n,
  double* A, const int* lda,
  int* ipiv,
  double* work, const int* lwork,
  int* info );

void LAPACK(csytrf)
( const char* uplo, const int* n,
  scomplex* A, const int* lda,
  int* ipiv,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zsytrf)
( const char* uplo, const int* n,
  dcomplex* A, const int* lda,
  int* ipiv,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(ssytri)
( const char* uplo, const int* n,
  float* A, const int* lda,
  int* ipiv,
  float* work,
  int* info );

void LAPACK(dsytri)
( const char* uplo, const int* n,
  double* A, const int* lda,
  int* ipiv,
  double* work,
  int* info );

void LAPACK(csytri)
( const char* uplo, const int* n,
  scomplex* A, const int* lda,
  int* ipiv,
  scomplex* work,
  int* info );

void LAPACK(zsytri)
( const char* uplo, const int* n,
  dcomplex* A, const int* lda,
  int* ipiv,
  dcomplex* work,
  int* info );

} // extern "C"

namespace psp {
namespace lapack {

//----------------------------------------------------------------------------//
// Unpivoted QR                                                               //
//----------------------------------------------------------------------------//

inline void QR
( int m, int n, 
  float* A, int lda, 
  float* tau, 
  float* work, int lwork )
{
    int info;
    LAPACK(sgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, sgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void QR
( int m, int n, 
  double* A, int lda, 
  double* tau, 
  double* work, int lwork )
{
    int info;
    LAPACK(dgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, dgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void QR
( int m, int n, 
  std::complex<float>* A, int lda, 
  std::complex<float>* tau, 
  std::complex<float>* work, int lwork )
{
    int info;
    LAPACK(cgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, cgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void QR
( int m, int n, 
  std::complex<double>* A, int lda, 
  std::complex<double>* tau, 
  std::complex<double>* work, int lwork )
{
    int info;
    LAPACK(zgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, zgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Pivoted QR                                                                 //
//----------------------------------------------------------------------------//

inline void PivotedQR
( int m, int n, 
  float* A, int lda, 
  int* jpvt,
  float* tau, 
  float* work, int lwork )
{
    int info;
    LAPACK(sgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, sgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void PivotedQR
( int m, int n, 
  double* A, int lda, 
  int* jpvt,
  double* tau, 
  double* work, int lwork )
{
    int info;
    LAPACK(dgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, dgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void PivotedQR
( int m, int n, 
  std::complex<float>* A, int lda, 
  int* jpvt,
  std::complex<float>* tau, 
  std::complex<float>* work, int lwork,
  float* rwork )
{
    int info;
    LAPACK(cgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, cgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void PivotedQR
( int m, int n, 
  std::complex<double>* A, int lda, 
  int* jpvt,
  std::complex<double>* tau, 
  std::complex<double>* work, int lwork,
  double* rwork )
{
    int info;
    LAPACK(zgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, zgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Apply Q from a QR factorization                                            //
//----------------------------------------------------------------------------//

inline void ApplyQ
( char side, char trans, int m, int n, int k, 
  const float* A, int lda,
  const float* tau, 
        float* C, int ldc,
        float* work, int lwork )
{
    // Convert the more general complex option to the real case
    if( trans == 'C' || trans == 'c' )
        trans = 'T';

    int info;
    LAPACK(sormqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, sormqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void ApplyQ
( char side, char trans, int m, int n, int k, 
  const double* A, int lda,
  const double* tau, 
        double* C, int ldc,
        double* work, int lwork )
{
    // Convert the more general complex option to the real case
    if( trans == 'C' || trans == 'c' )
        trans = 'T';

    int info;
    LAPACK(dormqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, dormqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void ApplyQ
( char side, char trans, int m, int n, int k, 
  const std::complex<float>* A, int lda,
  const std::complex<float>* tau, 
        std::complex<float>* C, int ldc,
        std::complex<float>* work, int lwork )
{
    int info;
    LAPACK(cunmqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, cunmqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void ApplyQ
( char side, char trans, int m, int n, int k, 
  const std::complex<double>* A, int lda,
  const std::complex<double>* tau, 
        std::complex<double>* C, int ldc,
        std::complex<double>* work, int lwork )
{
    int info;
    LAPACK(zunmqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, zunmqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Form Q from a QR factorization                                             //
//----------------------------------------------------------------------------//

inline void FormQ
( int m, int n, int k, 
        float* A, int lda,
  const float* tau, 
        float* work, int lwork ) 
{
    int info;
    LAPACK(sorgqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, sorgqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void FormQ
( int m, int n, int k,
        double* A, int lda,
  const double* tau,
        double* work, int lwork )
{
    int info;
    LAPACK(dorgqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, dorgqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void FormQ
( int m, int n, int k,
        std::complex<float>* A, int lda,
  const std::complex<float>* tau,
        std::complex<float>* work, int lwork )
{
    int info;
    LAPACK(cungqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, cungqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void FormQ
( int m, int n, int k,
        std::complex<double>* A, int lda,
  const std::complex<double>* tau,
        std::complex<double>* work, int lwork )
{
    int info;
    LAPACK(zungqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, zungqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// SVD                                                                        //
//----------------------------------------------------------------------------//

inline void SVD
( char jobu, char jobvh, int m, int n, 
  float* A, int lda,
  float* s, 
  float* U, int ldu, 
  float* VH, int ldvh,
  float* work, int lwork )
{
    int info;
    LAPACK(sgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, sgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n, 
  double* A, int lda,
  double* s, 
  double* U, int ldu, 
  double* VH, int ldvh,
  double* work, int lwork )
{
    int info;
    LAPACK(dgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, dgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n, 
  std::complex<float>* A, int lda,
  float* s, 
  std::complex<float>* U, int ldu, 
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork, 
  float* rwork )
{
    int info;
    LAPACK(cgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, cgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n, 
  std::complex<double>* A, int lda,
  double* s, 
  std::complex<double>* U, int ldu, 
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork, 
  double* rwork )
{
    int info;
    LAPACK(zgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, zgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Pseudo-inverse (using an SVD)                                              //
//----------------------------------------------------------------------------//

inline void PseudoInverse
( int m, int n, 
  float* A, int lda,
  float* s, 
  float* U, int ldu, 
  float* VH, int ldvh,
  float* work, int lwork )
{
    lapack::SVD( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork );

    // Get the safe minimum for our singular value thresholding
    char cmach = 'S';
    const float safeMin = LAPACK(slamch)( &cmach );

#if defined(PACK_DURING_PSEUDO_INVERSE)
    // Combine the inversion of the sufficiently large singular values with 
    // the scaling of U. We can skip the columns of U that correspond to 
    // numerically zero singular values
    const int k = std::min( m, n );
    int packedColumns = 0;
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            const float invSigma = 1 / s[j];
            // Split our approach based on whether or not the source and 
            // destination buffers are the same
            if( packedColumns == j )
            {
                float* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i ) 
                    UCol[i] *= invSigma;
            }
            else
            {
                // Form the scaled column of U in its packed location
                float* RESTRICT UPackedCol = &U[packedColumns*ldu];
                const float* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i )
                    UPackedCol[i] = invSigma*UCol[i];
                // Since we moved a column of U, we have to move the
                // corresponding column of V as well. This may ruin the benefit
                // of the packed approach
                for( int i=0; i<n; ++i )
                    VH[packedColumns+i*ldvh] = VH[j+i*ldvh];
            }
            ++packedColumns;
        }
    }

    // Form A := U V^H, where U and V have been compressed
    blas::Gemm
    ( 'N', 'N', m, n, packedColumns, 1, U, ldu, VH, ldvh, 0, A, lda );
#else
    // Scale the columns of U using thresholded inversion of the singular values
    const int k = std::min( m, n );
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            // Scale the j'th column by 1/s[j]
            const float invSigma = 1 / s[j];
            float* RESTRICT UCol = &U[j*ldu];
            for( int i=0; i<m; ++i ) 
                UCol[i] *= invSigma;
        }
        else
        {
            // Scale the j'th column by 0
            std::memset( &U[j*ldu], 0, m*sizeof(float) );
        }
    }

    // Form A := U V^H, where U has been scaled
    blas::Gemm
    ( 'N', 'N', m, n, k, 1, U, ldu, VH, ldvh, 0, A, lda );
#endif // PACK_DURING_PSEUDO_INVERSE
}

inline void PseudoInverse
( int m, int n, 
  double* A, int lda,
  double* s, 
  double* U, int ldu, 
  double* VH, int ldvh,
  double* work, int lwork )
{
    lapack::SVD( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork );

    // Get the safe minimum for our singular value thresholding
    char cmach = 'S';
    const double safeMin = LAPACK(dlamch)( &cmach );

#if defined(PACK_DURING_PSEUDO_INVERSE)
    // Combine the inversion of the sufficiently large singular values with 
    // the scaling of U. We can skip the columns of U that correspond to 
    // numerically zero singular values
    const int k = std::min( m, n );
    int packedColumns = 0;
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            const double invSigma = 1 / s[j];
            // Split our approach based on whether or not the source and 
            // destination buffers are the same
            if( packedColumns == j )
            {
                double* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i ) 
                    UCol[i] *= invSigma;
            }
            else
            {
                // Form the scaled column of U in its packed location
                double* RESTRICT UPackedCol = &U[packedColumns*ldu];
                const double* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i )
                    UPackedCol[i] = invSigma*UCol[i];
                // Since we moved a column of U, we have to move the
                // corresponding column of V as well. This may ruin the benefit
                // of the packed approach
                for( int i=0; i<n; ++i )
                    VH[packedColumns+i*ldvh] = VH[j+i*ldvh];
            }
            ++packedColumns;
        }
    }

    // Form A := U V^H, where U and V have been compressed
    blas::Gemm
    ( 'N', 'N', m, n, packedColumns, 1, U, ldu, VH, ldvh, 0, A, lda );
#else
    // Scale the columns of U using thresholded inversion of the singular values
    const int k = std::min( m, n );
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            // Scale the j'th column by 1/s[j]
            const double invSigma = 1 / s[j];
            double* RESTRICT UCol = &U[j*ldu];
            for( int i=0; i<m; ++i ) 
                UCol[i] *= invSigma;
        }
        else
        {
            // Scale the j'th column by 0
            std::memset( &U[j*ldu], 0, m*sizeof(double) );
        }
    }

    // Form A := U V^H, where U has been scaled
    blas::Gemm
    ( 'N', 'N', m, n, k, 1, U, ldu, VH, ldvh, 0, A, lda );
#endif // PACK_DURING_PSEUDO_INVERSE
}

inline void PseudoInverse
( int m, int n, 
  std::complex<float>* A, int lda,
  float* s, 
  std::complex<float>* U, int ldu, 
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork, 
  float* rwork )
{
    lapack::SVD
    ( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, rwork );

    // Get the safe minimum for our singular value thresholding
    char cmach = 'S';
    const float safeMin = LAPACK(slamch)( &cmach );

#if defined(PACK_DURING_PSEUDO_INVERSE)
    // Combine the inversion of the sufficiently large singular values with 
    // the scaling of U. We can skip the columns of U that correspond to 
    // numerically zero singular values
    const int k = std::min( m, n );
    int packedColumns = 0;
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            const float invSigma = 1 / s[j];
            // Split our approach based on whether or not the source and 
            // destination buffers are the same
            if( packedColumns == j )
            {
                std::complex<float>* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i ) 
                    UCol[i] *= invSigma;
            }
            else
            {
                // Form the scaled column of U in its packed location
                std::complex<float>* RESTRICT 
                    UPackedCol = &U[packedColumns*ldu];
                const std::complex<float>* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i )
                    UPackedCol[i] = invSigma*UCol[i];
                // Since we moved a column of U, we have to move the
                // corresponding column of V as well. This may ruin the benefit
                // of the packed approach
                for( int i=0; i<n; ++i )
                    VH[packedColumns+i*ldvh] = VH[j+i*ldvh];
            }
            ++packedColumns;
        }
    }

    // Form A := U V^H, where U and V have been compressed
    blas::Gemm
    ( 'N', 'N', m, n, packedColumns, 1, U, ldu, VH, ldvh, 0, A, lda );
#else
    // Scale the columns of U using thresholded inversion of the singular values
    const int k = std::min( m, n );
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            // Scale the j'th column by 1/s[j]
            const float invSigma = 1 / s[j];
            std::complex<float>* RESTRICT UCol = &U[j*ldu];
            for( int i=0; i<m; ++i ) 
                UCol[i] *= invSigma;
        }
        else
        {
            // Scale the j'th column by 0
            std::memset( &U[j*ldu], 0, m*sizeof(std::complex<float>) );
        }
    }

    // Form A := U V^H, where U has been scaled
    blas::Gemm
    ( 'N', 'N', m, n, k, 1, U, ldu, VH, ldvh, 0, A, lda );
#endif // PACK_DURING_PSEUDO_INVERSE
}

inline void PseudoInverse
( int m, int n, 
  std::complex<double>* A, int lda,
  double* s, 
  std::complex<double>* U, int ldu, 
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork, 
  double* rwork )
{
    lapack::SVD
    ( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, rwork );

    // Get the safe minimum for our singular value thresholding
    char cmach = 'S';
    const double safeMin = LAPACK(dlamch)( &cmach );

#if defined(PACK_DURING_PSEUDO_INVERSE)
    // Combine the inversion of the sufficiently large singular values with 
    // the scaling of U. We can skip the columns of U that correspond to 
    // numerically zero singular values
    const int k = std::min( m, n );
    int packedColumns = 0;
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            const double invSigma = 1 / s[j];
            // Split our approach based on whether or not the source and 
            // destination buffers are the same
            if( packedColumns == j )
            {
                std::complex<double>* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i ) 
                    UCol[i] *= invSigma;
            }
            else
            {
                // Form the scaled column of U in its packed location
                std::complex<double>* RESTRICT 
                    UPackedCol = &U[packedColumns*ldu];
                const std::complex<double>* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i )
                    UPackedCol[i] = invSigma*UCol[i];
                // Since we moved a column of U, we have to move the
                // corresponding column of V as well. This may ruin the benefit
                // of the packed approach
                for( int i=0; i<n; ++i )
                    VH[packedColumns+i*ldvh] = VH[j+i*ldvh];
            }
            ++packedColumns;
        }
    }

    // Form A := U V^H, where U and V have been compressed
    blas::Gemm
    ( 'N', 'N', m, n, packedColumns, 1, U, ldu, VH, ldvh, 0, A, lda );
#else
    // Scale the columns of U using thresholded inversion of the singular values
    const int k = std::min( m, n );
    for( int j=0; j<k; ++j )
    {
        if( s[j] >= safeMin )
        {
            // Scale the j'th column by 1/s[j]
            const double invSigma = 1 / s[j];
            std::complex<double>* RESTRICT UCol = &U[j*ldu];
            for( int i=0; i<m; ++i ) 
                UCol[i] *= invSigma;
        }
        else
        {
            // Scale the j'th column by 0
            std::memset( &U[j*ldu], 0, m*sizeof(std::complex<double>) );
        }
    }

    // Form A := U V^H, where U has been scaled
    blas::Gemm
    ( 'N', 'N', m, n, k, 1, U, ldu, VH, ldvh, 0, A, lda );
#endif // PACK_DURING_PSEUDO_INVERSE
}

//----------------------------------------------------------------------------//
// LU Factorization                                                           //
//----------------------------------------------------------------------------//

inline void LU
( int m, int n, float* A, int lda, int* ipiv )
{
    int info;
    LAPACK(sgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, sgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LU
( int m, int n, double* A, int lda, int* ipiv )
{
    int info;
    LAPACK(dgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, dgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LU
( int m, int n, std::complex<float>* A, int lda, int* ipiv )
{
    int info;
    LAPACK(cgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, cgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LU
( int m, int n, std::complex<double>* A, int lda, int* ipiv )
{
    int info;
    LAPACK(zgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, zgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Invert an LU factorization                                                 //
//----------------------------------------------------------------------------//

inline void InvertLU
( int n, float* A, int lda, int* ipiv, 
  float* work, int lwork )
{
    int info;
    LAPACK(sgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, sgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLU
( int n, double* A, int lda, int* ipiv, 
  double* work, int lwork )
{
    int info;
    LAPACK(dgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, dgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLU
( int n, std::complex<float>* A, int lda, int* ipiv, 
  std::complex<float>* work, int lwork )
{
    int info;
    LAPACK(cgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, cgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLU
( int n, std::complex<double>* A, int lda, int* ipiv, 
  std::complex<double>* work, int lwork )
{
    int info;
    LAPACK(zgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, zgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// LDL^T Factorization                                                        //
//----------------------------------------------------------------------------//

inline void LDLT
( char uplo, int n, float* A, int lda, int* ipiv, 
  float* work, int lwork )
{
    int info;
    LAPACK(ssytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, ssytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LDLT
( char uplo, int n, double* A, int lda, int* ipiv, 
  double* work, int lwork )
{
    int info;
    LAPACK(dsytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, dsytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LDLT
( char uplo, int n, std::complex<float>* A, int lda, int* ipiv, 
  std::complex<float>* work, int lwork )
{
    int info;
    LAPACK(csytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, csytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LDLT
( char uplo, int n, std::complex<double>* A, int lda, int* ipiv, 
  std::complex<double>* work, int lwork )
{
    int info;
    LAPACK(zsytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, zsytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Invert an LDL^T factorization                                              //
//----------------------------------------------------------------------------//

inline void InvertLDLT
( char uplo, int n, float* A, int lda, int* ipiv, 
  float* work )
{
    int info;
    LAPACK(ssytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, ssytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLDLT
( char uplo, int n, double* A, int lda, int* ipiv, 
  double* work )
{
    int info;
    LAPACK(dsytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, dsytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLDLT
( char uplo, int n, std::complex<float>* A, int lda, int* ipiv, 
  std::complex<float>* work )
{
    int info;
    LAPACK(csytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, csytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLDLT
( char uplo, int n, std::complex<double>* A, int lda, int* ipiv, 
  std::complex<double>* work )
{
    int info;
    LAPACK(zsytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, zsytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

} // namespace lapack
} // namespace psp

#endif // PSP_LAPACK_HPP
