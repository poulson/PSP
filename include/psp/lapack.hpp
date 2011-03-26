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
#include <sstream>
#include <stdexcept>

#if defined(LAPACK_POST)
#define LAPACK(name) name ## _
#else
#define LAPACK(name) name
#endif

//----------------------------------------------------------------------------//
// Routines for QR factorizations and Q application                           //
//----------------------------------------------------------------------------//

inline void psp::lapack::QR
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

inline void psp::lapack::QR
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

inline void psp::lapack::ApplyQ
( char side, char trans, int m, int n, int k, 
  const std::complex<float>* A, int lda,
        std::complex<float>* tau, 
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

inline void psp::lapack::ApplyQ
( char side, char trans, int m, int n, int k, 
  const std::complex<double>* A, int lda,
        std::complex<double>* tau, 
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
// SVD                                                                        //
//----------------------------------------------------------------------------//

inline void psp::lapack::SVD
( char jobu, char jobvt, int m, int n, 
  std::complex<float>* A, int lda,
  float* s, 
  std::complex<float>* U, int ldu, 
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork, 
  float* rwork )
{
    int info;
    LAPACK(cgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
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

inline void psp::lapack::SVD
( char jobu, char jobvt, int m, int n, 
  std::complex<double>* A, int lda,
  double* s, 
  std::complex<double>* U, int ldu, 
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork, 
  double* rwork )
{
    int info;
    LAPACK(zgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
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

inline void psp::lapack::PseudoInverse
( int m, int n, 
  std::complex<float>* A, int lda,
  float* s, 
  std::complex<float>* U, int ldu, 
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork, 
  float* rwork )
{
    char jobu = 'S'; 
    char jobvt = 'S'; 
    int info;
    LAPACK(cgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, cgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif

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
            const PetscReal invSigma = 1 / s[j];
            // Split our approach based on whether or not the source and 
            // destination buffers are the same
            if( packedColumns == j )
            {
                PetscScalar* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i ) 
                    UCol[i] *= invSigma;
            }
            else
            {
                // Form the scaled column of U in its packed location
                PetscScalar* RESTRICT UPackedCol = &U[packedColumns*ldu];
                const PetscScalar* RESTRICT UCol = &U[j*ldu];
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
            const PetscReal invSigma = 1 / s[j];
            PetscScalar* RESTRICT UCol = &U[j*ldu];
            for( int i=0; i<m; ++i ) 
                UCol[i] *= invSigma;
        }
        else
        {
            // Scale the j'th column by 0
            std::memset( &U[j*ldu], 0, m*sizeof(PetscScalar) );
        }
    }

    // Form A := U V^H, where U has been scaled
    blas::Gemm
    ( 'N', 'N', m, n, k, 1, U, ldu, VH, ldvh, 0, A, lda );
#endif // PACK_DURING_PSEUDO_INVERSE
}

inline void psp::lapack::PseudoInverse
( int m, int n, 
  std::complex<double>* A, int lda,
  double* s, 
  std::complex<double>* U, int ldu, 
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork, 
  double* rwork )
{
    char jobu = 'S'; 
    char jobvt = 'S'; 
    int info;
    LAPACK(zgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, zgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif

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
            const PetscReal invSigma = 1 / s[j];
            // Split our approach based on whether or not the source and 
            // destination buffers are the same
            if( packedColumns == j )
            {
                PetscScalar* RESTRICT UCol = &U[j*ldu];
                for( int i=0; i<m; ++i ) 
                    UCol[i] *= invSigma;
            }
            else
            {
                // Form the scaled column of U in its packed location
                PetscScalar* RESTRICT UPackedCol = &U[packedColumns*ldu];
                const PetscScalar* RESTRICT UCol = &U[j*ldu];
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
            const PetscReal invSigma = 1 / s[j];
            PetscScalar* RESTRICT UCol = &U[j*ldu];
            for( int i=0; i<m; ++i ) 
                UCol[i] *= invSigma;
        }
        else
        {
            // Scale the j'th column by 0
            std::memset( &U[j*ldu], 0, m*sizeof(PetscScalar) );
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

inline void psp::lapack::LU
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

inline void psp::lapack::LU
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

inline void psp::lapack::InvertLU
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

inline void psp::lapack::InvertLU
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

inline void psp::lapack::LDLT
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

inline void psp::lapack::LDLT
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

inline void psp::lapack::InvertLDLT
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

inline void psp::lapack::InvertLDLT
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

#endif // PSP_LAPACK_HPP
