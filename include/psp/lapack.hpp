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

void psp::lapack::QR
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

void psp::lapack::QR
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

void psp::lapack::ApplyQ
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

void psp::lapack::ApplyQ
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

void psp::lapack::SVD
( char jobu, char jobvt, int m, int n, 
  std::complex<float>* A, int lda,
  float* s, 
  std::complex<float>* U, int ldu, 
  std::complex<float>* VT, int ldvt,
  std::complex<float>* work, int lwork, 
  float* rwork )
{
    int info;
    LAPACK(cgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VT, &ldvt, work, &lwork,
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

void psp::lapack::SVD
( char jobu, char jobvt, int m, int n, 
  std::complex<double>* A, int lda,
  double* s, 
  std::complex<double>* U, int ldu, 
  std::complex<double>* VT, int ldvt,
  std::complex<double>* work, int lwork, 
  double* rwork )
{
    int info;
    LAPACK(zgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VT, &ldvt, work, &lwork,
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
// LU Factorization                                                           //
//----------------------------------------------------------------------------//

void psp::lapack::LU
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

void psp::lapack::LU
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
// LDL^T Factorization                                                        //
//----------------------------------------------------------------------------//

void psp::lapack::LDLT
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

void psp::lapack::LDLT
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
// General Inversion                                                          //
//----------------------------------------------------------------------------//

void psp::lapack::InvertLU
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

void psp::lapack::InvertLU
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
// Symmetric Inversion                                                        //
//----------------------------------------------------------------------------//

void psp::lapack::InvertLDLT
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

void psp::lapack::InvertLDLT
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
