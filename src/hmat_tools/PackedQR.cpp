/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 1992-2008 The University of Tennessee
   All rights reserved.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This file is partially based upon the LAPACK routines dlarfg.f and zlarfg.f.

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

namespace {

// Compute a Householder vector in-place and return the tau factor

template<typename Real>
Real Householder( const int m, Real* buffer )
{
    if( m == 1 )
    {
        buffer[0] = -buffer[0];
        return (Real)2;
    }

    Real alpha = buffer[0];
    Real norm = psp::blas::Nrm2( m-1, &buffer[1], 1 );

    Real beta;
    if( alpha <= 0 )
        beta = psp::lapack::SafeNorm( alpha, norm );
    else
        beta = -psp::lapack::SafeNorm( alpha, norm );
    
    // Avoid overflow by scaling the vector
    const Real safeMin = psp::lapack::MachineSafeMin<Real>() /
                         psp::lapack::MachineEpsilon<Real>();
    int count = 0;
    if( psp::Abs(beta) < safeMin )
    {
        Real invOfSafeMin = static_cast<Real>(1) / safeMin;
        do
        {
            ++count;
            psp::blas::Scal( m-1, invOfSafeMin, &buffer[1], 1 );
            alpha *= invOfSafeMin;
            beta *= invOfSafeMin;
        } while( psp::Abs( beta ) < safeMin );

        norm = psp::blas::Nrm2( m-1, &buffer[1], 1 );
        if( alpha <= 0 ) 
            beta = psp::lapack::SafeNorm( alpha, norm );
        else
            beta = -psp::lapack::SafeNorm( alpha, norm );
    }

    Real tau = ( beta - alpha ) / beta;
    psp::blas::Scal( m-1, static_cast<Real>(1)/(alpha-beta), &buffer[1], 1 );

    // Rescale the vector
    for( int j=0; j<count; ++j )
        beta *= safeMin;
    buffer[0] = beta;

    return tau;
}

template<typename Real>
std::complex<Real> Householder( const int m, std::complex<Real>* buffer )
{
    typedef std::complex<Real> Scalar;

    Scalar alpha = buffer[0];
    Real norm = psp::blas::Nrm2( m-1, &buffer[1], 1 );

    if( norm == 0 && imag(alpha) == (Real)0 )
    {
        buffer[0] = -buffer[0];
        return (Real)2;
    }

    Real beta;
    if( real(alpha) <= 0 )
        beta = psp::lapack::SafeNorm( real(alpha), imag(alpha), norm );
    else
        beta = -psp::lapack::SafeNorm( real(alpha), imag(alpha), norm );
    
    // Avoid overflow by scaling the vector
    const Real safeMin = psp::lapack::MachineSafeMin<Real>() /
                         psp::lapack::MachineEpsilon<Real>();
    int count = 0;
    if( psp::Abs(beta) < safeMin )
    {
        Real invOfSafeMin = static_cast<Real>(1) / safeMin;
        do
        {
            ++count;
            psp::blas::Scal( m-1, (Scalar)invOfSafeMin, &buffer[1], 1 );
            alpha *= invOfSafeMin;
            beta *= invOfSafeMin;
        } while( psp::Abs( beta ) < safeMin );

        norm = psp::blas::Nrm2( m-1, &buffer[1], 1 );
        if( real(alpha) <= 0 ) 
            beta = psp::lapack::SafeNorm( real(alpha), imag(alpha), norm );
        else
            beta = -psp::lapack::SafeNorm( real(alpha), imag(alpha), norm );
    }

    Scalar tau = Scalar( (beta-real(alpha))/beta, -imag(alpha)/beta );
    psp::blas::Scal( m-1, static_cast<Scalar>(1)/(alpha-beta), &buffer[1], 1 );

    // Rescale the vector
    for( int j=0; j<count; ++j )
        beta *= safeMin;
    buffer[0] = beta;

    return tau;
}

}

// Perform a QR factorization of size 2r x r where only the upper triangles
// of the matrix are stored, and the nonzeros are packed columnwise.
//
// The work buffer must be of size r-1.
//
// NOTE: This has _NOT_ yet been verified/debugged.
template<typename Scalar>
void psp::hmat_tools::PackedQR
( const int r, Scalar* RESTRICT A, Scalar* RESTRICT work )
{
    // Initialize the pointer at the first diagonal value
    Scalar* diag = A;

    for( int j=0; j<r; ++j )
    {
        // Compute the Householder vector, v, and scalar, tau, in-place
        const Scalar tau = Householder( j+2, &diag[0] );

        // Form z := A(I_j,j+1:n-1)' v in the work vector 
        int offset = 1; // start off pointing at lower-triangle of j'th col
        for( int i=0; i<(r-(j+1)); ++i )
        {
            // Update offset from lower-triangle of (j+i)'th col such that
            // diag[offset] = A(j,j+i+1)
            offset += (j+i+1) + j;

            // z[i] := Conj(A(j,j+1)) v(0) = Conj(A(j,j+1))
            work[i] = Conj(diag[offset]);

            // Move to the lower-triangle of the (j+i+1)'th column of A
            offset += i+1;

            // Traverse over this col of the lower triangle
            for( int k=0; k<j+1; ++k )
                work[i] += psp::Conj(diag[offset+k])*diag[k+1];
        }

        // A(I_j,j+1:n-1) -= conj(tau) v z'
        offset = 1; // start off pointing at lower-triangle of j'th col
        for( int i=0; i<(r-(i+1)); ++i )
        {
            const Scalar scale = Conj(tau)*work[i];

            // Update offset from lower-triangle of (j+i)'th col such that
            // diag[offset] = A(j,j+i+1)
            offset += (j+i+1) + j;

            // A(j,j+i+1) -= conj(tau) v(0) z[k] = conj(tau) z[k]
            diag[offset] -= scale;

            // Move to the lower-triangle of the (j+i+1)'th column of A
            offset += i+1;

            // Traverse over the relevant piece of this col of the 
            // lower-triangle
            for( int k=0; k<j+1; ++k )
                diag[offset+k] -= scale*diag[k+1];
        }

        // Move the pointer to the next diagonal value
        diag += 2*(j+1)+1;
    }
}

template void psp::hmat_tools::PackedQR
( int r, 
  float* RESTRICT A, 
  float* RESTRICT work );
template void psp::hmat_tools::PackedQR
( int r, 
  double* RESTRICT A, 
  double* RESTRICT work );
template void psp::hmat_tools::PackedQR
( int r, 
  std::complex<float>* RESTRICT A, 
  std::complex<float>* RESTRICT work );
template void psp::hmat_tools::PackedQR
( int r, 
  std::complex<double>* RESTRICT A, 
  std::complex<double>* RESTRICT work );