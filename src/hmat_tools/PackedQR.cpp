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

} // anonymous namespace

// Perform a QR factorization of size 2r x r where only the upper triangles
// of the matrix are stored, and the nonzeros are packed columnwise.
//
// The work buffer must be of size r-1.
//
// NOTE: The top of the j'th column is the (j^2+j)'th entry.
template<typename Scalar>
void psp::hmat_tools::PackedQR
( const int r, Scalar* RESTRICT packedA, Scalar* RESTRICT tau, 
  Scalar* RESTRICT work )
{
    for( int j=0; j<r; ++j )
    {
        // Compute the Householder vector, v, and scalar, tau, in-place
        const int diag = j*(j+2);
        tau[j] = Householder( j+2, &packedA[diag] );

        // Form z := A(I_j,j+1:end)' v in the work vector 
        for( int i=0; i<r-1-j; ++i )
        {
            // z[i] := Conj(A(j,j+i+1)) v(0) = Conj(A(j,j+1))
            const int right = (j+i+1)*(j+i+2) + j;
            work[i] = Conj(packedA[right]);

            // Traverse over this col of the lower triangle
            for( int k=0; k<j+1; ++k )
                work[i] += Conj(packedA[right+(i+2)+k])*packedA[diag+k+1];
        }

        // A(I_j,j+1:n-1) -= conj(tau) v z'
        for( int i=0; i<r-1-j; ++i )
        {
            const Scalar scale = Conj(tau[j])*Conj(work[i]);

            // A(j,j+i+1) -= conj(tau) v(0) z[k] = conj(tau) z[k]
            const int right = (j+i+1)*(j+i+2) + j;
            packedA[right] -= scale;

            // Traverse over the relevant piece of this col of the 
            // lower-triangle
            for( int k=0; k<j+1; ++k )
                packedA[right+(i+2)+k] -= scale*packedA[diag+k+1];
        }
    }
}

template<typename Scalar>
void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau, 
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::ApplyPackedQFromLeft");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix.");
#endif
    const int n = B.Width();
    Scalar* BBuffer = B.Buffer();
    const int BLDim = B.LDim();
    for( int j=r-1; j>=0; --j )
    {
        // B := B - tau_j v_j v_j' B

        // 1) Form w_j := B' v_j
        // Since v_j's only nonzero entries are a 1 in the j'th entry and 
        // arbitrary values in the r:r+j entries, 
        //     w_j = B(j,:)' + B(r:r+j,:)' v_j(r:r+j)
        for( int i=0; i<n; ++i )
            work[i] = Conj(BBuffer[j+i*BLDim]);
        blas::Gemv
        ( 'C', j+1, n, 
          (Scalar)1, &BBuffer[r],           BLDim,
                     &packedA[(j+1)*(j+1)], 1,
          (Scalar)1, work,                  1 );

        // 2) B := B - tau_j v_j w_j'
        // Since v_j has the structure described above, we only need to 
        // subtract tau_j w_j' from the j'th row of B and then perform the
        // update 
        //     B(r:r+j,:) -= tau_j v_j(r:r+j) w_j'
        const Scalar tauj = tau[j];
        for( int i=0; i<n; ++i )
            BBuffer[j+i*BLDim] -= tauj*Conj(work[i]);
        blas::Ger
        ( j+1, n, 
          -tauj, &packedA[(j*j+j)+(j+1)], 1,
                 work,                    1,
                 &BBuffer[r],             BLDim );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau, 
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::ApplyPackedQAdjointFromLeft");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix.");
#endif
    const int n = B.Width();
    Scalar* BBuffer = B.Buffer();
    const int BLDim = B.LDim();
    for( int j=0; j<r; ++j )
    {
        // B := B - conj(tau_j) v_j v_j' B

        // 1) Form w_j := B' v_j
        // Since v_j's only nonzero entries are a 1 in the j'th entry and 
        // arbitrary values in the r:r+j entries, 
        //     w_j = B(j,:)' + B(r:r+j,:)' v_j(r:r+j)
        for( int i=0; i<n; ++i )
            work[i] = Conj(BBuffer[j+i*BLDim]);
        blas::Gemv
        ( 'C', j+1, n, 
          (Scalar)1, &BBuffer[r],           BLDim,
                     &packedA[(j+1)*(j+1)], 1,
          (Scalar)1, work,                  1 );

        // 2) B := B - tau_j v_j w_j'
        // Since v_j has the structure described above, we only need to 
        // subtract tau_j w_j' from the j'th row of B and then perform the
        // update 
        //     B(r:r+j,:) -= tau_j v_j(r:r+j) w_j'
        const Scalar conjTauj = Conj(tau[j]);
        for( int i=0; i<n; ++i )
            BBuffer[j+i*BLDim] -= conjTauj*Conj(work[i]);
        blas::Ger
        ( j+1, n, 
          -conjTauj, &packedA[(j*j+j)+(j+1)], 1,
                     work,                    1,
                     &BBuffer[r],             BLDim );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::PrintPacked
( const int r, const Scalar* packedA, const std::string& msg )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::PrintPacked");
#endif
    std::cout << msg << "\n";
    // Print the upper triangle
    for( int i=0; i<r; ++i )
    {
        for( int j=0; j<i; ++j )
            std::cout << "0 ";
        for( int j=i; j<r; ++j )
            std::cout << packedA[(j*j+j)+i] << " ";
        std::cout << "\n";
    }
    // Print the lower triangle
    for( int i=0; i<r; ++i )
    {
        for( int j=0; j<i; ++j )
            std::cout << "0 ";
        for( int j=i; j<r; ++j )
            std::cout << packedA[(j*j+j)+(i+j+1)] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
#ifndef RELEASE
    PopCallStack();
#endif
}

template void psp::hmat_tools::PackedQR
( const int r, 
  float* RESTRICT A, 
  float* RESTRICT tau,
  float* RESTRICT work );
template void psp::hmat_tools::PackedQR
( const int r, 
  double* RESTRICT A, 
  double* RESTRICT tau,
  double* RESTRICT work );
template void psp::hmat_tools::PackedQR
( const int r, 
  std::complex<float>* RESTRICT A, 
  std::complex<float>* RESTRICT tau,
  std::complex<float>* RESTRICT work );
template void psp::hmat_tools::PackedQR
( const int r, 
  std::complex<double>* RESTRICT A, 
  std::complex<double>* RESTRICT tau,
  std::complex<double>* RESTRICT work );

template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, 
  const float* RESTRICT A, 
  const float* RESTRICT tau,
        Dense<float>& B, 
        float* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, 
  const double* RESTRICT A, 
  const double* RESTRICT tau,
        Dense<double>& B, 
        double* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, 
  const std::complex<float>* RESTRICT A, 
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, 
  const std::complex<double>* RESTRICT A, 
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, 
  const float* RESTRICT A, 
  const float* RESTRICT tau,
        Dense<float>& B, 
        float* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, 
  const double* RESTRICT A, 
  const double* RESTRICT tau,
        Dense<double>& B, 
        double* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, 
  const std::complex<float>* RESTRICT A, 
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, 
  const std::complex<double>* RESTRICT A, 
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void psp::hmat_tools::PrintPacked
( const int r, const float* packedA, const std::string& msg );
template void psp::hmat_tools::PrintPacked
( const int r, const double* packedA, const std::string& msg );
template void psp::hmat_tools::PrintPacked
( const int r, const std::complex<float>* packedA, const std::string& msg );
template void psp::hmat_tools::PrintPacked
( const int r, const std::complex<double>* packedA, const std::string& msg );
