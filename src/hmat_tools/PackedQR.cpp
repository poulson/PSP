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

// Perform a QR factorization of size (s+t) x r where only the upper triangles
// of the s x r and t x r submatrices are nonzero, and the nonzeros are packed 
// columnwise.
//
// The work buffer must be of size t-1.
template<typename Scalar>
void psp::hmat_tools::PackedQR
( const int r, const int s, const int t,
  Scalar* RESTRICT packedA, Scalar* RESTRICT tau, Scalar* RESTRICT work )
{
    const int minDim = std::min(s+t,r);

    int jCol = 0;
    for( int j=0; j<minDim; ++j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );

        // Compute the Householder vector, v, and scalar, tau, in-place
        const int jDiag = jCol + j;
        tau[j] = Householder( S+T-j, &packedA[jDiag] );

        // Form z := A(I_j,j+1:end)' v in the work vector 
        int iCol = jCol + S + T;
        for( int i=0; i<r-(j+1); ++i )
        {
            const int Si = std::min(j+i+2,s);
            const int Ti = std::min(j+i+2,t);

            // z[i] := Conj(A(j,j+i+1)) v(0) = Conj(A(j,j+i+1))
            const int iDiagRight = iCol + j;
            work[i] = Conj(packedA[iDiagRight]);

            // Traverse over this col of the lower triangle
            const int jump = ( j >= s ? 1 : Si-j );
            for( int k=0; k<T-overlap; ++k )
                work[i] += Conj(packedA[iDiagRight+k+jump])*packedA[jDiag+k+1];

            iCol += Si + Ti;
        }

        // A(I_j,j+1:end) -= conj(tau) v z'
        iCol = jCol + S + T;
        for( int i=0; i<r-(j+1); ++i )
        {
            const int Si = std::min(j+i+2,s);
            const int Ti = std::min(j+i+2,t);

            const Scalar scale = Conj(tau[j])*Conj(work[i]);

            // A(j,j+i+1) -= conj(tau) v(0) z[k] = conj(tau) z[k]
            const int iDiagRight = iCol + j;
            packedA[iDiagRight] -= scale;

            // Traverse over the relevant piece of this col of the 
            // lower-triangle
            const int jump = ( j >= s ? 1 : Si-j );
            for( int k=0; k<T-overlap; ++k )
                packedA[iDiagRight+k+jump] -= scale*packedA[jDiag+k+1];

            iCol += Si + Ti;
        }

        jCol += S + T;
    }
}

template<typename Scalar>
void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau, 
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::ApplyPackedQFromLeft");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Height() != s+t )
        throw std::logic_error("B is not the correct height");
#endif
    Scalar* BBuffer = B.Buffer();
    const int n = B.Width();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    const int packedATopSize = (s*s+s)/2 + (r-s)*s;
    const int packedABottomSize = (t*t+t)/2 + (r-t)*t;
    const int packedASize = packedATopSize + packedABottomSize;

    int jCol = packedASize;
    for( int j=minDim-1; j>=0; --j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );
        jCol -= S + T;

        // B := (I - tau_j v_j v_j') B
        //    = B - tau_j v_j (v_j' B)
        //    = B - tau_j v_j (B' v_j)'

        // 1) Form w_j := B' v_j
        // Since v_j's only nonzero entries are a 1 in the j'th entry and 
        // arbitrary values in the r:r+j entries, 
        //     w_j = B(j,:)' + B(s:s+T-1,:)' v_j(s:s+T-1)
        for( int i=0; i<n; ++i )
            work[i] = Conj(BBuffer[j+i*BLDim]);
        blas::Gemv
        ( 'C', T-overlap, n, 
          (Scalar)1, &BBuffer[s+overlap],      BLDim,
                     &packedA[jCol+S+overlap], 1,
          (Scalar)1, work,                     1 );

        // 2) B := B - tau_j v_j w_j'
        // Since v_j has the structure described above, we only need to 
        // subtract tau_j w_j' from the j'th row of B and then perform the
        // update 
        //     B(s:s+T-1,:) -= tau_j v_j(s:s+T-1) w_j'
        const Scalar tauj = tau[j];
        for( int i=0; i<n; ++i )
            BBuffer[j+i*BLDim] -= tauj*Conj(work[i]);
        blas::Ger
        ( T-overlap, n, 
          -tauj, &packedA[jCol+S+overlap], 1,
                 work,                     1,
                 &BBuffer[s+overlap],      BLDim );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau, 
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::ApplyPackedQAdjointFromLeft");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Height() != s+t )
        throw std::logic_error("B is not the correct height");
#endif
    Scalar* BBuffer = B.Buffer();
    const int n = B.Width();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    int jCol = 0;
    for( int j=0; j<minDim; ++j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );

        // B := (I - conj(tau_j) v_j v_j') B
        //    = B - conj(tau_j) v_j v_j' B
        //    = B - conj(tau_j) v_j (B' v_j)'

        // 1) Form w_j := B' v_j
        // Since v_j's only nonzero entries are a 1 in the j'th entry and 
        // arbitrary values in the r:r+j entries, 
        //     w_j = B(j,:)' + B(s:s+T-1,:)' v_j(s:s+T-1)
        for( int i=0; i<n; ++i )
            work[i] = Conj(BBuffer[j+i*BLDim]);
        blas::Gemv
        ( 'C', T-overlap, n, 
          (Scalar)1, &BBuffer[s+overlap],      BLDim,
                     &packedA[jCol+S+overlap], 1,
          (Scalar)1, work,                     1 );

        // 2) B := B - tau_j v_j w_j'
        // Since v_j has the structure described above, we only need to 
        // subtract tau_j w_j' from the j'th row of B and then perform the
        // update 
        //     B(s:s+T-1,:) -= tau_j v_j(s:s+T-1) w_j'
        const Scalar conjTauj = Conj(tau[j]);
        for( int i=0; i<n; ++i )
            BBuffer[j+i*BLDim] -= conjTauj*Conj(work[i]);
        blas::Ger
        ( T-overlap, n, 
          -conjTauj, &packedA[jCol+S+overlap], 1,
                     work,                     1,
                     &BBuffer[s+overlap],      BLDim );

        jCol += S + T;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::ApplyPackedQFromRight
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau, 
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::ApplyPackedQFromRight");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Width() != s+t )
        throw std::logic_error("B is not the correct width");
#endif
    Scalar* BBuffer = B.Buffer();
    const int m = B.Height();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    int jCol = 0;
    for( int j=0; j<minDim; ++j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );

        // B := B (I - tau_j v_j v_j')
        //    = B - (tau_j B v_j) v_j'

        // 1) Form w_j := tau_j B v_j
        const Scalar tauj = tau[j];
        for( int i=0; i<m; ++i )
            work[i] = tauj*BBuffer[i+j*BLDim];
        blas::Gemv
        ( 'N', m, T-overlap,
          tauj,      &BBuffer[(s+overlap)*BLDim], BLDim,
                     &packedA[jCol+S+overlap],    1,
          (Scalar)1, work,                        1 );

        // 2) B := B - w_j v_j'
        for( int i=0; i<m; ++i )
            BBuffer[i+j*BLDim] -= work[i];
        blas::Ger
        ( m, T-overlap,
         (Scalar)-1, work,                        1,
                     &packedA[jCol+S+overlap],    1,
                     &BBuffer[(s+overlap)*BLDim], BLDim );

        jCol += S + T;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau, 
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::ApplyPackedQAdjointFromRight");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Width() != s+t )
        throw std::logic_error("B is not the correct width");
#endif
    Scalar* BBuffer = B.Buffer();
    const int m = B.Height();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    const int packedATopSize = (s*s+s)/2 + (r-s)*s;
    const int packedABottomSize = (t*t+t)/2 + (r-t)*t;
    const int packedASize = packedATopSize + packedABottomSize;

    int jCol = packedASize;
    for( int j=minDim-1; j>=0; --j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );
        jCol -= S + T;

        // B := B (I - conj(tau)_j v_j v_j')
        //    = B - (conj(tau_j) B v_j) v_j'

        // 1) Form w_j := conj(tau_j) B v_j
        const Scalar conjTauj = Conj(tau[j]);
        for( int i=0; i<m; ++i )
            work[i] = conjTauj*BBuffer[i+j*BLDim];
        blas::Gemv
        ( 'N', m, T-overlap,
          conjTauj,  &BBuffer[(s+overlap)*BLDim], BLDim,
                     &packedA[jCol+S+overlap],    1,
          (Scalar)1, work,              1 );

        // 2) B := B - w_j v_j'
        for( int i=0; i<m; ++i )
            BBuffer[i+j*BLDim] -= work[i];
        blas::Ger
        ( m, T-overlap,
         (Scalar)-1, work,                        1,
                     &packedA[jCol+S+overlap],    1,
                     &BBuffer[(s+overlap)*BLDim], BLDim );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::PrintPacked
( std::ostream& os, const std::string& msg, 
  const int r, const int s, const int t,
  const Scalar* packedA )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::PrintPacked");
#endif
    os << msg << "\n";

    // Print the upper triangle
    int iCol = 0;
    for( int i=0; i<s; ++i )
    {
        const int Si = std::min(i+1,s);
        const int Ti = std::min(i+1,t); 

        for( int j=0; j<i; ++j )
            os << "0 ";

        int jCol = iCol;
        for( int j=i; j<r; ++j )
        {
            const int Sj = std::min(j+1,s);
            const int Tj = std::min(j+1,t);

            const Scalar value = packedA[jCol+i];
            os << ScalarWrapper<Scalar>(value) << " ";

            jCol += Sj + Tj;
        }
        os << "\n";

        iCol += Si + Ti;
    }

    // Print the lower triangle
    iCol = 0;
    for( int i=0; i<t; ++i )
    {
        const int Si = std::min(i+1,s);
        const int Ti = std::min(i+1,t); 

        for( int j=0; j<i; ++j )
            os << "0 ";

        int jCol = iCol;
        for( int j=i; j<r; ++j )
        {
            const int Sj = std::min(j+1,s);
            const int Tj = std::min(j+1,t);

            const Scalar value = packedA[jCol+Sj+i];
            os << ScalarWrapper<Scalar>(value) << " ";

            jCol += Sj + Tj;
        }
        os << "\n";

        iCol += Si + Ti;
    }

    os.flush();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void psp::hmat_tools::PrintPacked
( const std::string& msg, 
  const int r, const int s, const int t,
  const Scalar* packedA )
{
    PrintPacked( std::cout, msg, r, s, t, packedA );
}

template void psp::hmat_tools::PackedQR
( const int r, const int s, const int t,
  float* RESTRICT A, 
  float* RESTRICT tau,
  float* RESTRICT work );
template void psp::hmat_tools::PackedQR
( const int r, const int s, const int t,
  double* RESTRICT A, 
  double* RESTRICT tau,
  double* RESTRICT work );
template void psp::hmat_tools::PackedQR
( const int r, const int s, const int t,
  std::complex<float>* RESTRICT A, 
  std::complex<float>* RESTRICT tau,
  std::complex<float>* RESTRICT work );
template void psp::hmat_tools::PackedQR
( const int r, const int s, const int t,
  std::complex<double>* RESTRICT A, 
  std::complex<double>* RESTRICT tau,
  std::complex<double>* RESTRICT work );

template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const float* RESTRICT A, 
  const float* RESTRICT tau,
        Dense<float>& B, 
        float* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const double* RESTRICT A, 
  const double* RESTRICT tau,
        Dense<double>& B, 
        double* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A, 
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A, 
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const float* RESTRICT A, 
  const float* RESTRICT tau,
        Dense<float>& B, 
        float* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const double* RESTRICT A, 
  const double* RESTRICT tau,
        Dense<double>& B, 
        double* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A, 
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A, 
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void psp::hmat_tools::ApplyPackedQFromRight
( const int r, const int s, const int t,
  const float* RESTRICT A, 
  const float* RESTRICT tau,
        Dense<float>& B, 
        float* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromRight
( const int r, const int s, const int t,
  const double* RESTRICT A, 
  const double* RESTRICT tau,
        Dense<double>& B, 
        double* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromRight
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A, 
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQFromRight
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A, 
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void psp::hmat_tools::ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const float* RESTRICT A, 
  const float* RESTRICT tau,
        Dense<float>& B, 
        float* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const double* RESTRICT A, 
  const double* RESTRICT tau,
        Dense<double>& B, 
        double* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A, 
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void psp::hmat_tools::ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A, 
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void psp::hmat_tools::PrintPacked
( std::ostream& os, const std::string& msg,
  const int r, const int s, const int t, const float* packedA );
template void psp::hmat_tools::PrintPacked
( std::ostream& os, const std::string& msg,
  const int r, const int s, const int t, const double* packedA );
template void psp::hmat_tools::PrintPacked
( std::ostream& os, const std::string& msg,
  const int r, const int s, const int t, const std::complex<float>* packedA );
template void psp::hmat_tools::PrintPacked
( std::ostream& os, const std::string& msg,
  const int r, const int s, const int t, const std::complex<double>* packedA );

template void psp::hmat_tools::PrintPacked
( const std::string& msg,
  const int r, const int s, const int t, const float* packedA );
template void psp::hmat_tools::PrintPacked
( const std::string& msg,
  const int r, const int s, const int t, const double* packedA );
template void psp::hmat_tools::PrintPacked
( const std::string& msg,
  const int r, const int s, const int t, const std::complex<float>* packedA );
template void psp::hmat_tools::PrintPacked
( const std::string& msg,
  const int r, const int s, const int t, const std::complex<double>* packedA );

