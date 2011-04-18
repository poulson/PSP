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

// B :~= alpha A + beta B
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  Real alpha, const LowRankMatrix<Real,Conjugated>& A, 
  Real beta,        LowRankMatrix<Real,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixUpdateRounded (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Incompatible matrix dimensions");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int minDim = std::min(m,n);
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( std::min(r,minDim), maxRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        Scale( beta, B.U );
        B.U.Resize( m, r );
        for( int j=0; j<Ar; ++j )
        {
            Real* RESTRICT BUACol = B.U.Buffer(0,j+Br);
            const Real* RESTRICT AUCol = A.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                BUACol[i] = alpha*AUCol[i];
        }

        // Copy A.V into the right half of B.V
        B.V.Resize( n, r );
        for( int j=0; j<Ar; ++j )
        {
            std::memcpy
            ( B.V.Buffer(0,j+Br), A.V.LockedBuffer(0,j), n*sizeof(Real) );
        }

#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Form U := [(alpha A.U), (beta B.U)]
    DenseMatrix<Real> U( m, r );
    for( int j=0; j<Ar; ++j )
    {
        Real* RESTRICT packedAUCol = U.Buffer(0,j);
        const Real* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedAUCol[i] = alpha*AUCol[i];
    }
    for( int j=0; j<Br; ++j )
    {
        Real* RESTRICT packedBUCol = U.Buffer(0,j+Ar);
        const Real* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedBUCol[i] = beta*BUCol[i];
    }

    // Form V := [A.V B.V]
    DenseMatrix<Real> V( n, r );
    for( int j=0; j<Ar; ++j )
        std::memcpy( V.Buffer(0,j), A.V.LockedBuffer(0,j), n*sizeof(Real) );
    for( int j=0; j<Br; ++j )
        std::memcpy( V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n*sizeof(Real) );

#if defined(PIVOTED_QR)
    // TODO 
    throw std::logic_error("Pivoted QR is not yet supported.");
#else
    // Perform an unpivoted QR decomposition on U
    const int minDimU = std::min(m,r);
    std::vector<Real> tauU( minDimU );
    std::vector<Real> workU( std::max(1,m*r) );
    lapack::QR( m, r, U.Buffer(), U.LDim(), &tauU[0], &workU[0], workU.size() );

    // Perform an unpivoted QR decomposition on V
    const int minDimV = std::min(n,r);
    std::vector<Real> tauV( minDimV );
    std::vector<Real> workV( std::max(1,n*r) );
    lapack::QR( n, r, V.Buffer(), V.LDim(), &tauV[0], &workV[0], workV.size() );

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    workU.resize( r*r );
    std::memset( &workU[0], 0, r*r*sizeof(Real) );
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &workU[j*r], U.LockedBuffer(0,j), std::min(m,j+1)*sizeof(Real) );
    }

    // Copy R2 (the right factor's R from QR) into a zeroed buffer
    workV.resize( r*r );
    std::memset( &workV[0], 0, r*r*sizeof(Real) );
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &workV[j*r], V.LockedBuffer(0,j), std::min(n,j+1)*sizeof(Real) );
    }

    // Form W := R1 R2^T.
    DenseMatrix<Real> W( minDimU, minDimV );
    blas::Gemm
    ( 'N', 'T', minDimU, minDimV, r,
      (Real)1, &workU[0], r, &workV[0], r, (Real)0, W.Buffer(), W.LDim() );

    // Get the SVD of R1 R2^T, overwriting R1 R2^T with UNew
    std::vector<Real> s( std::min(minDimU,minDimV) );
    DenseMatrix<Real> VT( std::min(minDimU,minDimV), minDimV );
    const int lworkSVD = lapack::SVDWorkSize( minDimU, minDimV );
    std::vector<Real> workSVD( lworkSVD );
    lapack::SVD
    ( 'O', 'S', minDimU, minDimV, W.Buffer(), W.LDim(), 
      &s[0], 0, 1, VT.Buffer(), VT.LDim(), &workSVD[0], lworkSVD );

    // Get B ready for the rounded low-rank matrices
    B.U.Resize( m, roundedRank );
    B.V.Resize( n, roundedRank );

    // Form the rounded B.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Real)0, B.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = W.LockedBuffer(0,j);
        Real* RESTRICT UColScaled = B.U.Buffer(0,j);
        for( int i=0; i<minDimU; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1
    workU.resize( std::max(1,m*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, minDimU, U.LockedBuffer(), U.LDim(), &tauU[0], 
      B.U.Buffer(), B.U.LDim(), &workU[0], workU.size() );

    // Form the rounded B.V by first filling it with 
    //  | (VT_Top)^T |, and then hitting it from the left with Q2
    //  |      0     |
    Scale( (Real)0, B.V );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = VT.LockedBuffer(j,0);
        const int VTLDim = VT.LDim();
        Real* RESTRICT VCol = B.V.Buffer(0,j);
        for( int i=0; i<minDimV; ++i )
            VCol[i] = VTRow[i*VTLDim];
    }
    // Apply Q2 
    workV.resize( std::max(1,n*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, minDimV, V.LockedBuffer(), V.LDim(), &tauV[0],
      B.V.Buffer(), B.V.LDim(), &workV[0], workV.size() );
#endif // PIVOTED_QR
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, 
  const LowRankMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta,        
        LowRankMatrix<std::complex<Real>,Conjugated>& B )
{
    typedef std::complex<Real> Scalar;
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixUpdateRounded (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Incompatible matrix dimensions");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int minDim = std::min(m,n);
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( std::min(r,minDim), maxRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        Scale( beta, B.U );
        B.U.Resize( m, r );
        for( int j=0; j<Ar; ++j )
        {
            Scalar* RESTRICT BUACol = B.U.Buffer(0,j+Br);
            const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                BUACol[i] = alpha*AUCol[i];
        }

        // Copy A.V into the right half of B.V
        B.V.Resize( n, r );
        for( int j=0; j<Ar; ++j )
        {
            std::memcpy
            ( B.V.Buffer(0,j+Br), A.V.LockedBuffer(0,j), n*sizeof(Scalar) );
        }

#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Form U := [(alpha A.U) (beta B.U)]
    DenseMatrix<Scalar> U( m, r );
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT packedAUCol = U.Buffer(0,j);
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedAUCol[i] = alpha*AUCol[i];
    }
    for( int j=0; j<Br; ++j )
    {
        Scalar* RESTRICT packedBUCol = U.Buffer(0,j+Ar);
        const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedBUCol[i] = beta*BUCol[i];
    }

    // Form V := [A.V B.V]
    DenseMatrix<Scalar> V( n, r );
    for( int j=0; j<Ar; ++j )
    {
        std::memcpy
        ( V.Buffer(0,j), A.V.LockedBuffer(0,j), n*sizeof(Scalar) );
    }
    for( int j=0; j<Br; ++j )
    {
        std::memcpy
        ( V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n*sizeof(Scalar) );
    }

#if defined(PIVOTED_QR)
    // TODO 
    throw std::logic_error("Pivoted QR is not yet supported.");
#else
    // Perform an unpivoted QR decomposition on U
    const int minDimU = std::min(m,r);
    std::vector<Scalar> tauU( minDimU );
    std::vector<Scalar> workU( std::max(1,m*r) );
    lapack::QR( m, r, U.Buffer(), U.LDim(), &tauU[0], &workU[0], workU.size() );

    // Perform an unpivoted QR decomposition on V
    const int minDimV = std::min(n,r);
    std::vector<Scalar> tauV( minDimV );
    std::vector<Scalar> workV( std::max(1,n*r) );
    lapack::QR( n, r, V.Buffer(), V.LDim(), &tauV[0], &workV[0], workV.size() );

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    workU.resize( r*r );
    std::memset( &workU[0], 0, r*r*sizeof(Scalar) );
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &workU[j*r], U.LockedBuffer(0,j), std::min(m,j+1)*sizeof(Scalar) );
    }

    // Copy R2 (the right factor's R from QR) into a zeroed buffer
    workV.resize( r*r );
    std::memset( &workV[0], 0, r*r*sizeof(Scalar) );
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &workV[j*r], V.LockedBuffer(0,j), std::min(n,j+1)*sizeof(Scalar) );
    }

    // Form W := R1 R2^[T,H]
    const char option = ( Conjugated ? 'C' : 'T' );
    DenseMatrix<Scalar> W( minDimU, minDimV );
    blas::Gemm
    ( 'N', option, minDimU, minDimV, r,
      (Scalar)1, &workU[0], r, &workV[0], r, (Scalar)0, W.Buffer(), W.LDim() );

    // Get the SVD of R1 R2^[T,H], overwriting R1 R2^[T,H] with UNew
    std::vector<Real> s( std::min(minDimU,minDimV) );
    DenseMatrix<Scalar> VH( std::min(minDimU,minDimV), minDimV );
    const int lworkSVD = lapack::SVDWorkSize( minDimU, minDimV );
    std::vector<Scalar> workSVD( lworkSVD );
    std::vector<Real> realWorkSVD( 5*r );
    lapack::SVD
    ( 'O', 'S', minDimU, minDimV, W.Buffer(), W.LDim(),
      &s[0], 0, 1, VH.Buffer(), VH.LDim(), &workSVD[0], lworkSVD, 
      &realWorkSVD[0] );

    // Get B ready for the rounded low-rank matrices
    B.U.Resize( m, roundedRank );
    B.V.Resize( n, roundedRank );

    // Form the rounded B.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Scalar)0, B.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Scalar* RESTRICT UCol = W.Buffer(0,j);
        Scalar* RESTRICT UColScaled = B.U.Buffer(0,j);
        for( int i=0; i<minDimU; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1
    workU.resize( std::max(1,m*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, minDimU, U.LockedBuffer(), U.LDim(), &tauU[0], 
      B.U.Buffer(), B.U.LDim(), &workU[0], workU.size() );

    // Form the rounded B.V by first filling it with 
    //  | (VH_Top)^[T,H] |, and then hitting it from the left with Q2
    //  |      0         |
    Scale( (Scalar)0, B.V );
    if( Conjugated )
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
            const int VHLDim = VH.LDim();
            Scalar* RESTRICT VCol = B.V.Buffer(0,j);
            for( int i=0; i<minDimV; ++i )
                VCol[i] = Conj( VHRow[i*VHLDim] );
        }
    }
    else
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
            const int VHLDim = VH.LDim();
            Scalar* RESTRICT VColConj = B.V.Buffer(0,j);
            for( int i=0; i<minDimV; ++i )
                VColConj[i] = VHRow[i*VHLDim];
        }
    }
    // Apply Q2
    workV.resize( std::max(1,n*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, minDimV, V.LockedBuffer(), V.LDim(), &tauV[0],
      B.V.Buffer(), B.V.LDim(), &workV[0], workV.size() );
#endif // PIVOTED_QR
#ifndef RELEASE
    PopCallStack();
#endif
}

template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  float alpha, const LowRankMatrix<float,false>& A,
  float beta,        LowRankMatrix<float,false>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  float alpha, const LowRankMatrix<float,true>& A,
  float beta,        LowRankMatrix<float,true>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  double alpha, const LowRankMatrix<double,false>& A,
  double beta,        LowRankMatrix<double,false>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  double alpha, const LowRankMatrix<double,true>& A,
  double beta,        LowRankMatrix<double,true>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta, 
        LowRankMatrix<std::complex<float>,false>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta, 
        LowRankMatrix<std::complex<float>,true>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta, 
        LowRankMatrix<std::complex<double>,false>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta, 
        LowRankMatrix<std::complex<double>,true>& B );
