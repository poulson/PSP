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
template<typename Real,bool Conjugate>
void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real,Conjugate>& A, 
  Real beta,        FactorMatrix<Real,Conjugate>& B )
{
    const int m = A.m;
    const int n = A.n;
    const int r = A.r + B.r;
    const int roundedRank = std::min( r, maxRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        B.U.resize( m*r );
        B.V.resize( n*r );

        // Scale B.U by beta
        {
            const int Br = B.r;
            Real* RESTRICT BULeft = &B.U[0];
            for( int j=0; j<Br; ++j )
                for( int i=0; i<m; ++i )
                    BULeft[i+j*m] *= beta;
        }
        // Copy alpha A.U into the right half of B.U
        {
            const int Ar = A.r;
            Real* RESTRICT BURight = &B.U[m*B.r];
            const Real* RESTRICT AU = &A.U[0];
            for( int j=0; j<Ar; ++j )
                for( int i=0; i<m; ++i )
                    BURight[i+j*m] = alpha*AU[i+j*m];
        }

        // Copy A.V into the right half of B.V
        std::memcpy( &B.V[n*B.r], &A.V[0], n*A.r*sizeof(Real) );

        // Mark the new rank
        B.r = roundedRank;

        return;
    }

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = 4*r*r;
    std::vector<Real> buffer
    ( leftPanelSize+rightPanelSize+
      std::max(std::max(leftPanelSize,rightPanelSize),2*blockSize+lworkSVD) );

    // Put [(alpha A.U), (beta B.U)] into the first 'leftPanelSize' entries of
    // our buffer and [A.V, B.V] into the next 'rightPanelSize' entries
    // Copy in (alpha A.U)
    {
        const int Ar = A.r;
        Real* RESTRICT packedAU = &buffer[0];
        const Real* RESTRICT AU = &A.U[0];
        for( int j=0; j<Ar; ++j )
            for( int i=0; i<m; ++i )
                packedAU[i+j*m] = alpha*AU[i+j*m];
    }
    // Copy in (beta B.U)
    {
        const int Br = B.r;
        Real* RESTRICT packedBU = &buffer[m*A.r];
        const Real* RESTRICT BU = &B.U[0];
        for( int j=0; j<Br; ++j )
            for( int i=0; i<m; ++i )
                packedBU[i+j*m] = beta*BU[i+j*m];
    }
    // Copy in A.V
    std::memcpy
    ( &buffer[leftPanelSize], &A.V[0], n*A.r*sizeof(Real) );
    // Copy in B.V
    std::memcpy
    ( &buffer[leftPanelSize+n*A.r], &B.V[0], n*B.r*sizeof(Real) );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize): [(alpha A.U), (beta B.U)]                          //
    //  [leftPanelSize,leftPanelSize+rightPanelSize): [A.V, B.V]              //
    //------------------------------------------------------------------------//
    const int offset = leftPanelSize + rightPanelSize;

#if defined(PIVOTED_QR)
    // TODO 
    throw std::logic_error("Pivoted QR is not yet supported.");
#else
    // Perform an unpivoted QR decomposition on [(alpha A.U), (beta B.U)]
    std::vector<Real> tauU( std::min( m, r ) );
    lapack::QR( m, r, &buffer[0], m, &tauU[0], &buffer[offset], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):      qr([(alpha A.U), (beta B.U)])                 //
    //  [leftPanelSize,offset): [A.V, B.V]                                    //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on [A.V, B.V]
    std::vector<Real> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, &buffer[leftPanelSize], n, &tauV[0], 
      &buffer[offset], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):      qr([(alpha A.U), (beta B.U)])                 //
    //  [leftPanelSize,offset): qr([A.V, B.V])                                //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        Real* RESTRICT W = &buffer[offset];
        const Real* RESTRICT R1 = &buffer[0];
        std::memset( W, 0, blockSize*sizeof(Real) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                W[i+j*r] = R1[i+j*m];
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):         qr([(alpha A.U), (beta B.U)])              //
    //  [leftPanelSize,offset):    qr([A.V, B.V])                             //
    //  [offset,offset+blockSize): R1                                         //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^T. We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm
    ( 'R', 'U', 'T', 'N', r, r, 
      1, &buffer[leftPanelSize], n, &buffer[offset], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):         qr([(alpha A.U), (beta B.U)])              //
    //  [leftPanelSize,offset):    qr([A.V, B.V])                             //
    //  [offset,offset+blockSize): R1 R2^T                                    //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^T, overwriting R1 R2^T with U
    std::vector<Real> s( r );
    lapack::SVD
    ( 'O', 'A', r, r, &buffer[offset], r, 
      &s[0], 0, 0, &buffer[offset+blockSize], r, 
      &buffer[offset+2*blockSize], lworkSVD );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):                     qr([(alpha A.U),(beta B.U)])   //
    //  [leftPanelSize,offset):                qr([A.V, B.V])                 //
    //  [offset,offset+blockSize):             U of R1 R2^T                   //
    //  [offset+blockSize,offset+2*blockSize): V^T of R1 R2^T                 //
    //------------------------------------------------------------------------//

    // Get B ready for the rounded factors
    B.r = roundedRank;
    B.U.resize( m*roundedRank );
    B.V.resize( n*roundedRank );

    // Form the rounded B.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    std::memset( &B.U[0], 0, m*roundedRank*sizeof(Real) );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = &buffer[offset+j*r];
        Real* RESTRICT UColScaled = &B.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'T', m, roundedRank, r, &buffer[0], B.m, &tauU[0], &B.U[0], B.m, 
      &buffer[offset], blockSize );

    // Form the rounded B.V by first filling it with 
    //  | (VT_Top)^T |, and then hitting it from the left with Q2
    //  |      0     |
    std::memset( &B.V[0], 0, n*roundedRank*sizeof(Real) );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = &buffer[offset+blockSize+j];
        Real* RESTRICT VCol = &B.V[j*n];
        for( int i=0; i<r; ++i )
            VCol[i] = VTRow[i*r];
    }
    // Apply Q2 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'T', n, roundedRank, r, &buffer[leftPanelSize], B.n, &tauV[0], 
      &B.V[0], B.n, &buffer[offset], blockSize );
#endif // PIVOTED_QR
}

template<typename Real,bool Conjugate>
void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix<std::complex<Real>,Conjugate>& A,
  std::complex<Real> beta,        FactorMatrix<std::complex<Real>,Conjugate>& B 
)
{
    typedef std::complex<Real> Scalar;

    const int m = A.m;
    const int n = A.n;
    const int r = A.r + B.r;
    const int roundedRank = std::min( r, maxRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        B.U.resize( m*r );
        B.V.resize( n*r );

        // Scale B.U by beta
        {
            const int Br = B.r;
            Scalar* RESTRICT BULeft = &B.U[0];
            for( int j=0; j<Br; ++j )
                for( int i=0; i<m; ++i )
                    BULeft[i+j*m] *= beta;
        }
        // Copy alpha A.U into the right half of B.U
        {
            const int Ar = A.r;
            Scalar* RESTRICT BURight = &B.U[m*B.r];
            const Scalar* RESTRICT AU = &A.U[0];
            for( int j=0; j<Ar; ++j )
                for( int i=0; i<m; ++i )
                    BURight[i+j*m] = alpha*AU[i+j*m];
        }

        // Copy A.V into the right half of B.V
        std::memcpy( &B.V[n*B.r], &A.V[0], n*A.r*sizeof(Scalar) );

        // Mark the new rank
        B.r = roundedRank;

        return;
    }

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = 4*r*r;
    std::vector<Scalar> buffer
    ( leftPanelSize+rightPanelSize+
      std::max(std::max(leftPanelSize,rightPanelSize),2*blockSize+lworkSVD) );

    // Put [(alpha A.U), (beta B.U)] into the first 'leftPanelSize' entries of
    // our buffer and [A.V, B.V] into the next 'rightPanelSize' entries
    // Copy in (alpha A.U)
    {
        const int Ar = A.r;
        Scalar* RESTRICT packedAU = &buffer[0];
        const Scalar* RESTRICT AU = &A.U[0];
        for( int j=0; j<Ar; ++j )
            for( int i=0; i<m; ++i )
                packedAU[i+j*m] = alpha*AU[i+j*m];
    }
    // Copy in (beta B.U)
    {
        const int Br = B.r;
        Scalar* RESTRICT packedBU = &buffer[m*A.r];
        const Scalar* RESTRICT BU = &B.U[0];
        for( int j=0; j<Br; ++j )
            for( int i=0; i<m; ++i )
                packedBU[i+j*m] = beta*BU[i+j*m];
    }
    // Copy in A.V
    std::memcpy
    ( &buffer[leftPanelSize], &A.V[0], n*A.r*sizeof(Scalar) );
    // Copy in B.V
    std::memcpy
    ( &buffer[leftPanelSize+n*A.r], &B.V[0], n*B.r*sizeof(Scalar) );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize): [(alpha A.U), (beta B.U)]                          //
    //  [leftPanelSize,leftPanelSize+rightPanelSize): [A.V, B.V]              //
    //------------------------------------------------------------------------//
    const int offset = leftPanelSize + rightPanelSize;

#if defined(PIVOTED_QR)
    // TODO 
    throw std::logic_error("Pivoted QR is not yet supported.");
#else
    // Perform an unpivoted QR decomposition on [(alpha A.U), (beta B.U)]
    std::vector<Scalar> tauU( std::min( m, r ) );
    lapack::QR( m, r, &buffer[0], m, &tauU[0], &buffer[offset], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):      qr([(alpha A.U), (beta B.U)])                 //
    //  [leftPanelSize,offset): [A.V, B.V]                                    //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on [A.V, B.V]
    std::vector<Scalar> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, &buffer[leftPanelSize], n, &tauV[0], 
      &buffer[offset], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):      qr([(alpha A.U), (beta B.U)])                 //
    //  [leftPanelSize,offset): qr([A.V, B.V])                                //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        Scalar* RESTRICT W = &buffer[offset];
        const Scalar* RESTRICT R1 = &buffer[0];
        std::memset( W, 0, blockSize*sizeof(Scalar) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                W[i+j*r] = R1[i+j*m];
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):         qr([(alpha A.U), (beta B.U)])              //
    //  [leftPanelSize,offset):    qr([A.V, B.V])                             //
    //  [offset,offset+blockSize): R1                                         //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^[T,H]. We are unfortunately performing 2x as many
    // flops as are required.
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Trmm
    ( 'R', 'U', option, 'N', r, r, 
      1, &buffer[leftPanelSize], n, &buffer[offset], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):         qr([(alpha A.U), (beta B.U)])              //
    //  [leftPanelSize,offset):    qr([A.V, B.V])                             //
    //  [offset,offset+blockSize): R1 R2^[T,H]                                //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^[T,H], overwriting R1 R2^[T,H] with U
    std::vector<Real> realBuffer( 6*r );
    lapack::SVD
    ( 'O', 'A', r, r, &buffer[offset], r, 
      &realBuffer[0], 0, 0, &buffer[offset+blockSize], r, 
      &buffer[offset+2*blockSize], lworkSVD, &realBuffer[r] );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):                     qr([(alpha A.U),(beta B.U)])   //
    //  [leftPanelSize,offset):                qr([A.V, B.V])                 //
    //  [offset,offset+blockSize):             U of R1 R2^[T,H]               //
    //  [offset+blockSize,offset+2*blockSize): V^H of R1 R2^[T,H]             //
    //                                                                        //
    // realBuffer contains:                                                   //
    //   [0,r): singular values of R1 R2^[T,H]                                //
    //------------------------------------------------------------------------//

    // Get B ready for the rounded factors
    B.r = roundedRank;
    B.U.resize( m*roundedRank );
    B.V.resize( n*roundedRank );

    // Form the rounded B.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    std::memset( &B.U[0], 0, m*roundedRank*sizeof(Scalar) );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        const Scalar* RESTRICT UCol = &buffer[offset+j*r];
        Scalar* RESTRICT UColScaled = &B.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'C', m, roundedRank, r, &buffer[0], B.m, &tauU[0], &B.U[0], B.m, 
      &buffer[offset], blockSize );

    // Form the rounded B.V by first filling it with 
    //  | (VH_Top)^[T,H] |, and then hitting it from the left with Q2
    //  |      0         |
    std::memset( &B.V[0], 0, n*roundedRank*sizeof(Scalar) );
    if( Conjugate )
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[offset+blockSize+j];
            Scalar* RESTRICT VCol = &B.V[j*n];
            for( int i=0; i<r; ++i )
                VCol[i] = Conj( VHRow[i*r] );
        }
    }
    else
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[offset+blockSize+j];
            Scalar* RESTRICT VColConj = &B.V[j*n];
            for( int i=0; i<r; ++i )
                VColConj[i] = VHRow[i*r];
        }
    }
    // Apply Q2 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'C', n, roundedRank, r, &buffer[leftPanelSize], B.n, &tauV[0], 
      &B.V[0], B.n, &buffer[offset], blockSize );
#endif // PIVOTED_QR
}

template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  float alpha, const FactorMatrix<float,false>& A,
  float beta,        FactorMatrix<float,false>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  float alpha, const FactorMatrix<float,true>& A,
  float beta,        FactorMatrix<float,true>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  double alpha, const FactorMatrix<double,false>& A,
  double beta,        FactorMatrix<double,false>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  double alpha, const FactorMatrix<double,true>& A,
  double beta,        FactorMatrix<double,true>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,        FactorMatrix<std::complex<float>,false>& B );template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,        FactorMatrix<std::complex<float>,true>& B );
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,        FactorMatrix<std::complex<double>,false>& B 
);
template void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,        FactorMatrix<std::complex<double>,true>& B 
);