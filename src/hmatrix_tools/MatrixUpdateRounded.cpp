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
  Real alpha, const FactorMatrix<Real,Conjugated>& A, 
  Real beta,        FactorMatrix<Real,Conjugated>& B )
{
    const int m = A.Height();
    const int n = A.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( r, maxRank );

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
    for( int j=0; j<Ar; ++j )
    {
        Real* RESTRICT packedAUCol = &buffer[j*m];
        const Real* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedAUCol[i] = alpha*AUCol[i];
    }
    // Copy in (beta B.U)
    for( int j=0; j<Br; ++j )
    {
        Real* RESTRICT packedBUCol = &buffer[(j+Ar)*m];
        const Real* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedBUCol[i] = beta*BUCol[i];
    }
    // Copy in A.V
    for( int j=0; j<Ar; ++j )
    {
        std::memcpy
        ( &buffer[leftPanelSize+j*n], A.V.LockedBuffer(0,j), 
          n*sizeof(Real) );
    }
    // Copy in B.V
    for( int j=0; j<Br; ++j )
    {
        std::memcpy
        ( &buffer[leftPanelSize+(j+Ar)*n], B.V.LockedBuffer(0,j), 
          n*sizeof(Real) );
    }

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
    B.U.Resize( m, roundedRank );
    B.V.Resize( n, roundedRank );

    // Form the rounded B.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Real)0, B.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = &buffer[offset+j*r];
        Real* RESTRICT UColScaled = B.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[0], B.Height(), &tauU[0], 
      B.U.Buffer(), B.U.LDim(), &buffer[offset], blockSize );

    // Form the rounded B.V by first filling it with 
    //  | (VT_Top)^T |, and then hitting it from the left with Q2
    //  |      0     |
    Scale( (Real)0, B.V );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = &buffer[offset+blockSize+j];
        Real* RESTRICT VCol = B.V.Buffer(0,j);
        for( int i=0; i<r; ++i )
            VCol[i] = VTRow[i*r];
    }
    // Apply Q2 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[leftPanelSize], B.Width(), &tauV[0],
      B.V.Buffer(), B.V.LDim(), &buffer[offset], blockSize );
#endif // PIVOTED_QR
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, 
  const FactorMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta,        
        FactorMatrix<std::complex<Real>,Conjugated>& B )
{
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = A.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( r, maxRank );

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
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT packedAUCol = &buffer[j*m];
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedAUCol[i] = alpha*AUCol[i];
    }
    // Copy in (beta B.U)
    for( int j=0; j<Br; ++j )
    {
        Scalar* RESTRICT packedBUCol = &buffer[(j+Ar)*m];
        const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedBUCol[i] = beta*BUCol[i];
    }
    // Copy in A.V
    for( int j=0; j<Ar; ++j )
    {
        std::memcpy
        ( &buffer[leftPanelSize+j*n], A.V.LockedBuffer(0,j), 
          n*sizeof(Scalar) );
    }
    // Copy in B.V
    for( int j=0; j<Br; ++j )
    {
        std::memcpy
        ( &buffer[leftPanelSize+(j+Ar)*n], B.V.LockedBuffer(0,j),
          n*sizeof(Scalar) );
    }

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
    const char option = ( Conjugated ? 'C' : 'T' );
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
    B.U.Resize( m, roundedRank );
    B.V.Resize( n, roundedRank );

    // Form the rounded B.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Scalar)0, B.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        const Scalar* RESTRICT UCol = &buffer[offset+j*r];
        Scalar* RESTRICT UColScaled = B.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[0], B.Height(), &tauU[0], 
      B.U.Buffer(), B.U.LDim(), &buffer[offset], blockSize );

    // Form the rounded B.V by first filling it with 
    //  | (VH_Top)^[T,H] |, and then hitting it from the left with Q2
    //  |      0         |
    Scale( (Scalar)0, B.V );
    if( Conjugated )
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[offset+blockSize+j];
            Scalar* RESTRICT VCol = B.V.Buffer(0,j);
            for( int i=0; i<r; ++i )
                VCol[i] = Conj( VHRow[i*r] );
        }
    }
    else
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[offset+blockSize+j];
            Scalar* RESTRICT VColConj = B.V.Buffer(0,j);
            for( int i=0; i<r; ++i )
                VColConj[i] = VHRow[i*r];
        }
    }
    // Apply Q2 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[leftPanelSize], B.Width(), &tauV[0],
      B.V.Buffer(), B.V.LDim(), &buffer[offset], blockSize );
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
