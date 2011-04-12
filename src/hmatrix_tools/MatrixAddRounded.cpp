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

// C :~= alpha A + beta B
//
// TODO:
// We could make use of a pivoted QLP factorization [Stewart, 1999] in order to
// get a more accurate approximation to the truncated singular value 
// decomposition in O(mnk) work for an m x n matrix and k singular vectors.
//
// See Huckaby and Chan's 2004 paper:
// "Stewart's pivoted QLP decomposition for low-rank matrices".
//
// The current implementation attempts to pack as many of the needed buffers
// into one place and minimize data movement and flops as much as possible 
// while still using BLAS3. This can be considered as most of the work towards
// almost entirely avoiding memory allocation since we could keep a sufficiently
// large buffer lying around and pack into it instead. This approach might be
// overly complicated, but rounded addition is supposedly one of the most 
// expensive parts of H-algebra.

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  Real alpha, const LowRankMatrix<Real,Conjugated>& A, 
  Real beta,  const LowRankMatrix<Real,Conjugated>& B, 
                    LowRankMatrix<Real,Conjugated>& C )
{
#ifndef RELEASE
    if( A.Height() != B.Height() || A.Width() != B.Width() )
    {
        throw std::logic_error
        ("Incompatible matrix dimensions in MatrixAddRounded");
    }
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int minDim = std::min(m,n);
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( std::min(r,minDim), maxRank );
#ifndef RELEASE
    if( Ar > minDim )
    {
        throw std::logic_error
        ("rank(A) entering MatrixAddRounded larger than minimum dimension");
    }
    if( Br > minDim )
    {
        throw std::logic_error
        ("rank(B) entering MatrixAddRounded larger than minimum dimension");
    }
#endif

    C.U.SetType( GENERAL ); C.U.Resize( m, roundedRank );
    C.V.SetType( GENERAL ); C.V.Resize( n, roundedRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        // Copy alpha A.U into the left half of C.U
        for( int j=0; j<Ar; ++j )
        {
            Real* RESTRICT CUACol = C.U.Buffer(0,j);
            const Real* RESTRICT AUCol = A.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUACol[i] = alpha*AUCol[i];
        }
        // Copy beta B.U into the right half of C.U
        for( int j=0; j<Br; ++j )
        {
            Real* RESTRICT CUBCol = C.U.Buffer(0,j+Ar);
            const Real* RESTRICT BUCol = B.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUBCol[i] = beta*BUCol[i];
        }

        // Copy A.V into the left half of C.V
        for( int j=0; j<Ar; ++j )
        {
            std::memcpy
            ( C.V.Buffer(0,j), A.V.LockedBuffer(0,j), n*sizeof(Real) );
        }
        // Copy B.V into the right half of C.V
        for( int j=0; j<Br; ++j )
        {
            std::memcpy
            ( C.V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n*sizeof(Real) );
        }

        return;
    }

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = lapack::SVDWorkSize( r, r );
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
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
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
    //  [offset+blockSize,offset+2*blockSize): V^H of R1 R2^T                 //
    //------------------------------------------------------------------------//

    // Form the rounded C.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Real)0, C.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = &buffer[offset+j*r];
        Real* RESTRICT UColScaled = C.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, 
      &buffer[0], m, &tauU[0], C.U.Buffer(), C.U.LDim(), 
      &buffer[offset], blockSize );

    // Form the rounded C.V by first filling it with 
    //  | (VT_Top)^T |, and then hitting it from the left with Q2
    //  |      0     |
    Scale( (Real)0, C.V );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = &buffer[offset+blockSize+j];
        Real* RESTRICT VCol = C.V.Buffer(0,j);
        for( int i=0; i<r; ++i )
            VCol[i] = VTRow[i*r];
    }
    // Apply Q2 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[leftPanelSize], n, &tauV[0], 
      C.V.Buffer(), C.V.LDim(), &buffer[offset], blockSize );
#endif // PIVOTED_QR
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  std::complex<Real> alpha, 
  const LowRankMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta,  
  const LowRankMatrix<std::complex<Real>,Conjugated>& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& C )
{
#ifndef RELEASE
    if( A.Height() != B.Height() || A.Width() != B.Width() )
    {
        throw std::logic_error
        ("Incompatible matrix dimensions in MatrixAddRounded");
    }
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = A.Width();
    const int minDim = std::min(m,n);
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( r, maxRank );
#ifndef RELEASE
    if( Ar > minDim )
    {
        throw std::logic_error
        ("rank(A) entering MatrixAddRounded larger than minimum dimension");
    }
    if( Br > minDim )
    {
        throw std::logic_error
        ("rank(B) entering MatrixAddRounded larger than minimum dimension");
    }
#endif

    C.U.SetType( GENERAL ); C.U.Resize( m, roundedRank );
    C.V.SetType( GENERAL ); C.V.Resize( n, roundedRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        // Copy alpha A.U into the left half of C.U
        for( int j=0; j<Ar; ++j )
        {
            Scalar* RESTRICT CUACol = C.U.Buffer(0,j);
            const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUACol[i] = alpha*AUCol[i];
        }
        // Copy beta B.U into the right half of C.U
        for( int j=0; j<Br; ++j )
        {
            Scalar* RESTRICT CUBCol = C.U.Buffer(0,j+Ar);
            const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUBCol[i] = beta*BUCol[i];
        }

        // Copy A.V into the left half of C.V
        for( int j=0; j<Ar; ++j )
        {
            std::memcpy
            ( C.V.Buffer(0,j), A.V.LockedBuffer(0,j), n*sizeof(Scalar) );
        }
        // Copy B.V into the right half of C.V
        for( int j=0; j<Br; ++j )
        {
            std::memcpy
            ( C.V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n*sizeof(Scalar) );
        }

        return;
    }

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = lapack::SVDWorkSize( r, r );;
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
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
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
    const char option = ( Conjugated ? 'C' : 'N' );
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

    // Form the rounded C.U by first filling it with 
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Scalar)0, C.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        const Scalar* RESTRICT UCol = &buffer[offset+j*r];
        Scalar* RESTRICT UColScaled = C.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[0], m, &tauU[0], 
      C.U.Buffer(), C.U.LDim(), &buffer[offset], blockSize );

    // Form the rounded C.V by first filling it with 
    //  | (VH_Top)^[T,H] |, and then hitting it from the left with Q2
    //  |      0         |
    Scale( (Scalar)0, C.V );
    if( Conjugated )
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[offset+blockSize+j];
            Scalar* RESTRICT VCol = C.V.Buffer(0,j);
            for( int i=0; i<r; ++i )
                VCol[i] = Conj( VHRow[i*r] );
        }
    }
    else
    {
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[offset+blockSize+j];
            Scalar* RESTRICT VColConj = C.V.Buffer(0,j);
            for( int i=0; i<r; ++i )
                VColConj[i] = VHRow[i*r];
        }
    }
    // Apply Q2 and use the unneeded U space for our work buffer
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[leftPanelSize], n, &tauV[0], 
      C.V.Buffer(), C.V.LDim(), &buffer[offset], blockSize );
#endif // PIVOTED_QR
}

template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  float alpha, const LowRankMatrix<float,false>& A,
  float beta,  const LowRankMatrix<float,false>& B,
                     LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  float alpha, const LowRankMatrix<float,true>& A,
  float beta,  const LowRankMatrix<float,true>& B,
                     LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  double alpha, const LowRankMatrix<double,false>& A,
  double beta,  const LowRankMatrix<double,false>& B,
                      LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  double alpha, const LowRankMatrix<double,true>& A,
  double beta,  const LowRankMatrix<double,true>& B,
                      LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  std::complex<float> alpha,
  const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,
  const LowRankMatrix<std::complex<float>,false>& B,
        LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  std::complex<float> alpha, const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,true>& B,
                                   LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  std::complex<double> alpha,
  const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,
  const LowRankMatrix<std::complex<double>,false>& B,
        LowRankMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixAddRounded
( int maxRank,
  std::complex<double> alpha,
  const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,
  const LowRankMatrix<std::complex<double>,true>& B,
        LowRankMatrix<std::complex<double>,true>& C );
